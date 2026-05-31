"""
Write a colored 3MF using the 3MF Materials & Properties extension.
Each of the 7 groups gets a solid color; per-triangle color references
are written into the triangle list so the color is embedded in the file.
"""
import sys; sys.path.insert(0, '/sessions/gifted-keen-volta/mnt/outputs')
import numpy as np, trimesh, rasterio, json, zipfile, io, warnings
from pyproj import Transformer
from shapely.geometry import shape, Polygon
from trimesh.creation import extrude_polygon
from build_roads import build_road_meshes
from build_water import build_water_meshes
warnings.filterwarnings('ignore')

SCALE=15000; VERT_EXAG=2.0; DEM_STEP=2; BASE_MM=4.0
TREE_H=4.0; TREE_R=0.8; TREE_TRUNK=0.4; TREE_SPACING=3
BLDG_H=3.5; OUTDIR='/sessions/gifted-keen-volta/mnt/outputs'
TARGET_BLDG_IDX=80

def mm(m): return m/SCALE*1000.0
def mm_v(m): return mm(m)*VERT_EXAG

with rasterio.open(f'{OUTDIR}/dem.tif') as src:
    dem_full=src.read(1).astype(np.float64)
    tf=src.transform; pxsz=tf.a; x0=tf.c; y0=tf.f
dem=dem_full[::DEM_STEP,::DEM_STEP]; R,C=dem.shape; step=pxsz*DEM_STEP; dem_min=dem.min()
W=(C-1)*mm(step); H=(R-1)*mm(step)

def elev_at(wx,wy):
    c=int(np.clip((wx-x0)/step,0,C-1)); r=int(np.clip((y0-wy)/step,0,R-1))
    return dem[r,c]
def print_xy(wx,wy):
    c=(wx-x0)/step; r=(y0-wy)/step
    return c*mm(step),(R-1-r)*mm(step)

# ── Rebuild all groups (same as before) ──
groups = {}

print('Building geometry...')

# Terrain
xs=np.arange(C)*mm(step); ys=(R-1-np.arange(R))*mm(step)
XX,YY=np.meshgrid(xs,ys); ZZ=mm_v(dem-dem_min)
top_v=np.column_stack([XX.ravel(),YY.ravel(),ZZ.ravel()])
bot_v=top_v.copy(); bot_v[:,2]=-BASE_MM
verts=np.vstack([top_v,bot_v]); N=R*C
def t(r,c): return r*C+c
def b(r,c): return N+r*C+c
rr,cc=np.mgrid[0:R-1,0:C-1]; rr=rr.ravel(); cc=cc.ravel()
faces=[]
faces.append(np.column_stack([t(rr,cc),t(rr,cc+1),t(rr+1,cc+1)]))
faces.append(np.column_stack([t(rr,cc),t(rr+1,cc+1),t(rr+1,cc)]))
faces.append(np.column_stack([b(rr,cc),b(rr+1,cc+1),b(rr,cc+1)]))
faces.append(np.column_stack([b(rr,cc),b(rr+1,cc),b(rr+1,cc+1)]))
def wall(ti,bi):
    n=len(ti)-1; t0=ti[:-1];t1=ti[1:];b0=bi[:-1];b1=bi[1:]
    return np.vstack([np.column_stack([t0,b0,b1]),np.column_stack([t0,b1,t1])])
ci=np.arange(C); ri=np.arange(R)
faces+=[wall(t(0,ci[::-1]),b(0,ci[::-1])),wall(t(R-1,ci),b(R-1,ci)),
        wall(t(ri,0),b(ri,0)),wall(t(ri[::-1],C-1),b(ri[::-1],C-1))]
terrain=trimesh.Trimesh(vertices=verts,faces=np.vstack(faces),process=True)
trimesh.repair.fix_normals(terrain)
groups['terrain']=[terrain]

with rasterio.open(f'{OUTDIR}/nlcd.tif') as src:
    nlcd=src.read(1); ntf=src.transform; nx0=ntf.c; ny0=ntf.f; npx=ntf.a
FOREST={41,42,43,90}; nr_,nc_=nlcd.shape; tree_list=[]; cnt=0
for nr in range(0,nr_,TREE_SPACING):
    for nc in range(0,nc_,TREE_SPACING):
        if nlcd[nr,nc] not in FOREST: continue
        wx=nx0+(nc+.5)*npx; wy=ny0+(nr+.5)*ntf.e
        bz=mm_v(elev_at(wx,wy)-dem_min); px_x,px_y=print_xy(wx,wy)
        if not(1<px_x<W-1 and 1<px_y<H-1): continue
        th=TREE_H*.35; ch=TREE_H*.65
        trunk=trimesh.creation.cylinder(radius=TREE_TRUNK,height=th,sections=6)
        trunk.apply_translation([0,0,th/2])
        canopy=trimesh.creation.cone(radius=TREE_R,height=ch,sections=8)
        canopy.apply_translation([0,0,th+ch/2])
        tree=trimesh.util.concatenate([trunk,canopy])
        tree.apply_translation([px_x,px_y,bz])
        tree_list.append(tree); cnt+=1
groups['trees']=tree_list
print(f'  {cnt} trees')

tr=Transformer.from_crs('EPSG:4326','EPSG:5070',always_xy=True)
with open(f'{OUTDIR}/overture_buildings.geojson') as f: geo=json.load(f)
features=geo['features']; bldg_list=[]; target_list=[]
for i,feat in enumerate(features):
    geom=shape(feat['geometry'])
    polys=[geom] if geom.geom_type=='Polygon' else list(geom.geoms)
    for poly_wgs in polys:
        coords=list(poly_wgs.exterior.coords)
        pts_5070=[(tr.transform(lon,lat)) for lon,lat in coords]
        if len(pts_5070)<3: continue
        bz=mm_v(min(elev_at(x,y) for x,y in pts_5070)-dem_min)
        pp=[print_xy(x,y) for x,y in pts_5070]
        poly2d=Polygon(pp[:-1])
        if not poly2d.is_valid: poly2d=poly2d.buffer(0)
        if poly2d.area<0.005: continue
        try:
            bld=extrude_polygon(poly2d,height=BLDG_H); bld.apply_translation([0,0,bz])
            (target_list if i==TARGET_BLDG_IDX else bldg_list).append(bld)
        except: pass
groups['buildings']=bldg_list
groups['387_cross_hill_road']=target_list

road_list=build_road_meshes(dem,DEM_STEP,tf.a,x0,y0,f'{OUTDIR}/roads_raw.json')
groups['roads']=road_list

pond_list,stream_list=build_water_meshes(dem,DEM_STEP,tf.a,x0,y0)
groups['ponds']=pond_list
groups['streams']=stream_list

# Merge each group
print('Merging groups...')
merged={}
for name,meshes in groups.items():
    if not meshes: continue
    m=trimesh.util.concatenate(meshes) if len(meshes)>1 else meshes[0]
    merged[name]=m
    print(f'  {name}: {len(m.faces):,} faces')

# ── Write colored 3MF ──
# Color index map (same order used in colorgroup)
COLOR_HEX = {
    'terrain':             '#C8B48C',
    'trees':               '#22781E',
    'buildings':           '#8B5A2B',
    '387_cross_hill_road': '#DC1E1E',
    'roads':               '#141414',
    'ponds':               '#1E6EC8',
    'streams':             '#3C96DC',
}
GROUP_ORDER = [k for k in COLOR_HEX if k in merged]
color_idx = {name: i for i, name in enumerate(GROUP_ORDER)}
COLOR_GROUP_ID = 2   # resource ID for the colorgroup

print('\nWriting colored 3MF XML...')

# Build XML efficiently using a list of strings
lines = []
lines.append("<?xml version='1.0' encoding='utf-8'?>")
lines.append('<model unit="millimeter"'
             ' xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"'
             ' xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02"'
             ' xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06">')
lines.append('<resources>')

# Color group
lines.append(f'<m:colorgroup id="{COLOR_GROUP_ID}">')
for name in GROUP_ORDER:
    lines.append(f'  <m:color color="{COLOR_HEX[name]}"/>')
lines.append('</m:colorgroup>')

# Single object containing all geometry
lines.append('<object id="1" type="model" p:UUID="a1b2c3d4-0000-0000-0000-000000000001">')
lines.append('<mesh>')
lines.append('<vertices>')

# Write all vertices (accumulated offset tracked separately)
vert_counts = []
face_counts  = []
for name in GROUP_ORDER:
    m = merged[name]
    vert_counts.append(len(m.vertices))
    face_counts.append(len(m.faces))
    for vx, vy, vz in m.vertices:
        lines.append(f'<vertex x="{vx:.4f}" y="{vy:.4f}" z="{vz:.4f}"/>')

lines.append('</vertices>')
lines.append('<triangles>')

v_offset = 0
for gi, name in enumerate(GROUP_ORDER):
    m = merged[name]
    ci = color_idx[name]
    for f0, f1, f2 in m.faces:
        i0 = f0 + v_offset
        i1 = f1 + v_offset
        i2 = f2 + v_offset
        lines.append(f'<triangle v1="{i0}" v2="{i1}" v3="{i2}" pid="{COLOR_GROUP_ID}" p1="{ci}"/>')
    v_offset += len(m.vertices)

lines.append('</triangles>')
lines.append('</mesh>')
lines.append('</object>')
lines.append('</resources>')
lines.append('<build>')
lines.append('<item objectid="1"/>')
lines.append('</build>')
lines.append('</model>')

model_xml = '\n'.join(lines)
print(f'  XML size: {len(model_xml)/1e6:.1f} MB, {sum(face_counts):,} triangles')

# Pack into 3MF zip
rels_xml = ('<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Target="/3D/3dmodel.model" Id="rel0"'
            ' Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
            '</Relationships>')

content_types = ('<?xml version="1.0" encoding="UTF-8"?>'
                 '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                 '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                 '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
                 '</Types>')

out_path = f'{OUTDIR}/richmondville_colored.3mf'
with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    zf.writestr('3D/3dmodel.model',  model_xml)
    zf.writestr('_rels/.rels',       rels_xml)
    zf.writestr('[Content_Types].xml', content_types)

import os
sz = os.path.getsize(out_path)/1e6
print(f'\n✓ Colored 3MF: {out_path}')
print(f'  File size: {sz:.1f} MB')
print(f'  Colors embedded: {len(GROUP_ORDER)} material groups')
for name in GROUP_ORDER:
    print(f'    {COLOR_HEX[name]}  {name}')
