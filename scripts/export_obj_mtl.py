"""
Export terrain model as OBJ + MTL (zipped) for full-color 3D printing services.
Merges all meshes into 7 named material groups for a clean, upload-ready file.
"""
import sys; sys.path.insert(0, '/sessions/gifted-keen-volta/mnt/outputs')
import numpy as np, trimesh, rasterio, json, zipfile, warnings
from pyproj import Transformer
from shapely.geometry import shape, Polygon
from trimesh.creation import extrude_polygon
from build_roads import build_road_meshes
from build_water import build_water_meshes
warnings.filterwarnings('ignore')

SCALE=15000; VERT_EXAG=2.0; DEM_STEP=2; BASE_MM=4.0
TREE_H=4.0; TREE_R=0.8; TREE_TRUNK=0.4; TREE_SPACING=3
BLDG_H=3.5
OUTDIR='/sessions/gifted-keen-volta/mnt/outputs'
TARGET_BLDG_IDX=80

def mm(m): return m/SCALE*1000.0
def mm_v(m): return mm(m)*VERT_EXAG

# ── Load DEM ──
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

# ── Build all geometry, grouped by material ──
groups = {}   # name -> list of trimesh meshes

# Terrain
print('Building terrain...')
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
groups['terrain'] = [terrain]

# Trees
print('Placing trees...')
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
groups['trees'] = tree_list
print(f'  {cnt} trees')

# Buildings
print('Adding buildings...')
tr=Transformer.from_crs('EPSG:4326','EPSG:5070',always_xy=True)
with open(f'{OUTDIR}/overture_buildings.geojson') as f: geo=json.load(f)
features=geo['features']; bldg_list=[]; target_list=[]; b_cnt=0
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
            bld=extrude_polygon(poly2d,height=BLDG_H)
            bld.apply_translation([0,0,bz])
            if i==TARGET_BLDG_IDX:
                target_list.append(bld); print(f'  ★ 387 Cross Hill Rd')
            else:
                bldg_list.append(bld)
            b_cnt+=1
        except: pass
groups['buildings']          = bldg_list
groups['387_cross_hill_road'] = target_list
print(f'  {b_cnt} buildings')

# Roads
print('Adding roads...')
road_list=build_road_meshes(dem,DEM_STEP,tf.a,x0,y0,f'{OUTDIR}/roads_raw.json')
groups['roads'] = road_list
print(f'  {len(road_list)} road segments')

# Water
print('Adding water...')
pond_list, stream_list=build_water_meshes(dem,DEM_STEP,tf.a,x0,y0)
groups['ponds']   = pond_list
groups['streams'] = stream_list
print(f'  {len(pond_list)} ponds, {len(stream_list)} stream segments')

# ── Merge each group into one mesh ──
print('\nMerging groups...')
merged = {}
for name, meshes in groups.items():
    if not meshes: continue
    m = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    merged[name] = m
    print(f'  {name}: {len(m.faces):,} faces')

# ── Write OBJ + MTL ──
# Material colors (R G B normalized 0-1)
MATERIALS = {
    'terrain':              (0.784, 0.706, 0.549),   # warm tan
    'trees':                (0.133, 0.471, 0.133),   # forest green
    'buildings':            (0.545, 0.353, 0.169),   # saddle brown
    '387_cross_hill_road':  (0.863, 0.118, 0.118),   # red
    'roads':                (0.078, 0.078, 0.078),   # near black
    'ponds':                (0.118, 0.431, 0.784),   # deep blue
    'streams':              (0.235, 0.588, 0.863),   # lighter blue
}

print('\nWriting OBJ + MTL...')
obj_lines = ['# Richmondville NY Terrain Model — 5 sq mi centered on 387 Cross Hill Road',
             '# Scale 1:15,000 | Vertical exaggeration 2x | USGS 3DEP elevation',
             '# Buildings: Overture Maps/Microsoft | Trees: NLCD | Roads+Water: OSM/NHD',
             'mtllib richmondville_colored.mtl', '']

mtl_lines = ['# Material library — Richmondville terrain model', '']

vert_offset = 0
for name, mesh in merged.items():
    if name not in MATERIALS: continue
    v = mesh.vertices
    f = mesh.faces

    obj_lines.append(f'# --- {name} ---')
    obj_lines.append(f'g {name}')
    obj_lines.append(f'usemtl {name}_mat')

    for vx, vy, vz in v:
        obj_lines.append(f'v {vx:.4f} {vy:.4f} {vz:.4f}')

    for f0, f1, f2 in f:
        i0 = f0 + vert_offset + 1   # OBJ is 1-indexed
        i1 = f1 + vert_offset + 1
        i2 = f2 + vert_offset + 1
        obj_lines.append(f'f {i0} {i1} {i2}')

    obj_lines.append('')
    vert_offset += len(v)

    # MTL entry
    r, g, b = MATERIALS[name]
    mtl_lines += [
        f'newmtl {name}_mat',
        f'Ka 0.100 0.100 0.100',
        f'Kd {r:.3f} {g:.3f} {b:.3f}',
        f'Ks 0.050 0.050 0.050',
        f'Ns 10.0',
        f'd 1.0',
        '',
    ]

obj_text = '\n'.join(obj_lines)
mtl_text = '\n'.join(mtl_lines)

obj_path = f'{OUTDIR}/richmondville_colored.obj'
mtl_path = f'{OUTDIR}/richmondville_colored.mtl'
zip_path = f'{OUTDIR}/richmondville_colored_print.zip'

with open(obj_path, 'w') as f: f.write(obj_text)
with open(mtl_path, 'w') as f: f.write(mtl_text)

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(obj_path, 'richmondville_colored.obj')
    zf.write(mtl_path, 'richmondville_colored.mtl')

import os
obj_mb  = os.path.getsize(obj_path)/1e6
zip_mb  = os.path.getsize(zip_path)/1e6
print(f'\n✓ OBJ:  {obj_mb:.1f} MB  ({vert_offset:,} total vertices)')
print(f'✓ ZIP:  {zip_mb:.1f} MB  → richmondville_colored_print.zip')
print(f'\nMaterial groups:')
for name in merged: print(f'  {name}_mat')
print('\nReady to upload to i.materialise, Shapeways, or WhiteClouds.')
