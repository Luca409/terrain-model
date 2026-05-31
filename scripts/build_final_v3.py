"""Final model v3 — terrain + trees + buildings + roads + water."""
import sys; sys.path.insert(0, '/sessions/gifted-keen-volta/mnt/outputs')
import numpy as np, trimesh, rasterio, json, warnings
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

C_TERRAIN =np.array([200,180,140,255],dtype=np.uint8)
C_TREE    =np.array([ 34,120, 34,255],dtype=np.uint8)
C_BUILDING=np.array([139, 90, 43,255],dtype=np.uint8)
C_TARGET  =np.array([220, 30, 30,255],dtype=np.uint8)

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
def colored(mesh,rgba):
    fc=np.tile(rgba,(len(mesh.faces),1))
    mesh.visual=trimesh.visual.ColorVisuals(mesh=mesh,face_colors=fc)
    return mesh

# ── Terrain ──
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
colored(terrain,C_TERRAIN)
print(f'  {len(terrain.faces):,} faces, watertight={terrain.is_watertight}')

# ── Trees ──
print('Placing trees...')
with rasterio.open(f'{OUTDIR}/nlcd.tif') as src:
    nlcd=src.read(1); ntf=src.transform; nx0=ntf.c; ny0=ntf.f; npx=ntf.a
FOREST={41,42,43,90}; nr_,nc_=nlcd.shape; tree_meshes=[]; cnt=0
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
        colored(tree,C_TREE); tree_meshes.append(tree); cnt+=1
print(f'  {cnt} trees')

# ── Buildings ──
print('Adding buildings...')
tr=Transformer.from_crs('EPSG:4326','EPSG:5070',always_xy=True)
with open(f'{OUTDIR}/overture_buildings.geojson') as f: geo=json.load(f)
features=geo['features']; bldg_meshes=[]; b_cnt=0
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
            col=C_TARGET if i==TARGET_BLDG_IDX else C_BUILDING
            colored(bld,col); bldg_meshes.append(bld); b_cnt+=1
            if i==TARGET_BLDG_IDX: print(f'  ★ 387 Cross Hill Rd (red)')
        except: pass
print(f'  {b_cnt} buildings')

# ── Roads ──
print('Adding roads...')
road_meshes=build_road_meshes(dem,DEM_STEP,tf.a,x0,y0,f'{OUTDIR}/roads_raw.json')
print(f'  {len(road_meshes)} road segments')

# ── Water ──
print('Adding water...')
pond_meshes, stream_meshes=build_water_meshes(dem,DEM_STEP,tf.a,x0,y0)
print(f'  {len(pond_meshes)} ponds, {len(stream_meshes)} stream segments')

# ── Export GLB ──
print('\nExporting GLB...')
scene=trimesh.scene.Scene()
scene.add_geometry(terrain,node_name='terrain')
for j,m in enumerate(tree_meshes):   scene.add_geometry(m,node_name=f'tree_{j}')
for j,m in enumerate(bldg_meshes):   scene.add_geometry(m,node_name=f'bldg_{j}')
for j,m in enumerate(road_meshes):   scene.add_geometry(m,node_name=f'road_{j}')
for j,m in enumerate(pond_meshes):   scene.add_geometry(m,node_name=f'pond_{j}')
for j,m in enumerate(stream_meshes): scene.add_geometry(m,node_name=f'stream_{j}')

out_glb=f'{OUTDIR}/richmondville_colored.glb'
with open(out_glb,'wb') as f: f.write(scene.export(file_type='glb'))

# ── Export STL ──
print('Exporting STL...')
stl_parts=[terrain]+road_meshes+bldg_meshes+pond_meshes+stream_meshes
stl=trimesh.util.concatenate(stl_parts)
out_stl=f'{OUTDIR}/richmondville_terrain.stl'
stl.export(out_stl)

import os
print(f'\n✓ GLB: {os.path.getsize(out_glb)/1e6:.1f} MB')
print(f'✓ STL: {os.path.getsize(out_stl)/1e6:.1f} MB')
print(f'  Print size: {stl.extents[0]:.1f} x {stl.extents[1]:.1f} x {stl.extents[2]:.1f} mm')
print(f'  Trees:{cnt}  Buildings:{b_cnt}  Roads:{len(road_meshes)}  Ponds:{len(pond_meshes)}  Streams:{len(stream_meshes)}')
