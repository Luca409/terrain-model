"""
generate_terrain.py — 3D terrain model generator
=================================================
Give it a bounding box (4 lat/lon points) and optionally a highlight point,
and it produces a print-ready 3MF, a colored GLB viewer file, and an OBJ+MTL zip.

QUICK START
-----------
1. Edit the CONFIG section below.
2. Run:  python3 generate_terrain.py
3. Find your output files in the folder you set as OUTPUT_DIR.

REQUIREMENTS
------------
pip install py3dep osmnx rasterio numpy trimesh shapely geopandas scipy \
            requests numpy-stl pyproj pynhd overturemaps mapbox-earcut pyarrow
"""

import sys, os, json, zipfile, warnings
import numpy as np
import trimesh
import rasterio
from rasterio.features import shapes as rasterio_shapes
from pyproj import Transformer
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union
from trimesh.creation import extrude_polygon
import requests, urllib.request, urllib.parse
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these values, then run the script
# ══════════════════════════════════════════════════════════════════════════════

# Bounding box corners in decimal degrees (WGS84)
# You can use Google Maps: right-click a point → "What's here?" to get lat/lon
SOUTH =  42.608741   # southernmost latitude
WEST  = -74.539130   # westernmost longitude
NORTH =  42.641067   # northernmost latitude
EAST  = -74.495196   # easternmost longitude

# Optional: a lat/lon point whose building will be highlighted RED.
# Set to None to skip (all buildings will be brown).
HIGHLIGHT_POINT = (42.6249041, -74.5171631)   # 387 Cross Hill Road, Richmondville NY

# Project name — used for output file names and folder
PROJECT_NAME = "richmondville"

# Where to save output files (folder will be created if it doesn't exist)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_" + PROJECT_NAME)

# Print scale and vertical exaggeration
SCALE      = 15000   # 1:15,000  →  ~29cm print for a 5 sq-mile area
VERT_EXAG  = 2.0     # 2× makes hills read clearly at tabletop size
BASE_MM    = 4.0     # solid base thickness in mm

# Tree density (lower = more trees, higher = fewer / faster)
# 3 = one tree per 90m real-world = dense; 5 = one per 150m = sparse
TREE_SPACING_NLCD_PX = 3

# Building height at print scale (mm)
BUILDING_HEIGHT_MM = 3.5

# ══════════════════════════════════════════════════════════════════════════════
# COLORS (R, G, B, A)
# ══════════════════════════════════════════════════════════════════════════════
C = {
    'terrain':   np.array([200, 180, 140, 255], dtype=np.uint8),
    'trees':     np.array([ 34, 120,  34, 255], dtype=np.uint8),
    'buildings': np.array([139,  90,  43, 255], dtype=np.uint8),
    'highlight': np.array([220,  30,  30, 255], dtype=np.uint8),
    'roads':     np.array([ 20,  20,  20, 255], dtype=np.uint8),
    'ponds':     np.array([ 30, 110, 200, 255], dtype=np.uint8),
    'streams':   np.array([ 60, 150, 220, 255], dtype=np.uint8),
}
COLOR_HEX = {
    'terrain':   '#C8B48C',
    'trees':     '#22781E',
    'buildings': '#8B5A2B',
    'highlight': '#DC1E1E',
    'roads':     '#141414',
    'ponds':     '#1E6EC8',
    'streams':   '#3C96DC',
}

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def mm(meters):        return meters / SCALE * 1000.0
def mm_v(meters):      return mm(meters) * VERT_EXAG
def colored(mesh, rgba):
    fc = np.tile(rgba, (len(mesh.faces), 1))
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=fc)
    return mesh


def fetch_dem(outdir):
    """Download USGS 3DEP elevation raster for the bbox."""
    import py3dep
    path = os.path.join(outdir, 'dem.tif')
    if os.path.exists(path):
        print('  [cached] dem.tif')
        return path
    print('  Downloading USGS 3DEP elevation data...')
    dem = py3dep.get_dem((WEST, SOUTH, EAST, NORTH), resolution=10)
    dem.rio.to_raster(path)
    return path


def fetch_nlcd(outdir):
    """Download NLCD 2021 land cover raster for the bbox."""
    path = os.path.join(outdir, 'nlcd.tif')
    if os.path.exists(path):
        print('  [cached] nlcd.tif')
        return path
    print('  Downloading NLCD 2021 land cover...')
    tr = Transformer.from_crs('EPSG:4326', 'EPSG:5070', always_xy=True)
    x_min, y_min = tr.transform(WEST, SOUTH)
    x_max, y_max = tr.transform(EAST, NORTH)
    params = {
        'service': 'WCS', 'version': '2.0.1', 'request': 'GetCoverage',
        'coverageId': 'NLCD_2021_Land_Cover_L48',
        'format': 'image/tiff',
        'subset': [f'X({x_min},{x_max})', f'Y({y_min},{y_max})'],
    }
    url = 'https://www.mrlc.gov/geoserver/mrlc_download/NLCD_2021_Land_Cover_L48/wcs'
    r = requests.get(url, params=params, timeout=30)
    with open(path, 'wb') as f:
        f.write(r.content)
    return path


def fetch_buildings(outdir):
    """Download building footprints from Overture Maps."""
    import overturemaps
    path = os.path.join(outdir, 'buildings.geojson')
    if os.path.exists(path):
        print('  [cached] buildings.geojson')
        return path
    print('  Downloading building footprints (Overture Maps)...')
    cli = os.path.expanduser('~/.local/bin/overturemaps')
    if not os.path.exists(cli):
        cli = 'overturemaps'
    os.system(f'{cli} download --bbox="{WEST},{SOUTH},{EAST},{NORTH}" '
              f'-f geojson --type=building -o "{path}"')
    return path


def fetch_roads(outdir):
    """Download road network from OpenStreetMap via Overpass."""
    path = os.path.join(outdir, 'roads.json')
    if os.path.exists(path):
        print('  [cached] roads.json')
        return path
    print('  Downloading roads (OpenStreetMap)...')
    query = (f'[out:json][timeout:30][bbox:{SOUTH},{WEST},{NORTH},{EAST}];'
             f'(way[highway];);out body;>;out skel qt;')
    url = 'https://overpass-api.de/api/interpreter?' + urllib.parse.urlencode({'data': query})
    req = urllib.request.Request(url, headers={'User-Agent': 'terrain3d/1.0'})
    with urllib.request.urlopen(req, timeout=35) as resp:
        data = json.loads(resp.read().decode())
    with open(path, 'w') as f:
        json.dump(data, f)
    return path


def fetch_water(outdir):
    """Download water bodies (OSM + NHD)."""
    osm_path = os.path.join(outdir, 'water_osm.json')
    nhd_path = os.path.join(outdir, 'nhd_flowlines.geojson')
    nhd_wb_path = os.path.join(outdir, 'nhd_waterbodies.geojson')

    if not os.path.exists(osm_path):
        print('  Downloading waterways (OpenStreetMap)...')
        query = (f'[out:json][timeout:30][bbox:{SOUTH},{WEST},{NORTH},{EAST}];'
                 f'(way[natural=water];way[waterway];way[natural=wetland];);'
                 f'out body;>;out skel qt;')
        url = 'https://overpass-api.de/api/interpreter?' + urllib.parse.urlencode({'data': query})
        req = urllib.request.Request(url, headers={'User-Agent': 'terrain3d/1.0'})
        with urllib.request.urlopen(req, timeout=35) as resp:
            data = json.loads(resp.read().decode())
        with open(osm_path, 'w') as f:
            json.dump(data, f)
    else:
        print('  [cached] water_osm.json')

    if not os.path.exists(nhd_path):
        print('  Downloading NHD flowlines...')
        try:
            import pynhd
            bbox = (WEST, SOUTH, EAST, NORTH)
            fl = pynhd.WaterData('nhdflowline_network').bybox(bbox)
            with open(nhd_path, 'w') as f:
                f.write(fl[['gnis_name','streamorde','geometry']].to_json())
        except Exception as e:
            print(f'    NHD flowlines unavailable: {e}')
            with open(nhd_path, 'w') as f:
                json.dump({'type':'FeatureCollection','features':[]}, f)
    else:
        print('  [cached] nhd_flowlines.geojson')

    if not os.path.exists(nhd_wb_path):
        print('  Downloading NHD waterbodies...')
        try:
            import pynhd
            bbox = (WEST, SOUTH, EAST, NORTH)
            wb = pynhd.WaterData('nhdwaterbody').bybox(bbox)
            with open(nhd_wb_path, 'w') as f:
                f.write(wb[['ftype','gnis_name','areasqkm','geometry']].to_json())
        except Exception as e:
            print(f'    NHD waterbodies unavailable: {e}')
            with open(nhd_wb_path, 'w') as f:
                json.dump({'type':'FeatureCollection','features':[]}, f)
    else:
        print('  [cached] nhd_waterbodies.geojson')

    return osm_path, nhd_path, nhd_wb_path


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_terrain(dem, R, C, step, dem_min):
    xs = np.arange(C)*mm(step); ys = (R-1-np.arange(R))*mm(step)
    XX, YY = np.meshgrid(xs, ys); ZZ = mm_v(dem - dem_min)
    top_v = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    bot_v = top_v.copy(); bot_v[:, 2] = -BASE_MM
    verts = np.vstack([top_v, bot_v]); N = R*C
    def t(r,c): return r*C+c
    def b(r,c): return N+r*C+c
    rr,cc = np.mgrid[0:R-1,0:C-1]; rr=rr.ravel(); cc=cc.ravel()
    faces = [
        np.column_stack([t(rr,cc),t(rr,cc+1),t(rr+1,cc+1)]),
        np.column_stack([t(rr,cc),t(rr+1,cc+1),t(rr+1,cc)]),
        np.column_stack([b(rr,cc),b(rr+1,cc+1),b(rr,cc+1)]),
        np.column_stack([b(rr,cc),b(rr+1,cc),b(rr+1,cc+1)]),
    ]
    def wall(ti,bi):
        n=len(ti)-1; t0=ti[:-1];t1=ti[1:];b0=bi[:-1];b1=bi[1:]
        return np.vstack([np.column_stack([t0,b0,b1]),np.column_stack([t0,b1,t1])])
    ci=np.arange(C); ri=np.arange(R)
    faces += [wall(t(0,ci[::-1]),b(0,ci[::-1])), wall(t(R-1,ci),b(R-1,ci)),
              wall(t(ri,0),b(ri,0)), wall(t(ri[::-1],C-1),b(ri[::-1],C-1))]
    mesh = trimesh.Trimesh(vertices=verts, faces=np.vstack(faces), process=True)
    trimesh.repair.fix_normals(mesh)
    return mesh


def find_highlight_building(features, tr_4326_5070, highlight_lat, highlight_lon):
    """Return index of building whose polygon contains the highlight point."""
    from shapely.geometry import Point
    pt_5070 = Point(tr_4326_5070.transform(highlight_lon, highlight_lat))
    best_i, best_d = None, float('inf')
    for i, feat in enumerate(features):
        geom = shape(feat['geometry'])
        polys = [geom] if geom.geom_type=='Polygon' else list(geom.geoms)
        for poly in polys:
            coords = [tr_4326_5070.transform(lon,lat) for lon,lat in poly.exterior.coords]
            p5070 = Polygon(coords)
            if p5070.contains(pt_5070):
                return i
            d = p5070.centroid.distance(pt_5070)
            if d < best_d:
                best_d, best_i = d, i
    print(f'  Highlight: nearest building is {best_d:.0f}m away (index {best_i})')
    return best_i


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'\n{"═"*60}')
    print(f'  Terrain model: {PROJECT_NAME}')
    print(f'  BBox: ({SOUTH},{WEST}) → ({NORTH},{EAST})')
    print(f'  Scale 1:{SCALE}  Vert exag {VERT_EXAG}×')
    print(f'  Output: {OUTPUT_DIR}')
    print(f'{"═"*60}\n')

    DEM_STEP = 2   # subsample every 2nd DEM pixel (keeps file size manageable)

    # ── Fetch data ──
    print('STEP 1/6 — Fetching data')
    dem_path       = fetch_dem(OUTPUT_DIR)
    nlcd_path      = fetch_nlcd(OUTPUT_DIR)
    bldg_path      = fetch_buildings(OUTPUT_DIR)
    roads_path     = fetch_roads(OUTPUT_DIR)
    osm_w, nhd_fl, nhd_wb = fetch_water(OUTPUT_DIR)

    # ── Load DEM ──
    print('\nSTEP 2/6 — Processing elevation data')
    with rasterio.open(dem_path) as src:
        dem_full = src.read(1).astype(np.float64)
        tf = src.transform; pxsz = tf.a; x0 = tf.c; y0 = tf.f
    dem = dem_full[::DEM_STEP, ::DEM_STEP]
    R, C = dem.shape; step = pxsz * DEM_STEP; dem_min = dem.min()
    W = (C-1)*mm(step); H = (R-1)*mm(step)
    print(f'  DEM: {R}×{C} pts, {step:.1f}m/px, '
          f'elevation {dem.min():.0f}–{dem.max():.0f}m, '
          f'print size {W:.1f}×{H:.1f}mm')

    tr = Transformer.from_crs('EPSG:4326', 'EPSG:5070', always_xy=True)

    def elev_at(wx, wy):
        c = int(np.clip((wx-x0)/step, 0, C-1))
        r = int(np.clip((y0-wy)/step, 0, R-1))
        return dem[r, c]
    def print_xy(wx, wy):
        c = (wx-x0)/step; r = (y0-wy)/step
        return c*mm(step), (R-1-r)*mm(step)
    def in_bounds(px, py, margin=1):
        return margin < px < W-margin and margin < py < H-margin

    groups = {}

    # ── Terrain ──
    print('\nSTEP 3/6 — Building terrain mesh')
    terrain = build_terrain(dem, R, C, step, dem_min)
    colored(terrain, C['terrain'])
    groups['terrain'] = [terrain]
    print(f'  {len(terrain.faces):,} faces, watertight={terrain.is_watertight}')

    # ── Trees ──
    print('\nSTEP 4/6 — Placing trees + buildings + roads + water')
    print('  Trees...')
    with rasterio.open(nlcd_path) as src:
        nlcd = src.read(1); ntf = src.transform
        nx0=ntf.c; ny0=ntf.f; npx=ntf.a; nrows,ncols=nlcd.shape
    FOREST = {41, 42, 43, 90}
    tree_list = []
    for nr in range(0, nrows, TREE_SPACING_NLCD_PX):
        for nc in range(0, ncols, TREE_SPACING_NLCD_PX):
            if nlcd[nr, nc] not in FOREST: continue
            wx = nx0+(nc+.5)*npx; wy = ny0+(nr+.5)*ntf.e
            bz = mm_v(elev_at(wx,wy)-dem_min)
            px_x, px_y = print_xy(wx, wy)
            if not in_bounds(px_x, px_y): continue
            th = 4.0*.35; ch = 4.0*.65
            trunk = trimesh.creation.cylinder(radius=0.4, height=th, sections=6)
            trunk.apply_translation([0, 0, th/2])
            canopy = trimesh.creation.cone(radius=0.8, height=ch, sections=8)
            canopy.apply_translation([0, 0, th+ch/2])
            tree = trimesh.util.concatenate([trunk, canopy])
            tree.apply_translation([px_x, px_y, bz])
            tree_list.append(tree)
    groups['trees'] = tree_list
    print(f'    {len(tree_list)} trees')

    # ── Buildings ──
    print('  Buildings...')
    with open(bldg_path) as f: geo = json.load(f)
    features = geo['features']
    highlight_idx = None
    if HIGHLIGHT_POINT:
        highlight_idx = find_highlight_building(features, tr, *HIGHLIGHT_POINT)
        print(f'    Highlight building: index {highlight_idx}')
    bldg_list = []; highlight_list = []; b_cnt = 0
    for i, feat in enumerate(features):
        geom = shape(feat['geometry'])
        polys = [geom] if geom.geom_type=='Polygon' else list(geom.geoms)
        for poly_wgs in polys:
            coords = list(poly_wgs.exterior.coords)
            pts_5070 = [tr.transform(lon, lat) for lon, lat in coords]
            if len(pts_5070) < 3: continue
            bz = mm_v(min(elev_at(x,y) for x,y in pts_5070) - dem_min)
            pp = [print_xy(x,y) for x,y in pts_5070]
            poly2d = Polygon(pp[:-1])
            if not poly2d.is_valid: poly2d = poly2d.buffer(0)
            if poly2d.area < 0.005: continue
            try:
                bld = extrude_polygon(poly2d, height=BUILDING_HEIGHT_MM)
                bld.apply_translation([0, 0, bz])
                if i == highlight_idx:
                    colored(bld, C['highlight']); highlight_list.append(bld)
                else:
                    colored(bld, C['buildings']); bldg_list.append(bld)
                b_cnt += 1
            except: pass
    groups['buildings'] = bldg_list
    groups['highlight'] = highlight_list
    print(f'    {b_cnt} buildings ({len(highlight_list)} highlighted)')

    # ── Roads ──
    print('  Roads...')
    ROAD_WIDTH = {'motorway':3.0,'trunk':2.5,'primary':2.5,'secondary':2.0,
                  'tertiary':1.8,'unclassified':1.4,'residential':1.2,
                  'service':0.9,'track':0.7,'default':1.0}
    with open(roads_path) as f: road_data = json.load(f)
    road_nodes = {e['id']:(e['lon'],e['lat'])
                  for e in road_data['elements'] if e['type']=='node'}
    road_ways  = [e for e in road_data['elements'] if e['type']=='way']
    road_list  = []
    for way in road_ways:
        hw = way.get('tags',{}).get('highway','default')
        half_w = ROAD_WIDTH.get(hw, ROAD_WIDTH['default']) / 2.0
        pts_5070 = []
        for nid in way['nodes']:
            if nid not in road_nodes: continue
            lon, lat = road_nodes[nid]
            pts_5070.append(tr.transform(lon, lat))
        if len(pts_5070) < 2: continue
        pairs = []
        for si in range(len(pts_5070)-1):
            x1,y1=pts_5070[si]; x2,y2=pts_5070[si+1]
            seg_mm = np.hypot(x2-x1,y2-y1)/SCALE*1000
            n = max(2, int(seg_mm*1.5)+1)
            for t_val in np.linspace(0,1,n,endpoint=(si==len(pts_5070)-2)):
                wx=x1+t_val*(x2-x1); wy=y1+t_val*(y2-y1)
                pxx,pxy=print_xy(wx,wy)
                if not(-2<=pxx<=W+2 and -2<=pxy<=H+2): continue
                pz=mm_v(elev_at(wx,wy)-dem_min)+0.35
                p3=np.array([pxx,pxy,pz])
                dx,dy=x2-x1,y2-y1; L=np.hypot(dx,dy)
                if L<1e-6: continue
                px_p=-dy/L*half_w; py_p=dx/L*half_w
                pairs.append((p3+np.array([px_p,py_p,0]),
                               p3-np.array([px_p,py_p,0])))
        # clip pairs to bounds together
        pairs=[p for p in pairs if in_bounds(p[0][0],p[0][1],-2) and
                                    in_bounds(p[1][0],p[1][1],-2)]
        if len(pairs)<2: continue
        Lv=np.array([p[0] for p in pairs]); Rv=np.array([p[1] for p in pairs])
        n=len(Lv); vv=np.vstack([Lv,Rv])
        ff=[]
        for i in range(n-1): ff+=[[i,n+i,n+i+1],[i,n+i+1,i+1]]
        mesh=trimesh.Trimesh(vertices=vv,faces=np.array(ff),process=False)
        colored(mesh, C['roads']); road_list.append(mesh)
    groups['roads'] = road_list
    print(f'    {len(road_list)} road segments')

    # ── Water ──
    print('  Water...')
    pond_list = []; stream_list = []

    # NHD waterbody polygons
    nhd_polys_5070 = []
    try:
        with open(nhd_wb) as f: nhd_wb_geo = json.load(f)
        for feat in nhd_wb_geo.get('features',[]):
            geom = shape(feat['geometry'])
            polys = [geom] if geom.geom_type=='Polygon' else list(geom.geoms)
            for p in polys:
                coords = [tr.transform(lon,lat) for lon,lat in p.exterior.coords]
                nhd_polys_5070.append(Polygon(coords))
    except: pass

    # NLCD class-11 vectorized
    with rasterio.open(nlcd_path) as src:
        nlcd_w=src.read(1); ntf_w=src.transform
    water_mask = (nlcd_w==11).astype(np.uint8)
    nlcd_water_polys = []
    for geom_d, val in rasterio_shapes(water_mask, mask=water_mask, transform=ntf_w):
        if val==1:
            p = shape(geom_d).buffer(15).buffer(-15).simplify(10)
            if p.is_valid and p.area>400: nlcd_water_polys.append(p)
    nhd_union = unary_union(nhd_polys_5070) if nhd_polys_5070 else None
    all_ponds = [(True,p) for p in nhd_polys_5070] + \
                [(False,p) for p in nlcd_water_polys
                 if not (nhd_union and p.intersects(nhd_union))]
    for _, poly_5070 in all_ponds:
        minx,miny,maxx,maxy = poly_5070.bounds
        sample_elevs = [elev_at(sx,sy)
                        for sx in np.arange(minx,maxx,step)
                        for sy in np.arange(miny,maxy,step)]
        water_z = mm_v((np.percentile(sample_elevs,10) if sample_elevs
                        else elev_at((minx+maxx)/2,(miny+maxy)/2)) - dem_min) + 0.25
        pp = [print_xy(x,y) for x,y in poly_5070.exterior.coords]
        pp = [p for p in pp if -2<=p[0]<=W+2 and -2<=p[1]<=H+2]
        if len(pp)<3: continue
        poly2d=Polygon(pp)
        if not poly2d.is_valid: poly2d=poly2d.buffer(0)
        if poly2d.area<0.1: continue
        try:
            slab=extrude_polygon(poly2d, height=0.15)
            slab.apply_translation([0,0,water_z])
            colored(slab, C['ponds']); pond_list.append(slab)
        except: pass

    def build_stream(coords_wgs):
        pts_5070 = [tr.transform(lon,lat) for lon,lat in coords_wgs]
        pairs=[]
        for si in range(len(pts_5070)-1):
            x1,y1=pts_5070[si]; x2,y2=pts_5070[si+1]
            seg_mm=np.hypot(x2-x1,y2-y1)/SCALE*1000
            n=max(2,int(seg_mm*1.5)+1)
            for t_val in np.linspace(0,1,n,endpoint=(si==len(pts_5070)-2)):
                wx=x1+t_val*(x2-x1); wy=y1+t_val*(y2-y1)
                pxx,pxy=print_xy(wx,wy)
                if not(-2<=pxx<=W+2 and -2<=pxy<=H+2): continue
                pz=mm_v(elev_at(wx,wy)-dem_min)+0.30
                p3=np.array([pxx,pxy,pz])
                dx,dy=x2-x1,y2-y1; L=np.hypot(dx,dy)
                if L<1e-6: continue
                px_p=-dy/L*0.45; py_p=dx/L*0.45
                pairs.append((p3+np.array([px_p,py_p,0]),
                               p3-np.array([px_p,py_p,0])))
        pairs=[p for p in pairs if -2<=p[0][0]<=W+2 and -2<=p[0][1]<=H+2]
        if len(pairs)<2: return None
        Lv=np.array([p[0] for p in pairs]); Rv=np.array([p[1] for p in pairs])
        n=len(Lv); vv=np.vstack([Lv,Rv])
        ff=[]
        for i in range(n-1): ff+=[[i,n+i,n+i+1],[i,n+i+1,i+1]]
        m=trimesh.Trimesh(vertices=vv,faces=np.array(ff),process=False)
        colored(m,C['streams']); return m

    try:
        with open(nhd_fl) as f: fl_geo=json.load(f)
        for feat in fl_geo.get('features',[]):
            geom=feat['geometry']
            lines=([geom['coordinates']] if geom['type']=='LineString'
                   else geom['coordinates'])
            for coords in lines:
                m=build_stream(coords)
                if m: stream_list.append(m)
    except: pass
    try:
        with open(osm_w) as f: osm_wd=json.load(f)
        osm_nodes_w={e['id']:(e['lon'],e['lat'])
                     for e in osm_wd['elements'] if e['type']=='node'}
        for way in [e for e in osm_wd['elements']
                    if e['type']=='way' and e['nodes'][0]!=e['nodes'][-1]]:
            coords=[(osm_nodes_w[n]) for n in way['nodes'] if n in osm_nodes_w]
            if len(coords)<2: continue
            m=build_stream(coords)
            if m: stream_list.append(m)
    except: pass

    groups['ponds']   = pond_list
    groups['streams'] = stream_list
    print(f'    {len(pond_list)} ponds, {len(stream_list)} stream segments')

    # ── Merge groups ──
    print('\nSTEP 5/6 — Merging and exporting')
    merged = {}
    for name, meshes in groups.items():
        if not meshes: continue
        m = trimesh.util.concatenate(meshes) if len(meshes)>1 else meshes[0]
        merged[name] = m
    total_faces = sum(len(m.faces) for m in merged.values())
    print(f'  Total: {total_faces:,} faces across {len(merged)} groups')

    # ── Export GLB ──
    scene = trimesh.scene.Scene()
    for name, mesh in merged.items():
        scene.add_geometry(mesh, node_name=name)
    glb_path = os.path.join(OUTPUT_DIR, f'{PROJECT_NAME}_colored.glb')
    with open(glb_path, 'wb') as f:
        f.write(scene.export(file_type='glb'))
    print(f'  GLB:  {os.path.getsize(glb_path)/1e6:.1f} MB  → {os.path.basename(glb_path)}')

    # ── Export colored 3MF ──
    group_order = [k for k in COLOR_HEX if k in merged]
    color_idx   = {n:i for i,n in enumerate(group_order)}
    COLOR_GRP   = 2
    lines = [
        "<?xml version='1.0' encoding='utf-8'?>",
        '<model unit="millimeter"'
        ' xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"'
        ' xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02"'
        ' xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06">',
        '<resources>',
        f'<m:colorgroup id="{COLOR_GRP}">',
    ]
    for n in group_order:
        lines.append(f'  <m:color color="{COLOR_HEX[n]}"/>')
    lines += ['</m:colorgroup>',
              '<object id="1" type="model" p:UUID="a1b2c3d4-0000-0000-0000-000000000001">',
              '<mesh><vertices>']
    vc = []
    for n in group_order:
        m = merged[n]; vc.append(len(m.vertices))
        for vx,vy,vz in m.vertices:
            lines.append(f'<vertex x="{vx:.4f}" y="{vy:.4f}" z="{vz:.4f}"/>')
    lines.append('</vertices><triangles>')
    v_off=0
    for gi,n in enumerate(group_order):
        m=merged[n]; ci=color_idx[n]
        for f0,f1,f2 in m.faces:
            lines.append(f'<triangle v1="{f0+v_off}" v2="{f1+v_off}" v3="{f2+v_off}"'
                         f' pid="{COLOR_GRP}" p1="{ci}"/>')
        v_off+=len(m.vertices)
    lines += ['</triangles></mesh></object></resources>',
              '<build><item objectid="1"/></build></model>']
    model_xml='\n'.join(lines)
    rels_xml=('<?xml version="1.0" encoding="UTF-8"?>'
              '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
              '<Relationship Target="/3D/3dmodel.model" Id="rel0"'
              ' Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
              '</Relationships>')
    ct_xml=('<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
            '</Types>')
    mf_path=os.path.join(OUTPUT_DIR, f'{PROJECT_NAME}_colored.3mf')
    with zipfile.ZipFile(mf_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as zf:
        zf.writestr('3D/3dmodel.model', model_xml)
        zf.writestr('_rels/.rels', rels_xml)
        zf.writestr('[Content_Types].xml', ct_xml)
    print(f'  3MF:  {os.path.getsize(mf_path)/1e6:.1f} MB  → {os.path.basename(mf_path)}')

    # ── Export OBJ+MTL zip ──
    MTL_COLORS = {
        'terrain':   (0.784,0.706,0.549), 'trees':     (0.133,0.471,0.133),
        'buildings': (0.545,0.353,0.169), 'highlight': (0.863,0.118,0.118),
        'roads':     (0.078,0.078,0.078), 'ponds':     (0.118,0.431,0.784),
        'streams':   (0.235,0.588,0.863),
    }
    obj_lines=['mtllib model.mtl','']; mtl_lines=[]; voff=0
    for name,mesh in merged.items():
        if name not in MTL_COLORS: continue
        obj_lines+=[f'g {name}',f'usemtl {name}_mat']
        for vx,vy,vz in mesh.vertices: obj_lines.append(f'v {vx:.4f} {vy:.4f} {vz:.4f}')
        for f0,f1,f2 in mesh.faces: obj_lines.append(f'f {f0+voff+1} {f1+voff+1} {f2+voff+1}')
        obj_lines.append(''); voff+=len(mesh.vertices)
        r,g,b=MTL_COLORS[name]
        mtl_lines+=[f'newmtl {name}_mat',f'Ka 0.1 0.1 0.1',
                    f'Kd {r:.3f} {g:.3f} {b:.3f}',f'Ks 0.05 0.05 0.05','']
    zip_path=os.path.join(OUTPUT_DIR,f'{PROJECT_NAME}_colored_print.zip')
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f'{PROJECT_NAME}_colored.obj', '\n'.join(obj_lines))
        zf.writestr('model.mtl', '\n'.join(mtl_lines))
    print(f'  ZIP:  {os.path.getsize(zip_path)/1e6:.1f} MB  → {os.path.basename(zip_path)}')

    # ── Export plain STL ──
    stl_parts=[terrain]+road_list+bldg_list+highlight_list+pond_list+stream_list
    stl=trimesh.util.concatenate(stl_parts)
    stl_path=os.path.join(OUTPUT_DIR,f'{PROJECT_NAME}_terrain.stl')
    stl.export(stl_path)
    print(f'  STL:  {os.path.getsize(stl_path)/1e6:.1f} MB  → {os.path.basename(stl_path)}')

    print(f'\nSTEP 6/6 — Done!')
    print(f'  Print size: {stl.extents[0]:.1f} × {stl.extents[1]:.1f} × {stl.extents[2]:.1f} mm')
    print(f'  Files saved to: {OUTPUT_DIR}')
    print(f'{"═"*60}\n')


if __name__ == '__main__':
    main()
