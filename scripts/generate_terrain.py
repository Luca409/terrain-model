"""
generate_terrain.py — 3D terrain model generator
=================================================
Give it a bounding box (4 lat/lon points) and optionally a highlight point,
and it produces a print-ready 3MF, a colored GLB viewer file, and an OBJ+MTL zip.

QUICK START
-----------
1. Pick a region in places.json (or add a new one).
2. Run:  python3 generate_terrain.py --place west-fulton-ny
   (clears output_<place>/ first; use --keep-cache to skip re-downloading)
3. Find output in output_<place-name>/

REQUIREMENTS
------------
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
"""

import sys, os, json, zipfile, warnings, ssl, subprocess, shutil, math
import certifi
import numpy as np
from rasterio.transform import rowcol
import trimesh
import rasterio
from rasterio.features import shapes as rasterio_shapes
from pyproj import Transformer
from shapely.geometry import shape, Polygon, MultiPolygon, LineString, box
from shapely.ops import unary_union
from trimesh.creation import extrude_polygon
import requests, urllib.request, urllib.parse
warnings.filterwarnings('ignore')

# macOS python.org builds often lack CA certs; use certifi for all HTTPS.
_SSL_CA = certifi.where()
os.environ.setdefault('SSL_CERT_FILE', _SSL_CA)
os.environ.setdefault('REQUESTS_CA_BUNDLE', _SSL_CA)
_SSL_CTX = ssl.create_default_context(cafile=_SSL_CA)


def https_open(req, timeout=35):
    return urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — bbox/highlight come from places.json; edit globals below as needed
# ══════════════════════════════════════════════════════════════════════════════

PLACES_FILE = os.path.join(os.path.dirname(__file__), "..", "places.json")
DEFAULT_PLACE = "west-fulton-ny"

# Set by load_place() before main()
SOUTH = WEST = NORTH = EAST = None
HIGHLIGHT_POINT = None
PLACE_ID = None
PLACE_DESCRIPTION = None
PROJECT_NAME = None
OUTPUT_DIR = None

# Print scale (horizontal and vertical use the same ratio)
SCALE      = 15840   # 1:15,840  →  3 miles = 1 foot (15,840 ft per ft)
VERT_EXAG  = 1.0     # 1.0 = true vertical scale (matches horizontal)
BASE_MM    = 4.0     # solid base thickness in mm, below ELEV_REF_M

# Elevation (meters) that maps to print Z=0. Fixed at sea level (0) so every
# model shares the same vertical datum — tiles stay aligned if the map is
# expanded later. Real terrain sits at its true scaled height above this, which
# makes models thicker than if Z=0 were the local minimum elevation.
ELEV_REF_M = 0.0

# NLCD land-cover classes treated as forest
FOREST_CLASSES = {41, 42, 43, 90}  # deciduous, evergreen, mixed, woody wetlands

# False: white terrain base, no forest coloring or tree meshes (default)
# True:  place 3D tree meshes on NLCD forest pixels
INCLUDE_TREES = False
TREE_SPACING_NLCD_PX = 3   # NLCD pixels between trees (30m/px → ~90m spacing)

# Building height at print scale (mm)
BUILDING_HEIGHT_MM = 3.5

# ══════════════════════════════════════════════════════════════════════════════
# COLORS (R, G, B, A)
# ══════════════════════════════════════════════════════════════════════════════
BASE_RGBA = np.array([255, 255, 255, 255], dtype=np.uint8)
COLORS = {
    'terrain':   BASE_RGBA,
    'forest':    np.array([ 34, 120,  34, 255], dtype=np.uint8),
    'trees':     np.array([ 34, 120,  34, 255], dtype=np.uint8),
    'buildings': np.array([139,  90,  43, 255], dtype=np.uint8),
    'highlight': BASE_RGBA,
    'roads':     np.array([ 20,  20,  20, 255], dtype=np.uint8),
    'ponds':     np.array([ 30, 110, 200, 255], dtype=np.uint8),
    'streams':   np.array([ 60, 150, 220, 255], dtype=np.uint8),
}
COLOR_HEX = {
    'terrain':   '#FFFFFF',
    'forest':    '#22781E',
    'trees':     '#22781E',
    'buildings': '#8B5A2B',
    'highlight': '#FFFFFF',
    'roads':     '#141414',
    'ponds':     '#1E6EC8',
    'streams':   '#3C96DC',
}

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def mm(meters):        return meters / SCALE * 1000.0
def mm_v(meters):      return mm(meters) * VERT_EXAG


def load_place(place_id):
    """Load bbox and metadata from places.json into module-level config."""
    global SOUTH, WEST, NORTH, EAST, HIGHLIGHT_POINT
    global PLACE_ID, PLACE_DESCRIPTION, PROJECT_NAME, OUTPUT_DIR

    with open(PLACES_FILE, encoding='utf-8') as f:
        places = json.load(f)
    if place_id not in places:
        names = ', '.join(sorted(places))
        raise SystemExit(f'Unknown place "{place_id}". Available: {names}')

    entry = places[place_id]
    bbox = entry['bbox']
    SOUTH = bbox['south']
    WEST = bbox['west']
    NORTH = bbox['north']
    EAST = bbox['east']

    hp = entry.get('highlight_point')
    HIGHLIGHT_POINT = tuple(hp) if hp else None

    PLACE_ID = place_id
    PLACE_DESCRIPTION = entry.get('description', '')
    PROJECT_NAME = place_id
    OUTPUT_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "output_" + place_id))


def prepare_output_dir(keep_cache=False):
    """Remove prior output for this place so downloads match the current bbox."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if not OUTPUT_DIR.startswith(root) or not os.path.basename(OUTPUT_DIR).startswith("output_"):
        raise SystemExit(f'Unsafe output path, refusing to delete: {OUTPUT_DIR}')
    if keep_cache:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        return
    if os.path.isdir(OUTPUT_DIR):
        print(f'  Clearing {OUTPUT_DIR}')
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


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
    venv_cli = os.path.join(os.path.dirname(sys.executable), 'overturemaps')
    cli = venv_cli if os.path.exists(venv_cli) else shutil.which('overturemaps') or 'overturemaps'
    result = subprocess.run(
        [cli, 'download', f'--bbox={WEST},{SOUTH},{EAST},{NORTH}',
         '-f', 'geojson', '--type=building', '-o', path],
        env=os.environ, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f'overturemaps download failed:\n{result.stderr or result.stdout}')
    return path


def fetch_roads(outdir):
    """Download road network from OpenStreetMap via Overpass."""
    path = os.path.join(outdir, 'roads.json')
    if os.path.exists(path):
        print('  [cached] roads.json')
        return path
    print('  Downloading roads (OpenStreetMap)...')
    query = (f'[out:json][timeout:180][bbox:{SOUTH},{WEST},{NORTH},{EAST}];'
             f'(way[highway];);out body;>;out skel qt;')
    url = 'https://overpass-api.de/api/interpreter?' + urllib.parse.urlencode({'data': query})
    req = urllib.request.Request(url, headers={'User-Agent': 'terrain3d/1.0'})
    with https_open(req, timeout=35) as resp:
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
        query = (f'[out:json][timeout:180][bbox:{SOUTH},{WEST},{NORTH},{EAST}];'
                 f'(way[natural=water];way[waterway];way[natural=wetland];);'
                 f'out body;>;out skel qt;')
        url = 'https://overpass-api.de/api/interpreter?' + urllib.parse.urlencode({'data': query})
        req = urllib.request.Request(url, headers={'User-Agent': 'terrain3d/1.0'})
        with https_open(req, timeout=35) as resp:
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

def bbox_print_size_mm():
    """Print width/height from geographic bbox at SCALE (not DEM pixel extent)."""
    lat_c = math.radians((SOUTH + NORTH) / 2)
    w_m = (EAST - WEST) * 111_320 * math.cos(lat_c)
    h_m = (NORTH - SOUTH) * 111_320
    return mm(w_m), mm(h_m)


def sample_dem_elev(dem_arr, transform, lon, lat, tr):
    """Sample elevation from DEM at a WGS84 point (clamps to raster edges)."""
    wx, wy = tr.transform(lon, lat)
    r, c = rowcol(transform, wx, wy)
    r = int(np.clip(r, 0, dem_arr.shape[0] - 1))
    c = int(np.clip(c, 0, dem_arr.shape[1] - 1))
    return float(dem_arr[r, c])


def make_bbox_lonlat_grid(R, C):
    """Lon/lat for each DEM grid row (r=0 north) and column."""
    lons = np.linspace(WEST, EAST, C)
    lats = np.linspace(NORTH, SOUTH, R)
    return lons, lats


def print_xy_ll(lon, lat, W, H):
    """Map WGS84 to print mm; bbox corners -> (0,0), (W,0), (0,H), (W,H)."""
    px = (lon - WEST) / (EAST - WEST) * W
    py = (lat - SOUTH) / (NORTH - SOUTH) * H
    return px, py


def lonlat_from_print(px, py, W, H):
    lon = WEST + px / W * (EAST - WEST)
    lat = SOUTH + py / H * (NORTH - SOUTH)
    return lon, lat


def sample_dem_on_bbox_grid(dem_arr, transform, lons, lats, tr):
    """Sample DEM onto bbox-aligned lon/lat grid (R rows, C cols)."""
    R, C = len(lats), len(lons)
    grid = np.zeros((R, C), dtype=np.float64)
    for r in range(R):
        for c in range(C):
            grid[r, c] = sample_dem_elev(dem_arr, transform, lons[c], lats[r], tr)
    return grid


def forest_mask_from_nlcd(nlcd_path, lons, lats, tr_to_5070):
    """Return (R, C) bool mask: True where NLCD land cover is forest."""
    with rasterio.open(nlcd_path) as src:
        nlcd = src.read(1)
        ntf = src.transform
    tr_inv = Transformer.from_crs('EPSG:5070', 'EPSG:4326', always_xy=True)
    R, C = len(lats), len(lons)
    mask = np.zeros((R, C), dtype=bool)
    for r in range(R):
        for c in range(C):
            wx, wy = tr_to_5070.transform(lons[c], lats[r])
            nc = int(np.clip((wx - ntf.c) / ntf.a, 0, nlcd.shape[1] - 1))
            nr = int(np.clip((ntf.f - wy) / abs(ntf.e), 0, nlcd.shape[0] - 1))
            mask[r, c] = nlcd[nr, nc] in FOREST_CLASSES
    return mask


def build_terrain_grid(XX, YY, ZZ):
    """Build terrain mesh from bbox-aligned print-coordinate grids."""
    R, C = XX.shape
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


ROAD_OFFSET_MM = 0.35
ROAD_SAMPLES_PER_MM = 1.5


def _centerline_segments_in_bbox(center, W, H, clip_to_bbox):
    """Return LineString piece(s) for a centerline, clipped to the terrain rectangle."""
    line = LineString(center)
    if line.length < 0.01:
        return []
    if not clip_to_bbox:
        return [line]
    clipped = line.intersection(box(0, 0, W, H))
    if clipped.is_empty:
        return []
    if clipped.geom_type == 'LineString':
        return [clipped] if clipped.length >= 0.01 else []
    if clipped.geom_type == 'MultiLineString':
        return [g for g in clipped.geoms if g.length >= 0.01]
    return []


def _clamp_ribbon_vertex(p, W, H, elev_at_ll, z_ref, z_offset):
    """Pin a ribbon corner to the terrain edge and match elevation there."""
    q = np.asarray(p, dtype=float).copy()
    q[0] = float(np.clip(q[0], 0.0, W))
    q[1] = float(np.clip(q[1], 0.0, H))
    lon, lat = lonlat_from_print(q[0], q[1], W, H)
    q[2] = mm_v(elev_at_ll(lon, lat) - z_ref) + z_offset
    return q


def build_ribbon_pairs(pts_ll, half_w, W, H, tr, elev_at_ll, z_ref, z_offset,
                       clip_to_bbox=False):
    """Build ribbon vertex pairs along an OSM way in bbox-aligned print space."""
    center = []
    for si in range(len(pts_ll) - 1):
        lon1, lat1 = pts_ll[si]
        lon2, lat2 = pts_ll[si + 1]
        x1, y1 = tr.transform(lon1, lat1)
        x2, y2 = tr.transform(lon2, lat2)
        seg_mm = np.hypot(x2 - x1, y2 - y1) / SCALE * 1000
        n = max(2, int(seg_mm * ROAD_SAMPLES_PER_MM) + 1)
        for t in np.linspace(0, 1, n, endpoint=(si == len(pts_ll) - 2)):
            lon = lon1 + t * (lon2 - lon1)
            lat = lat1 + t * (lat2 - lat1)
            center.append(print_xy_ll(lon, lat, W, H))
    if len(center) < 2:
        center = [print_xy_ll(lon, lat, W, H) for lon, lat in pts_ll]
    if len(center) < 2:
        return []

    pairs = []
    for line in _centerline_segments_in_bbox(center, W, H, clip_to_bbox):
        n = max(2, int(line.length * ROAD_SAMPLES_PER_MM) + 1)
        for i, d in enumerate(np.linspace(0, line.length, n)):
            pt = line.interpolate(d)
            pxx, pxy = pt.x, pt.y
            lon, lat = lonlat_from_print(pxx, pxy, W, H)
            pz = mm_v(elev_at_ll(lon, lat) - z_ref) + z_offset
            d2 = min(d + 0.5, line.length)
            pt2 = line.interpolate(d2)
            dx, dy = pt2.x - pt.x, pt2.y - pt.y
            length = np.hypot(dx, dy)
            if length < 1e-9:
                if i == 0:
                    continue
                pt0 = line.interpolate(max(0, d - 0.5))
                dx, dy = pt.x - pt0.x, pt.y - pt0.y
                length = np.hypot(dx, dy)
                if length < 1e-9:
                    continue
            px_p = -dy / length * half_w
            py_p = dx / length * half_w
            p3 = np.array([pxx, pxy, pz])
            left = p3 + np.array([px_p, py_p, 0])
            right = p3 - np.array([px_p, py_p, 0])
            if clip_to_bbox:
                left = _clamp_ribbon_vertex(left, W, H, elev_at_ll, z_ref, z_offset)
                right = _clamp_ribbon_vertex(right, W, H, elev_at_ll, z_ref, z_offset)
            pairs.append((left, right))
    return pairs


def ribbon_mesh_from_pairs(pairs):
    """Turn ribbon pairs into a triangle mesh (None if degenerate)."""
    if len(pairs) < 2:
        return None
    lv = np.array([p[0] for p in pairs])
    rv = np.array([p[1] for p in pairs])
    n = len(lv)
    vv = np.vstack([lv, rv])
    faces = []
    for i in range(n - 1):
        faces += [[i, n + i, n + i + 1], [i, n + i + 1, i + 1]]
    return trimesh.Trimesh(vertices=vv, faces=np.array(faces), process=False)


def place_trees(nlcd_path, spacing, elev_at_ll, print_xy_ll_fn, z_ref, W, H, tr):
    """Place 3D tree meshes on NLCD forest pixels."""
    with rasterio.open(nlcd_path) as src:
        nlcd = src.read(1)
        ntf = src.transform
    tr_inv = Transformer.from_crs('EPSG:5070', 'EPSG:4326', always_xy=True)
    nrows, ncols = nlcd.shape
    tree_list = []
    for nr in range(0, nrows, spacing):
        for nc in range(0, ncols, spacing):
            if nlcd[nr, nc] not in FOREST_CLASSES:
                continue
            wx = ntf.c + (nc + 0.5) * ntf.a
            wy = ntf.f + (nr + 0.5) * ntf.e
            lon, lat = tr_inv.transform(wx, wy)
            px_x, px_y = print_xy_ll_fn(lon, lat, W, H)
            bz = mm_v(elev_at_ll(lon, lat) - z_ref)
            th, ch = 4.0 * 0.35, 4.0 * 0.65
            trunk = trimesh.creation.cylinder(radius=0.4, height=th, sections=6)
            trunk.apply_translation([0, 0, th / 2])
            canopy = trimesh.creation.cone(radius=0.8, height=ch, sections=8)
            canopy.apply_translation([0, 0, th + ch / 2])
            tree = trimesh.util.concatenate([trunk, canopy])
            tree.apply_translation([px_x, px_y, bz])
            tree_list.append(colored(tree, COLORS['trees']))
    return tree_list


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

def main(keep_cache=False):
    prepare_output_dir(keep_cache=keep_cache)
    print(f'\n{"═"*60}')
    print(f'  Place: {PLACE_ID}')
    if PLACE_DESCRIPTION:
        print(f'  {PLACE_DESCRIPTION}')
    print(f'  BBox: ({SOUTH},{WEST}) → ({NORTH},{EAST})')
    print(f'  Scale 1:{SCALE}  Vert exag {VERT_EXAG}×  Trees: {INCLUDE_TREES}')
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
        dem_tf = src.transform
    dem = dem_full[::DEM_STEP, ::DEM_STEP]
    R, C = dem.shape
    step = abs(dem_tf.a) * DEM_STEP
    z_ref = ELEV_REF_M   # elevation (m) mapped to print Z=0 (sea level), fixed for tile alignment
    W, H = bbox_print_size_mm()
    lons, lats = make_bbox_lonlat_grid(R, C)
    tr = Transformer.from_crs('EPSG:4326', 'EPSG:5070', always_xy=True)
    dem_bbox = sample_dem_on_bbox_grid(dem_full, dem_tf, lons, lats, tr)
    print(f'  DEM: {R}×{C} pts, {step:.1f}m/px, '
          f'elevation {dem_bbox.min():.0f}–{dem_bbox.max():.0f}m, '
          f'print size {W:.1f}×{H:.1f}mm (bbox-aligned)')
    print(f'  Z datum: {z_ref:.0f}m → print Z=0; base bottom at {-BASE_MM:.1f}mm '
          f'(terrain floats at true scaled elevation)')

    def elev_at_ll(lon, lat):
        return sample_dem_elev(dem_full, dem_tf, lon, lat, tr)

    def px_grid(W, H, lons, lats):
        PX = np.zeros((len(lats), len(lons)))
        PY = np.zeros((len(lats), len(lons)))
        for r, lat in enumerate(lats):
            for c, lon in enumerate(lons):
                px, py = print_xy_ll(lon, lat, W, H)
                PX[r, c] = px
                PY[r, c] = py
        return PX, PY

    PX, PY = px_grid(W, H, lons, lats)
    ZZ = mm_v(dem_bbox - z_ref)
    groups = {}

    # ── Terrain ──
    print('\nSTEP 3/6 — Building terrain mesh')
    terrain_full = build_terrain_grid(PX, PY, ZZ)
    colored(terrain_full, COLORS['terrain'])
    groups['terrain'] = [terrain_full]
    tree_list = []
    print(f'  {len(terrain_full.faces):,} faces, watertight={terrain_full.is_watertight}')

    if INCLUDE_TREES:
        forest_mask = forest_mask_from_nlcd(nlcd_path, lons, lats, tr)
        print(f'  Forest cover: {100.0 * forest_mask.mean():.0f}% (3D trees)')

    # ── Buildings + roads + water ──
    print('\nSTEP 4/6 — Buildings + roads + water')

    if INCLUDE_TREES:
        print('  Trees...')
        tree_list = place_trees(
            nlcd_path, TREE_SPACING_NLCD_PX, elev_at_ll, print_xy_ll, z_ref, W, H, tr)
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
            if len(coords) < 3: continue
            bz = mm_v(min(elev_at_ll(lon, lat) for lon, lat in coords) - z_ref)
            pp = [print_xy_ll(lon, lat, W, H) for lon, lat in coords]
            poly2d = Polygon(pp[:-1])
            if not poly2d.is_valid: poly2d = poly2d.buffer(0)
            if poly2d.is_empty: continue
            try:
                bld = extrude_polygon(poly2d, height=BUILDING_HEIGHT_MM)
                bld.apply_translation([0, 0, bz])
                if i == highlight_idx:
                    colored(bld, COLORS['highlight']); highlight_list.append(bld)
                else:
                    colored(bld, COLORS['buildings']); bldg_list.append(bld)
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
        pts_ll = []
        for nid in way['nodes']:
            if nid not in road_nodes: continue
            pts_ll.append(road_nodes[nid])
        if len(pts_ll) < 2: continue
        pairs = build_ribbon_pairs(
            pts_ll, half_w, W, H, tr, elev_at_ll, z_ref, ROAD_OFFSET_MM,
            clip_to_bbox=True)
        mesh = ribbon_mesh_from_pairs(pairs)
        if mesh is None: continue
        colored(mesh, COLORS['roads']); road_list.append(mesh)
    groups['roads'] = road_list
    print(f'    {len(road_list)} road segments')

    # ── Water ──
    print('  Water...')
    pond_list = []; stream_list = []

    # NHD waterbody polygons
    nhd_polys_wgs = []
    try:
        with open(nhd_wb) as f: nhd_wb_geo = json.load(f)
        for feat in nhd_wb_geo.get('features',[]):
            geom = shape(feat['geometry'])
            polys = [geom] if geom.geom_type=='Polygon' else list(geom.geoms)
            for p in polys:
                nhd_polys_wgs.append(p)
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
    nhd_union = unary_union(nhd_polys_wgs) if nhd_polys_wgs else None
    all_ponds = [(True, p) for p in nhd_polys_wgs] + \
                [(False, p) for p in nlcd_water_polys
                 if not (nhd_union and p.intersects(nhd_union))]
    tr_inv = Transformer.from_crs('EPSG:5070', 'EPSG:4326', always_xy=True)
    for is_nhd, poly in all_ponds:
        if is_nhd:
            coords_ll = list(poly.exterior.coords)
        else:
            coords_ll = [tr_inv.transform(x, y) for x, y in poly.exterior.coords]
        sample_elevs = [elev_at_ll(lon, lat) for lon, lat in coords_ll]
        water_z = mm_v((np.percentile(sample_elevs, 10) if sample_elevs
                        else elev_at_ll(*coords_ll[0])) - z_ref) + 0.25
        pp = [print_xy_ll(lon, lat, W, H) for lon, lat in coords_ll]
        if len(pp) < 3: continue
        poly2d = Polygon(pp)
        if not poly2d.is_valid: poly2d = poly2d.buffer(0)
        if poly2d.is_empty: continue
        poly2d = poly2d.intersection(box(0, 0, W, H))   # clip to map bounds
        if poly2d.is_empty: continue
        parts = list(poly2d.geoms) if poly2d.geom_type == 'MultiPolygon' else [poly2d]
        for part in parts:
            if part.is_empty or part.area <= 0: continue
            try:
                slab=extrude_polygon(part, height=0.15)
                slab.apply_translation([0,0,water_z])
                colored(slab, COLORS['ponds']); pond_list.append(slab)
            except: pass

    def build_stream(coords_wgs):
        if len(coords_wgs) < 2:
            return None
        pairs = build_ribbon_pairs(
            coords_wgs, 0.45, W, H, tr, elev_at_ll, z_ref, 0.30,
            clip_to_bbox=True)
        m = ribbon_mesh_from_pairs(pairs)
        if m is None:
            return None
        colored(m, COLORS['streams'])
        return m

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
        'terrain':   (1.0, 1.0, 1.0), 'forest':    (0.133,0.471,0.133),
        'trees':     (0.133,0.471,0.133),
        'buildings': (0.545,0.353,0.169), 'highlight': (1.0, 1.0, 1.0),
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
    stl_parts = [terrain_full] + tree_list + road_list + bldg_list
    stl_parts += highlight_list + pond_list + stream_list
    stl=trimesh.util.concatenate(stl_parts)
    stl_path=os.path.join(OUTPUT_DIR,f'{PROJECT_NAME}_terrain.stl')
    stl.export(stl_path)
    print(f'  STL:  {os.path.getsize(stl_path)/1e6:.1f} MB  → {os.path.basename(stl_path)}')

    print(f'\nSTEP 6/6 — Done!')
    print(f'  Print size: {stl.extents[0]:.1f} × {stl.extents[1]:.1f} × {stl.extents[2]:.1f} mm')
    print(f'  Files saved to: {OUTPUT_DIR}')
    print(f'{"═"*60}\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate a 3D terrain model from places.json')
    parser.add_argument(
        '--place', default=DEFAULT_PLACE,
        help=f'Place id from places.json (default: {DEFAULT_PLACE})')
    parser.add_argument(
        '--keep-cache', action='store_true',
        help='Reuse cached downloads in output_<place>/ (default: clear folder first)')
    args = parser.parse_args()
    load_place(args.place)
    main(keep_cache=args.keep_cache)
