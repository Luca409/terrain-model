"""
Water mesh builder — ponds (flat polygons) + streams (ribbons).
Sources: NHD waterbody polygon, NLCD class-11 raster, NHD flowlines, OSM streams.
"""
import numpy as np, trimesh, json, warnings
import rasterio
from rasterio.features import shapes as rasterio_shapes
from shapely.geometry import shape, Polygon, MultiPolygon, mapping
from shapely.ops import unary_union
from pyproj import Transformer
import pynhd
warnings.filterwarnings('ignore')

SCALE    = 15000
VERT_EXAG = 2.0
DEM_STEP  = 2
WATER_OFFSET_MM  = 0.25   # mm above terrain for pond surface
STREAM_OFFSET_MM = 0.30   # mm above terrain for stream ribbons
STREAM_WIDTH_MM  = 0.9    # stream ribbon half-width * 2
SAMPLES_PER_MM   = 1.5

C_POND   = np.array([ 30, 110, 200, 255], dtype=np.uint8)  # deep blue
C_STREAM = np.array([ 60, 150, 220, 255], dtype=np.uint8)  # lighter blue

def mm(m):  return m / SCALE * 1000.0
def mm_v(m): return mm(m) * VERT_EXAG

OUTDIR = '/sessions/gifted-keen-volta/mnt/outputs'


def build_water_meshes(dem, dem_step, pixel_size, x0, y0):
    R, C = dem.shape
    step = pixel_size * dem_step
    W = (C-1)*mm(step); H = (R-1)*mm(step)
    dem_min = dem.min()

    def elev_at(wx, wy):
        c = int(np.clip((wx-x0)/step, 0, C-1))
        r = int(np.clip((y0-wy)/step, 0, R-1))
        return dem[r, c]

    def print_xyz_flat(wx, wy, z_mm):
        c = (wx-x0)/step; r = (y0-wy)/step
        return np.array([c*mm(step), (R-1-r)*mm(step), z_mm])

    def print_xy(wx, wy):
        c = (wx-x0)/step; r = (y0-wy)/step
        return c*mm(step), (R-1-r)*mm(step)

    def in_bounds(px, py):
        return -2 <= px <= W+2 and -2 <= py <= H+2

    tr_4326_5070 = Transformer.from_crs('EPSG:4326','EPSG:5070',always_xy=True)
    tr_5070_4326 = Transformer.from_crs('EPSG:5070','EPSG:4326',always_xy=True)

    pond_meshes   = []
    stream_meshes = []

    # ────────────────────────────────────────────
    # 1. POND POLYGONS
    # ────────────────────────────────────────────

    # 1a. NHD waterbody polygon (accurate shape, EPSG:4326)
    nhd_polys_5070 = []
    try:
        bbox = (-74.539130, 42.608741, -74.495196, 42.641067)
        wd   = pynhd.WaterData('nhdwaterbody')
        wb   = wd.bybox(bbox)
        for _, row in wb.iterrows():
            geom = row.geometry
            polys = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
            for p in polys:
                coords_5070 = [tr_4326_5070.transform(lon, lat)
                               for lon, lat in p.exterior.coords]
                nhd_polys_5070.append(Polygon(coords_5070))
        print(f'  NHD: {len(nhd_polys_5070)} pond polygon(s)')
    except Exception as e:
        print(f'  NHD pond error: {e}')

    # 1b. NLCD class-11 vectorized polygons (EPSG:5070 already)
    with rasterio.open(f'{OUTDIR}/nlcd.tif') as src:
        nlcd = src.read(1); ntf = src.transform

    water_mask = (nlcd == 11).astype(np.uint8)
    nlcd_polys_5070 = []
    for geom_d, val in rasterio_shapes(water_mask, mask=water_mask, transform=ntf):
        if val == 1:
            p = shape(geom_d)
            # Smooth slightly: buffer out 10m then back in (removes sharp corners)
            p_smooth = p.buffer(15).buffer(-15).simplify(10)
            if p_smooth.is_valid and p_smooth.area > 400:
                nlcd_polys_5070.append(p_smooth)
    print(f'  NLCD: {len(nlcd_polys_5070)} water polygon(s)')

    # Merge NHD into NLCD list (NHD takes priority — remove NLCD pixels inside NHD)
    all_pond_polys = []
    nhd_union = unary_union(nhd_polys_5070) if nhd_polys_5070 else None

    for p in nhd_polys_5070:
        all_pond_polys.append(('nhd', p))

    for p in nlcd_polys_5070:
        if nhd_union and p.intersects(nhd_union):
            continue   # skip NLCD pixels already covered by NHD
        all_pond_polys.append(('nlcd', p))

    print(f'  Total ponds: {len(all_pond_polys)}')

    # Build flat polygon meshes for each pond
    for src_name, poly_5070 in all_pond_polys:
        # Sample terrain elevations inside polygon
        minx, miny, maxx, maxy = poly_5070.bounds
        sample_pts = []
        for sx in np.arange(minx, maxx, step):
            for sy in np.arange(miny, maxy, step):
                if poly_5070.contains_point_fast if hasattr(poly_5070,'contains_point_fast') else True:
                    sample_pts.append((sx, sy))
        if sample_pts:
            elevs = [elev_at(sx, sy) for sx, sy in sample_pts]
            water_elev = np.percentile(elevs, 10)   # use 10th percentile = low point
        else:
            water_elev = elev_at((minx+maxx)/2, (miny+maxy)/2)

        z_water = mm_v(water_elev - dem_min) + WATER_OFFSET_MM

        # Convert polygon exterior to print coords
        ext_coords = list(poly_5070.exterior.coords)
        pp = [print_xy(x, y) for x, y in ext_coords]
        pp = [p for p in pp if in_bounds(p[0], p[1])]
        if len(pp) < 3:
            continue
        poly_print = Polygon(pp)
        if not poly_print.is_valid:
            poly_print = poly_print.buffer(0)
        if poly_print.area < 0.1:
            continue

        try:
            # Flat mesh = extrude by tiny amount, then take just the top face
            # Use a very thin slab (0.1mm) so it prints as a flat pad
            slab = trimesh.creation.extrude_polygon(poly_print, height=0.15)
            slab.apply_translation([0, 0, z_water])
            fc = np.tile(C_POND, (len(slab.faces), 1))
            slab.visual = trimesh.visual.ColorVisuals(mesh=slab, face_colors=fc)
            pond_meshes.append(slab)
        except Exception as e:
            pass

    print(f'  Built {len(pond_meshes)} pond meshes')

    # ────────────────────────────────────────────
    # 2. STREAM RIBBONS (NHD flowlines + OSM streams)
    # ────────────────────────────────────────────
    half_w = STREAM_WIDTH_MM / 2.0

    def build_stream_ribbon(coords_wgs84):
        # coords_wgs84: list of (lon, lat)
        pts_5070 = [tr_4326_5070.transform(lon, lat) for lon, lat in coords_wgs84]
        ribbon_pairs = []
        for si in range(len(pts_5070)-1):
            x1,y1 = pts_5070[si]; x2,y2 = pts_5070[si+1]
            seg_mm = np.hypot(x2-x1, y2-y1) / SCALE * 1000
            n = max(2, int(seg_mm * SAMPLES_PER_MM) + 1)
            for t in np.linspace(0, 1, n, endpoint=(si==len(pts_5070)-2)):
                wx = x1 + t*(x2-x1); wy = y1 + t*(y2-y1)
                pxx, pxy = print_xy(wx, wy)
                if not in_bounds(pxx, pxy): continue
                pz = mm_v(elev_at(wx, wy) - dem_min) + STREAM_OFFSET_MM
                p3 = np.array([pxx, pxy, pz])
                dx, dy = (x2-x1), (y2-y1); L = np.hypot(dx,dy)
                if L < 1e-6: continue
                perp_x = -dy/L * half_w; perp_y = dx/L * half_w
                ribbon_pairs.append((
                    p3 + np.array([perp_x, perp_y, 0]),
                    p3 - np.array([perp_x, perp_y, 0])
                ))
        return ribbon_pairs

    # NHD flowlines
    try:
        with open(f'{OUTDIR}/nhd_flowlines.geojson') as f:
            fl_geo = json.load(f)
        for feat in fl_geo['features']:
            geom = feat['geometry']
            if geom['type'] == 'LineString':
                lines = [geom['coordinates']]
            elif geom['type'] == 'MultiLineString':
                lines = geom['coordinates']
            else:
                continue
            for coords in lines:
                pairs = build_stream_ribbon(coords)
                if len(pairs) < 2: continue
                L = np.array([p[0] for p in pairs])
                Rarr = np.array([p[1] for p in pairs])
                n = len(L)
                verts = np.vstack([L, Rarr])
                faces = []
                for i in range(n-1):
                    faces += [[i,n+i,n+i+1],[i,n+i+1,i+1]]
                mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)
                fc = np.tile(C_STREAM, (len(mesh.faces), 1))
                mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=fc)
                stream_meshes.append(mesh)
        print(f'  NHD flowlines → {len(stream_meshes)} stream ribbons')
    except Exception as e:
        print(f'  NHD flowline error: {e}')

    # OSM streams (for any not in NHD)
    try:
        with open(f'{OUTDIR}/water_raw.json') as f:
            osm_water = json.load(f)
        osm_nodes = {e['id']:(e['lon'],e['lat'])
                     for e in osm_water['elements'] if e['type']=='node'}
        osm_ways  = [e for e in osm_water['elements']
                     if e['type']=='way' and not (e['nodes'][0]==e['nodes'][-1])]
        before = len(stream_meshes)
        for way in osm_ways:
            coords = [osm_nodes[nid] for nid in way['nodes'] if nid in osm_nodes]
            if len(coords) < 2: continue
            pairs = build_stream_ribbon(coords)
            if len(pairs) < 2: continue
            L = np.array([p[0] for p in pairs])
            Rarr = np.array([p[1] for p in pairs])
            n = len(L)
            verts = np.vstack([L, Rarr])
            faces = []
            for i in range(n-1):
                faces += [[i,n+i,n+i+1],[i,n+i+1,i+1]]
            mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)
            fc = np.tile(C_STREAM, (len(mesh.faces), 1))
            mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=fc)
            stream_meshes.append(mesh)
        print(f'  OSM streams → {len(stream_meshes)-before} additional stream ribbons')
    except Exception as e:
        print(f'  OSM stream error: {e}')

    return pond_meshes, stream_meshes


if __name__ == '__main__':
    with rasterio.open(f'{OUTDIR}/dem.tif') as src:
        dem_full = src.read(1).astype(np.float64)
        tf = src.transform
    dem = dem_full[::DEM_STEP, ::DEM_STEP]
    ponds, streams = build_water_meshes(dem, DEM_STEP, tf.a, tf.c, tf.f)
    print(f'\nResult: {len(ponds)} ponds, {len(streams)} stream segments')
    total_faces = sum(len(m.faces) for m in ponds+streams)
    print(f'Total water faces: {total_faces:,}')
