"""
Road ribbon builder — terrain-following road meshes from OSM data.
Returns a list of trimesh objects colored black.
"""
import numpy as np, trimesh, json
from pyproj import Transformer
import warnings; warnings.filterwarnings('ignore')

SCALE=15000; VERT_EXAG=2.0; DEM_STEP=2
ROAD_OFFSET_MM = 0.35   # mm above terrain surface
SAMPLES_PER_MM = 1.5    # sample density along road (in print-mm)

ROAD_WIDTH_MM = {
    'motorway':3.0,'trunk':2.5,'primary':2.5,'secondary':2.0,
    'tertiary':1.8,'unclassified':1.4,'residential':1.2,
    'service':0.9,'track':0.7,'path':0.6,'footway':0.5,
    'default':1.0
}
C_ROAD = np.array([20, 20, 20, 255], dtype=np.uint8)

def mm(m): return m/SCALE*1000.0
def mm_v(m): return mm(m)*VERT_EXAG

def build_road_meshes(dem, dem_step, pixel_size, x0, y0, roads_json_path):
    R, C = dem.shape
    step = pixel_size * dem_step

    def elev_at(wx, wy):
        c = int(np.clip((wx-x0)/step, 0, C-1))
        r = int(np.clip((y0-wy)/step, 0, R-1))
        return dem[r, c]

    def print_xyz(wx, wy):
        c = (wx-x0)/step; r = (y0-wy)/step
        px = c*mm(step); py = (R-1-r)*mm(step)
        pz = mm_v(elev_at(wx,wy) - dem.min()) + ROAD_OFFSET_MM
        return np.array([px, py, pz])

    tr = Transformer.from_crs('EPSG:4326','EPSG:5070',always_xy=True)

    with open(roads_json_path) as f: obj=json.load(f)
    elements = obj['elements']
    nodes = {e['id']:(e['lon'],e['lat']) for e in elements if e['type']=='node'}
    ways  = [e for e in elements if e['type']=='way']

    W = (C-1)*mm(step); H = (R-1)*mm(step)

    road_meshes = []

    for way in ways:
        tags = way.get('tags', {})
        hw = tags.get('highway', 'default')
        half_w = ROAD_WIDTH_MM.get(hw, ROAD_WIDTH_MM['default']) / 2.0

        node_ids = way['nodes']
        # Build world-space (EPSG:5070) points for this road
        pts_world = []
        for nid in node_ids:
            if nid not in nodes: continue
            lon, lat = nodes[nid]
            x, y = tr.transform(lon, lat)
            pts_world.append((x, y))
        if len(pts_world) < 2: continue

        # For each segment, densely sample and build ribbon
        ribbon_left  = []
        ribbon_right = []

        for si in range(len(pts_world)-1):
            x1,y1 = pts_world[si]
            x2,y2 = pts_world[si+1]
            seg_len_mm = np.hypot((x2-x1), (y2-y1)) / SCALE * 1000
            n_samples = max(2, int(seg_len_mm * SAMPLES_PER_MM) + 1)

            for ti, t in enumerate(np.linspace(0, 1, n_samples, endpoint=(si==len(pts_world)-2))):
                wx = x1 + t*(x2-x1); wy = y1 + t*(y2-y1)
                p3 = print_xyz(wx, wy)

                # Road direction (use segment direction)
                dx, dy = (x2-x1), (y2-y1)
                length = np.hypot(dx, dy)
                if length < 1e-6: continue
                # Perpendicular in XY (rotate 90°)
                px_perp = -dy/length * half_w
                py_perp =  dx/length * half_w

                ribbon_left.append( p3 + np.array([px_perp, py_perp, 0]))
                ribbon_right.append(p3 - np.array([px_perp, py_perp, 0]))

        # Clip ribbon pairs to terrain print bounds (clip both sides together)
        pairs = [(l,r) for l,r in zip(ribbon_left,ribbon_right)
                 if (-1<=l[0]<=W+1 and -1<=l[1]<=H+1 and
                     -1<=r[0]<=W+1 and -1<=r[1]<=H+1)]
        if len(pairs) < 2: continue
        ribbon_left  = [p[0] for p in pairs]
        ribbon_right = [p[1] for p in pairs]

        # Build mesh from ribbon strips
        L = np.array(ribbon_left);  R_arr = np.array(ribbon_right)
        n = len(L)
        verts = np.vstack([L, R_arr])  # left = 0..n-1, right = n..2n-1
        faces = []
        for i in range(n-1):
            l0,l1 = i, i+1
            r0,r1 = n+i, n+i+1
            faces.append([l0, r0, r1])
            faces.append([l0, r1, l1])
        faces = np.array(faces)

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        fc = np.tile(C_ROAD, (len(faces), 1))
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=fc)
        road_meshes.append(mesh)

    return road_meshes

if __name__ == '__main__':
    import rasterio
    with rasterio.open('/sessions/gifted-keen-volta/mnt/outputs/dem.tif') as src:
        dem_full = src.read(1).astype(np.float64)
        tf = src.transform
    dem = dem_full[::DEM_STEP, ::DEM_STEP]
    meshes = build_road_meshes(dem, DEM_STEP, tf.a, tf.c, tf.f,
                                '/sessions/gifted-keen-volta/mnt/outputs/roads_raw.json')
    total_faces = sum(len(m.faces) for m in meshes)
    print(f'Built {len(meshes)} road meshes, {total_faces:,} total faces')
    for m in meshes[:3]:
        print(f'  mesh: {len(m.vertices)} verts, bounds Z={m.bounds[0][2]:.2f}-{m.bounds[1][2]:.2f}mm')
