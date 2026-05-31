RICHMONDVILLE NY — 3D TERRAIN MODEL
5 square miles centered on 387 Cross Hill Road
Generated May 2026

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL SPECS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Scale:               1:15,000
Vertical exaggeration: 2×  (makes the hills read clearly at tabletop size)
Print size:          ~287 × 288 × 57 mm
Elevation data:      USGS 3DEP, 10m resolution (297m – 660m elevation range)

COLOR KEY
  Tan/beige   — Terrain
  Forest green — Trees (from NLCD land cover satellite data, ~72% of area is forested)
  Brown       — Buildings (230 total, from Overture Maps / Microsoft Building Footprints)
  Red         — 387 Cross Hill Road specifically
  Black       — Roads (24 segments: Cross Hill Rd, Beards Hollow Rd, Ploss Rd, etc.)
  Deep blue   — Ponds (16 water bodies from USGS NHD + NLCD)
  Light blue  — Streams (8 segments from USGS National Hydrography Dataset + OSM)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

for_printing/
  richmondville_colored.3mf
      → Upload to Xometry.com for a full-color print quote.
        Colors are embedded via the 3MF Materials extension.
        Request "full-color binder jetting" or "ColorJet" process.

  richmondville_colored_print.zip
      → OBJ + MTL files zipped together.
        Upload to i.materialise.com (Multicolor+) or Shapeways (Full Color Sandstone).

  richmondville_terrain.stl
      → Single-color geometry only (no color data).
        Use this if you want to print on a home FDM printer (any color filament).
        Also use this for any service that doesn't support color printing.

for_viewing/
  richmondville_colored.glb
      → Drag this into https://3dviewer.net to see the model in 3D with all colors.
        Also opens in Windows 3D Viewer or macOS Quick Look.

source_data/
  dem.tif                    — USGS 3DEP elevation raster (EPSG:5070, 10m resolution)
  nlcd.tif                   — NLCD 2021 land cover raster (30m resolution)
  overture_buildings.geojson — 230 building footprints from Overture Maps
  roads_raw.json             — Road network from OpenStreetMap
  water_raw.json             — Waterways from OpenStreetMap
  nhd_flowlines.geojson      — Stream network from USGS National Hydrography Dataset

scripts/
  build_final_v3.py   — Main pipeline: terrain + trees + buildings + roads + water → GLB + STL
  export_3mf.py       — Exports the colored 3MF (Xometry-ready)
  export_obj_mtl.py   — Exports the OBJ+MTL zip (i.materialise/Shapeways-ready)
  build_roads.py      — Road ribbon mesh builder (imported by build_final_v3.py)
  build_water.py      — Water mesh builder (imported by build_final_v3.py)

  To regenerate everything from scratch, run:
      python3 build_final_v3.py
      python3 export_3mf.py
      python3 export_obj_mtl.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRINTING TIPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- The model fits a 300mm print bed at full size.
  For a 256mm bed, scale to 88% in your slicer.
  For a 220mm bed, scale to 75%.

- Full-color printing (all 7 colors): use the .3mf or .zip files above.
  The technology is called "full-color binder jetting" or "MultiJet" printing.
  Expected cost: roughly $150–$400 depending on service and infill.

- Single-color home printing: use the .stl file.
  Roads and streams will still be visible as raised ridges.
  Suggested settings: 0.2mm layer height, 15% infill, no supports needed.
