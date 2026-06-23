import json
import uuid
import math
import os

CLASS_COLORS = {
    1: [255, 0, 0],      # red
    2: [0, 255, 0],      # green
    3: [255, 255, 0],    # yellow
}

def parse_tile_xy(tile_name):
    """
    Expected tile name:
    WSI_NAME_xStart_yStart.png
    """
    stem = os.path.splitext(os.path.basename(tile_name))[0]
    parts = stem.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Cannot parse tile coordinates from: {tile_name}")

    wsi_base = parts[0]
    x_start = int(parts[1])
    y_start = int(parts[2])
    return wsi_base, x_start, y_start


def circle_polygon(cx, cy, radius=6, n_points=16):
    coords = []
    for i in range(n_points):
        a = 2 * math.pi * i / n_points
        coords.append([cx + radius * math.cos(a), cy + radius * math.sin(a)])
    coords.append(coords[0])
    return coords


def make_qupath_cell_feature(cx_wsi, cy_wsi, ctype, class_name=None, radius=6):
    color = CLASS_COLORS.get(int(ctype), [255, 255, 255])
    class_name = class_name or f"Class_{ctype}"

    polygon = circle_polygon(cx_wsi, cy_wsi, radius=radius)

    return {
        "type": "Feature",
        "id": str(uuid.uuid4()),
        "geometry": {
            "type": "Polygon",
            "coordinates": [polygon]
        },
        "properties": {
            "objectType": "cell",
            "classification": {
                "name": class_name,
                "color": color
            },
            "measurements": {
                "Centroid X": float(cx_wsi),
                "Centroid Y": float(cy_wsi),
                "Class ID": int(ctype)
            }
        }
    }


def save_qupath_geojson(features, save_path):
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(save_path, "w") as f:
        json.dump(geojson, f, indent=2)