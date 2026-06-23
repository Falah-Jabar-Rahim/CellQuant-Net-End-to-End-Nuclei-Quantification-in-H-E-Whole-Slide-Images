import os
import json
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors


def save_geojson(features, out_path):
    with open(out_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)


def load_cells_from_qupath_features(qupath_features):
    centroids, cell_types = [], []

    for feature in qupath_features:
        meas = feature.get("properties", {}).get("measurements", {})
        cx = meas.get("Centroid X")
        cy = meas.get("Centroid Y")
        ctype = meas.get("Class ID")

        if cx is None or cy is None or ctype is None:
            continue

        centroids.append([float(cx), float(cy)])
        cell_types.append(int(ctype))

    return np.asarray(centroids, dtype=np.float32), np.asarray(cell_types, dtype=np.int32)


def filter_cells(centroids, cell_types, include_types="all"):
    if include_types == "all":
        mask = np.ones(len(cell_types), dtype=bool)
    else:
        if isinstance(include_types, (int, np.integer)):
            include_types = [int(include_types)]
        mask = np.isin(cell_types, include_types)

    return centroids[mask], cell_types[mask], np.where(mask)[0]


def microns_to_pixels(radius_um, mpp):
    mpp = float(mpp)
    if mpp <= 0:
        raise ValueError(f"mpp must be > 0, got {mpp}")
    return float(radius_um) / mpp


def build_radius_graph(centroids, cell_types, radius_px, include_types="all"):
    pts, types, original_ids = filter_cells(centroids, cell_types, include_types)
    G = nx.Graph()

    for i in range(len(pts)):
        G.add_node(
            i,
            x=float(pts[i, 0]),
            y=float(pts[i, 1]),
            cell_type=int(types[i]),
            original_id=int(original_ids[i]),
        )

    if len(pts) <= 1:
        return G, pts, types, original_ids

    nbrs = NearestNeighbors(radius=radius_px, metric="euclidean")
    nbrs.fit(pts)
    distances, indices = nbrs.radius_neighbors(pts, return_distance=True)

    for i in range(len(pts)):
        for d, j in zip(distances[i], indices[i]):
            j = int(j)
            if i != j and not G.has_edge(i, j):
                G.add_edge(i, j, distance=float(d))

    return G, pts, types, original_ids


def count_cell_types(types_graph, class_names, area_mm2):
    rows = []
    total = len(types_graph)

    for t in sorted(np.unique(types_graph)):
        count = int(np.sum(types_graph == t))
        percentage = 100.0 * count / total if total > 0 else 0.0
        density = count / area_mm2 if area_mm2 > 0 else np.nan

        rows.append({
            "cell_type_id": int(t),
            "cell_type_name": class_names.get(int(t), str(t)),
            "count": count,
            "percentage": percentage,
            "density_per_mm2": density,
        })

    return pd.DataFrame(rows)


def immune_infiltration_score(G, tumor_type=1, immune_type=2):
    tumor_nodes = [n for n in G.nodes if G.nodes[n]["cell_type"] == tumor_type]
    scores = []

    for n in tumor_nodes:
        neigh = list(G.neighbors(n))
        if len(neigh) == 0:
            continue

        immune_count = sum(1 for m in neigh if G.nodes[m]["cell_type"] == immune_type)
        scores.append(immune_count / len(neigh))

    return float(np.mean(scores)) if len(scores) > 0 else np.nan


def make_connection_feature(x1, y1, x2, y2, distance_um, edge_name):
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[float(x1), float(y1)], [float(x2), float(y2)]],
        },
        "properties": {
            "objectType": "annotation",
            "classification": {
                "name": edge_name,
                "color": [0, 255, 255],
            },
            "measurements": {
                "distance_um": float(distance_um),
            },
        },
    }


def make_metric_label(label, x, y):
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [float(x), float(y)],
        },
        "properties": {
            "objectType": "annotation",
            "name": label,
            "classification": {
                "name": label,
                "color": [255, 255, 0],
            },
        },
    }


def get_count_row(count_df, class_name):
    row = count_df[count_df["cell_type_name"] == class_name]
    if len(row) == 0:
        return 0, 0.0, 0.0

    row = row.iloc[0]
    return int(row["count"]), float(row["percentage"]), float(row["density_per_mm2"])


def make_metric_features(count_df, infiltration_value, x=1000, y=1000, row_gap=250):
    tumor_n, tumor_pct, tumor_den = get_count_row(count_df, "Tumor")
    til_n, til_pct, til_den = get_count_row(count_df, "TILs")
    other_n, other_pct, other_den = get_count_row(count_df, "Other")

    til_tumor_ratio = float(til_n) / float(tumor_n) if tumor_n > 0 else np.nan

    labels = [
        f"Tumor: {tumor_n} ({tumor_pct:.1f}%, {tumor_den:.0f}/mm²)",
        f"TILs: {til_n} ({til_pct:.1f}%, {til_den:.0f}/mm²)",
        f"Other: {other_n} ({other_pct:.1f}%, {other_den:.0f}/mm²)",
        f"TIL/Tumor ratio: {til_tumor_ratio:.2f}",
        f"Immune infiltration: {infiltration_value:.2f}",
    ]

    return [
        make_metric_label(label, x=x, y=y + i * row_gap)
        for i, label in enumerate(labels)
    ]


def cell_neighborhood(
    args, qupath_features,
    include_types="all",
    radius_um=30.0,
    mpp=0.5,
    out_dir="cell_connectivity",
    total_tiles=None,
    tile_size = None,
    class_names=None,
):
    os.makedirs(out_dir, exist_ok=True)

    if class_names is None:
        class_names = args.class_names

    class_names = {int(k): str(v) for k, v in class_names.items()}


    if tile_size is None or total_tiles is None:
        raise ValueError("tile_size and total_tiles are required to compute ROI area.")
    pixel_size = mpp*1e-3  # 0.5 micrometers in millimeters
    # # area of ROI
    area_mm2 = tile_size*tile_size*total_tiles * pixel_size**2

    centroids, cell_types = load_cells_from_qupath_features(qupath_features)

    if len(centroids) == 0:
        raise ValueError("No valid cells found in qupath_features.")

    radius_px = microns_to_pixels(radius_um, mpp)

    G, pts_graph, types_graph, original_ids = build_radius_graph(
        centroids=centroids,
        cell_types=cell_types,
        radius_px=radius_px,
        include_types=include_types,
    )

    count_df = count_cell_types(types_graph, class_names, area_mm2)
    infiltration_value = immune_infiltration_score(G, tumor_type=1, immune_type=2)

    tumor_n = count_df.loc[count_df["cell_type_name"] == "Tumor", "count"]
    til_n = count_df.loc[count_df["cell_type_name"] == "TILs", "count"]

    tumor_n = int(tumor_n.iloc[0]) if len(tumor_n) > 0 else 0
    til_n = int(til_n.iloc[0]) if len(til_n) > 0 else 0
    til_tumor_ratio = float(til_n) / float(tumor_n) if tumor_n > 0 else np.nan

    count_df["total_cells"] = int(len(types_graph))
    count_df["ROI_area_mm2"] = float(area_mm2)
    count_df["radius_um"] = float(radius_um)
    count_df["radius_px"] = float(radius_px)
    count_df["til_tumor_ratio"] = til_tumor_ratio
    count_df["immune_infiltration_score"] = infiltration_value
    count_df["num_edges"] = int(G.number_of_edges())
    count_df["avg_degree"] = (
        float(np.mean([d for _, d in G.degree()]))
        if G.number_of_nodes() > 0 else 0.0
    )

    edge_rows = []
    connection_features = []

    for u, v, data in G.edges(data=True):
        u_type_id = G.nodes[u]["cell_type"]
        v_type_id = G.nodes[v]["cell_type"]

        u_type = class_names.get(u_type_id, str(u_type_id))
        v_type = class_names.get(v_type_id, str(v_type_id))

        x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"]
        x2, y2 = G.nodes[v]["x"], G.nodes[v]["y"]

        distance_um = data["distance"] * float(mpp)
        edge_name = "-".join(sorted([u_type, v_type]))

        edge_rows.append({
            "u": u,
            "v": v,
            "u_type": u_type,
            "v_type": v_type,
            "distance_px": data["distance"],
            "distance_um": distance_um,
        })

        connection_features.append(
            make_connection_feature(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                distance_um=distance_um,
                edge_name=edge_name,
            )
        )

    edges_graph_df = pd.DataFrame(edge_rows)

    metric_features = make_metric_features(
        count_df=count_df,
        infiltration_value=infiltration_value,
        x=1000,
        y=1000,
        row_gap=250,
    )

    updated_features = list(qupath_features) + connection_features + metric_features


    updated_geojson_path = os.path.join(out_dir, "qupath_cells_connectivity.geojson")
    save_geojson(updated_features, updated_geojson_path)

    count_df.to_csv(os.path.join(out_dir, "cell_type_stats.csv"), index=False)
    edges_graph_df.to_csv(os.path.join(out_dir, "connectivity_edges.csv"), index=False)

    print(f"Cell neighborhood results saved to: {out_dir}")
    print(f"Updated QuPath GeoJSON saved to: {updated_geojson_path}")

    return {
        "count_df": count_df,
        "edges_graph_df": edges_graph_df,
        "graph": G,
        "qupath_features_updated": updated_geojson_path,
    }