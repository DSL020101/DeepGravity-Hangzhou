"""Build a DeepGravity inference dataset for Hangzhou from OSM and WorldPop."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".tools" / "shapely2_38"))
sys.path.insert(0, str(ROOT / ".tools" / "osmium38"))

import numpy as np
import osmium
import pandas as pd
from pyproj import Transformer
from shapely import from_wkb
from shapely.geometry import LineString, Point, mapping, shape
from shapely.ops import transform
from shapely.strtree import STRtree


FEATURE_NAMES = [
    "commercial_landuse",
    "food_point",
    "food_polygon",
    "health_point",
    "health_polygon",
    "industrial_landuse",
    "main_road_line",
    "natural_landuse",
    "other_road_line",
    "residential_landuse",
    "residential_road_line",
    "retail_landuse",
    "retail_point",
    "retail_polygon",
    "education_point",
    "education_polygon",
    "transport_point",
    "transport_polygon",
]

LANDUSE_INDEX = {
    "commercial": 0,
    "industrial": 5,
    "residential": 9,
    "retail": 11,
}

NATURAL_LANDUSE = {
    "allotments", "basin", "cemetery", "conservation", "farmland", "farmyard",
    "forest", "grass", "greenfield", "meadow", "orchard", "plant_nursery",
    "recreation_ground", "reservoir", "salt_pond", "village_green", "vineyard",
}

MAIN_HIGHWAYS = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link",
}

FOOD_AMENITIES = {
    "bar", "biergarten", "cafe", "fast_food", "food_court", "ice_cream", "pub", "restaurant",
}
FOOD_SHOPS = {
    "alcohol", "bakery", "beverages", "butcher", "confectionery", "convenience",
    "deli", "dairy", "farm", "food", "frozen_food", "greengrocer", "seafood", "tea",
}
HEALTH_AMENITIES = {
    "clinic", "dentist", "doctors", "hospital", "nursing_home", "pharmacy", "veterinary",
}
EDUCATION_AMENITIES = {
    "college", "driving_school", "kindergarten", "language_school", "library",
    "music_school", "school", "university",
}
TRANSPORT_AMENITIES = {
    "bicycle_parking", "bus_station", "car_rental", "car_sharing", "charging_station",
    "ferry_terminal", "fuel", "parking", "parking_entrance", "taxi",
}
TRANSPORT_RAILWAY = {"halt", "station", "subway_entrance", "tram_stop"}
RETAIL_BUILDINGS = {"commercial", "department_store", "kiosk", "retail", "shop", "supermarket"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("pbf", type=Path)
    parser.add_argument("boundaries", type=Path)
    parser.add_argument("population", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def tags_dict(tags) -> dict[str, str]:
    return {tag.k: tag.v for tag in tags}


def facility_categories(tags: dict[str, str]) -> set[str]:
    categories: set[str] = set()
    amenity = tags.get("amenity", "")
    shop = tags.get("shop", "")
    if amenity in FOOD_AMENITIES or shop in FOOD_SHOPS:
        categories.add("food")
    if amenity in HEALTH_AMENITIES or "healthcare" in tags:
        categories.add("health")
    if amenity in EDUCATION_AMENITIES:
        categories.add("education")
    if (
        amenity in TRANSPORT_AMENITIES
        or tags.get("highway") == "bus_stop"
        or "public_transport" in tags
        or tags.get("railway") in TRANSPORT_RAILWAY
        or "aeroway" in tags
    ):
        categories.add("transport")
    if (shop and shop not in {"no", "vacant"}) or amenity == "marketplace" or tags.get("building") in RETAIL_BUILDINGS:
        categories.add("retail")
    return categories


class FeatureCollector(osmium.SimpleHandler):
    def __init__(self, zones: list, zone_area_m2: np.ndarray) -> None:
        super().__init__()
        self.zones = zones
        self.tree = STRtree(zones)
        self.zone_area_m2 = zone_area_m2
        self.values = np.zeros((len(zones), len(FEATURE_NAMES)), dtype=float)
        self.factory = osmium.geom.WKBFactory()
        self.stats = {"nodes": 0, "roads": 0, "areas": 0, "geometry_errors": 0}
        self.seen_areas: set[tuple[str, int]] = set()

    def _point_zone(self, point: Point) -> int | None:
        indices = self.tree.query(point, predicate="within")
        if len(indices):
            return int(indices[0])
        indices = self.tree.query(point, predicate="intersects")
        return int(indices[0]) if len(indices) else None

    def node(self, node: osmium.osm.Node) -> None:
        tags = tags_dict(node.tags)
        categories = facility_categories(tags)
        if not categories or not node.location.valid():
            return
        index = self._point_zone(Point(node.location.lon, node.location.lat))
        if index is None:
            return
        for category in categories:
            self.values[index, {"food": 1, "health": 3, "retail": 12, "education": 14, "transport": 16}[category]] += 1
        self.stats["nodes"] += 1

    def way(self, way: osmium.osm.Way) -> None:
        tags = tags_dict(way.tags)
        highway = tags.get("highway")
        if not highway or highway in {"construction", "proposed", "raceway"}:
            return
        try:
            geometry = from_wkb(bytes.fromhex(self.factory.create_linestring(way)))
        except Exception:
            self.stats["geometry_errors"] += 1
            return
        feature_index = 10 if highway in {"residential", "living_street"} else 6 if highway in MAIN_HIGHWAYS else 8
        for index in self.tree.query(geometry, predicate="intersects"):
            clipped = geometry.intersection(self.zones[int(index)])
            if not clipped.is_empty:
                self.values[int(index), feature_index] += clipped.length / self.zone_area_m2[int(index)]
        self.stats["roads"] += 1

    def area(self, area: osmium.osm.Area) -> None:
        tags = tags_dict(area.tags)
        landuse = tags.get("landuse")
        is_natural = "natural" in tags or landuse in NATURAL_LANDUSE
        categories = facility_categories(tags)
        if landuse not in LANDUSE_INDEX and not is_natural and not categories:
            return
        key = ("way" if area.from_way() else "relation", int(area.orig_id()))
        if key in self.seen_areas:
            return
        self.seen_areas.add(key)
        try:
            geometry = from_wkb(bytes.fromhex(self.factory.create_multipolygon(area)))
        except Exception:
            self.stats["geometry_errors"] += 1
            return
        if geometry.is_empty:
            return

        if landuse in LANDUSE_INDEX or is_natural:
            target_indices = []
            if landuse in LANDUSE_INDEX:
                target_indices.append(LANDUSE_INDEX[landuse])
            if is_natural:
                target_indices.append(7)
            for index in self.tree.query(geometry, predicate="intersects"):
                clipped = geometry.intersection(self.zones[int(index)])
                if clipped.is_empty:
                    continue
                for feature_index in target_indices:
                    self.values[int(index), feature_index] += clipped.area / self.zone_area_m2[int(index)]

        if categories:
            index = self._point_zone(geometry.representative_point())
            if index is not None:
                for category in categories:
                    self.values[index, {"food": 2, "health": 4, "retail": 13, "education": 15, "transport": 17}[category]] += 1
        self.stats["areas"] += 1


def write_zip_csv(frame: pd.DataFrame, path: Path, archive_name: str) -> None:
    csv_bytes = frame.to_csv(index=False).encode("utf-8")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(archive_name, csv_bytes)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processed = args.output_dir / "processed"
    processed.mkdir(exist_ok=True)

    collection = json.loads(args.boundaries.read_text(encoding="utf-8"))
    level8 = [feature for feature in collection["features"] if str(feature["properties"].get("admin_level")) == "8"]
    level6 = [feature for feature in collection["features"] if str(feature["properties"].get("admin_level")) == "6"]
    level5 = [feature for feature in collection["features"] if str(feature["properties"].get("admin_level")) == "5"]
    if len(level8) != 191 or len(level5) != 1:
        raise RuntimeError(f"Expected 191 level-8 areas and one Hangzhou boundary, got {len(level8)} and {len(level5)}")

    zones = [shape(feature["geometry"]) for feature in level8]
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True).transform
    to_wgs84 = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True).transform
    projected_zones = [transform(to_utm, zone) for zone in zones]
    area_m2 = np.array([geometry.area for geometry in projected_zones], dtype=float)

    district_geometries = [shape(feature["geometry"]) for feature in level6]
    district_tree = STRtree(district_geometries)
    population = pd.read_csv(args.population, dtype={"osm_id": str}).set_index("osm_id")

    collector = FeatureCollector(zones, area_m2)
    collector.apply_file(str(args.pbf), locations=True)

    output_features = []
    oa_ids = []
    names = []
    districts = []
    centroids = {}
    populations = []
    oa2features = OrderedDict()
    named_feature_rows = []
    tile_feature_dict = OrderedDict()

    order = sorted(range(len(level8)), key=lambda index: int(level8[index]["properties"]["osm_id"]))
    for index in order:
        feature = level8[index]
        props = feature["properties"]
        osm_id = str(props["osm_id"])
        oa_id = f"HZ_{osm_id}"
        name = props.get("name") or props.get("name_zh") or oa_id
        point = zones[index].representative_point()
        district_indices = district_tree.query(point, predicate="within")
        district = ""
        if len(district_indices):
            district_props = level6[int(district_indices[0])]["properties"]
            district = district_props.get("name") or district_props.get("name_zh") or ""

        centroid_utm = projected_zones[index].centroid
        centroid = transform(to_wgs84, centroid_utm)
        centroids[oa_id] = [float(centroid.x), float(centroid.y)]
        vector = [float(value) for value in collector.values[index].tolist()]
        oa2features[oa_id] = vector
        pop = max(float(population.loc[osm_id, "population"]), 1e-6)

        out_props = {
            "oa_ID": oa_id,
            "osm_id": osm_id,
            "name": name,
            "district": district,
            "admin_level": "8",
            "area_km2": float(area_m2[index] / 1_000_000),
            "population_2020": pop,
        }
        output_features.append({"type": "Feature", "properties": out_props, "geometry": feature["geometry"]})
        oa_ids.append(oa_id)
        names.append(name)
        districts.append(district)
        populations.append(pop)
        named_feature_rows.append({"oa_ID": oa_id, **dict(zip(FEATURE_NAMES, vector))})

        values = vector
        tile_feature_dict[oa_id] = {
            "commercial_landuse": [values[0]],
            "food_point": [values[1]],
            "food_poly": [values[2], 0],
            "health_point": [values[3]],
            "health_poly": [values[4], 0],
            "industrial_landuse": [values[5]],
            "main_road_line": [values[6]],
            "natural_landuse": [values[7]],
            "other_road_line": [values[8]],
            "residential_landuse": [values[9]],
            "residential_line": [values[10]],
            "retail_landuse": [values[11]],
            "retail_point": [values[12]],
            "retail_poly": [values[13], 0],
            "school_point": [values[14]],
            "school_poly": [values[15], 0],
            "transport_point": [values[16]],
            "transport_poly": [values[17], 0],
        }

    output_collection = {"type": "FeatureCollection", "name": "hangzhou_level8_output_areas", "features": output_features}
    (args.output_dir / "output_areas.geojson").write_text(json.dumps(output_collection, ensure_ascii=False), encoding="utf-8")
    tessellation = {
        "type": "FeatureCollection",
        "name": "hangzhou_prediction_region",
        "features": [{"type": "Feature", "properties": {"tile_ID": "HZ_ALL"}, "geometry": level5[0]["geometry"]}],
    }
    (args.output_dir / "tessellation.geojson").write_text(json.dumps(tessellation, ensure_ascii=False), encoding="utf-8")

    features_df = pd.DataFrame([[oa_id, *oa2features[oa_id]] for oa_id in oa_ids], columns=["oa_ID", *[str(i) for i in range(18)]])
    features_df.to_csv(args.output_dir / "features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(named_feature_rows).to_csv(args.output_dir / "features_named.csv", index=False, encoding="utf-8-sig")
    population_df = pd.DataFrame({"oa_ID": oa_ids, "name": names, "district": districts, "population": populations, "year": 2020, "source": "WorldPop R2025A 2020 1km"})
    population_df.to_csv(args.output_dir / "population.csv", index=False, encoding="utf-8-sig")
    population_df.to_csv(processed / "population.csv", index=False, encoding="utf-8-sig")

    oa_gdf = pd.DataFrame(
        {
            "geo_code": oa_ids,
            "centroid": [str(centroids[oa_id]) for oa_id in oa_ids],
            "area_km2": [output_features[i]["properties"]["area_km2"] for i in range(len(output_features))],
        }
    )
    oa_gdf.to_csv(processed / "oa_gdf.csv.gz", index=False, compression="gzip")
    with (processed / "oa2features.pkl").open("wb") as handle:
        pickle.dump(dict(oa2features), handle)
    with (processed / "oa2centroid.pkl").open("wb") as handle:
        pickle.dump(centroids, handle)
    with (processed / "od2flow.pkl").open("wb") as handle:
        pickle.dump({}, handle)
    (processed / "tileid2oa2handmade_features.json").write_text(
        json.dumps({"HZ_ALL": tile_feature_dict}, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(["HZ_ALL"]).to_csv(processed / "test_tiles.csv", index=False, header=False)
    # main.py constructs the training dataset even in test mode and reads
    # train_data[0] to infer the model input width.  Reusing the single tile
    # here only supplies that structural origin list; it does not make these
    # data suitable for training without observed OD flows.
    pd.DataFrame(["HZ_ALL"]).to_csv(processed / "train_tiles.csv", index=False, header=False)

    # The repository's load_data() derives oa2pop by grouping flows_oa.csv.zip
    # on residence.  Put one self-loop population-carrier row per area in this
    # compatibility file so inference receives the real WorldPop values.  The
    # actual OD dictionary remains empty below, so these rows are never treated
    # as observed commuting targets by FlowDataset.
    population_carriers = pd.DataFrame(
        {"residence": oa_ids, "workplace": oa_ids, "commuters": populations}
    )
    write_zip_csv(population_carriers, processed / "flows_oa.csv.zip", "flows_oa.csv")
    pd.DataFrame(columns=["origin", "destination", "flow"]).to_csv(args.output_dir / "flows.csv", index=False)

    schema = {
        "feature_order": FEATURE_NAMES,
        "geometry_measurement_compatibility": "Polygon areas in WGS84 degree^2 and road lengths in WGS84 degrees are divided by zone area in m^2 to match the scale of the repository's New York features; POI and facility polygons are raw counts.",
        "road_classes": {"residential": ["residential", "living_street"], "main": sorted(MAIN_HIGHWAYS), "other": "all remaining non-construction highway=* ways"},
        "population": "WorldPop R2025A 2020 1km polygon totals",
        "flows": "Observed OD flows are not available. flows.csv and od2flow.pkl are intentionally empty; do not train or calculate CPC with this dataset.",
        "processed_flows_oa": "Contains one synthetic self-loop per area solely to pass WorldPop totals through the repository's population-from-flows loader. These rows are not observed commuting flows.",
        "collector_stats": collector.stats,
    }
    (args.output_dir / "feature_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "dataset": "hangzhou_inference",
        "output_areas": len(oa_ids),
        "tiles": 1,
        "feature_count": 18,
        "population_sum": float(sum(populations)),
        "observed_flow_rows": 0,
        "population_carrier_rows": len(population_carriers),
        "all_geometries_present": len(output_features) == 191,
        "all_populations_positive": all(value > 0 for value in populations),
        "all_features_finite": bool(np.isfinite(collector.values).all()),
        "collector_stats": collector.stats,
    }
    (args.output_dir / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
