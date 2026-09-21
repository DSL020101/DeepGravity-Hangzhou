"""Measure coverage and overlaps for the extracted Hangzhou OSM boundaries."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parents[1] / ".tools" / "shapely2_38"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

from pyproj import Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform, unary_union


HANGZHOU_RELATION_ID = 3221112


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("geojson", type=Path)
    parser.add_argument("report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_collection = json.loads(args.geojson.read_text(encoding="utf-8"))
    records = [
        {"properties": feature["properties"], "geometry": shape(feature["geometry"])}
        for feature in feature_collection["features"]
    ]

    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True).transform
    for record in records:
        record["geometry_utm"] = transform(to_utm, record["geometry"])

    hz = next(
        record["geometry_utm"]
        for record in records
        if int(record["properties"]["osm_id"]) == HANGZHOU_RELATION_ID
    )
    hz_area = hz.area

    by_level: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_level[str(record["properties"].get("admin_level"))].append(record)

    levels = []
    uncovered_level8 = None
    for level, group in sorted(by_level.items()):
        geometries = [record["geometry_utm"] for record in group]
        merged = unary_union(geometries)
        sum_area = sum(geometry.area for geometry in geometries)
        union_area = merged.area
        covered = merged.intersection(hz).area
        uncovered = hz.difference(merged)
        outside = merged.difference(hz).area
        if level == "8":
            uncovered_level8 = uncovered
        levels.append(
            {
                "admin_level": level,
                "count": len(group),
                "valid_count": sum(geometry.is_valid for geometry in geometries),
                "coverage_of_hangzhou_percent": covered / hz_area * 100,
                "uncovered_hangzhou_km2": max(0, hz_area - covered) / 1_000_000,
                "outside_hangzhou_km2": outside / 1_000_000,
                "same_level_overlap_km2": max(0, sum_area - union_area) / 1_000_000,
                "area_km2_min": min(geometry.area for geometry in geometries) / 1_000_000,
                "area_km2_median": sorted(geometry.area for geometry in geometries)[
                    len(geometries) // 2
                ]
                / 1_000_000,
                "area_km2_max": max(geometry.area for geometry in geometries) / 1_000_000,
            }
        )

    level6 = by_level.get("6", [])
    level8 = by_level.get("8", [])
    district_counts = []
    for district in level6:
        district_geometry = district["geometry_utm"]
        children = [
            child
            for child in level8
            if district_geometry.covers(child["geometry_utm"].representative_point())
        ]
        district_counts.append(
            {
                "district": district["properties"].get("name"),
                "district_osm_id": int(district["properties"]["osm_id"]),
                "admin_level_8_count": len(children),
            }
        )

    report = {
        "source_geojson": str(args.geojson.resolve()),
        "hangzhou_area_km2": hz_area / 1_000_000,
        "levels": levels,
        "admin_level_8_by_district": sorted(district_counts, key=lambda row: row["district"]),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if uncovered_level8 is not None:
        to_wgs84 = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True).transform
        gap_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "description": "Hangzhou area not covered by OSM admin_level=8 boundaries",
                        "area_km2": uncovered_level8.area / 1_000_000,
                    },
                    "geometry": mapping(transform(to_wgs84, uncovered_level8)),
                }
            ],
        }
        args.report.with_name("hangzhou_osm_level8_uncovered.geojson").write_text(
            json.dumps(gap_geojson, ensure_ascii=False), encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
