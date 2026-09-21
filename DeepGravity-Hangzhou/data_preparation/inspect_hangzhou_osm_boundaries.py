"""Extract and assess Hangzhou administrative boundaries from an OSM PBF file."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import geopandas as gpd

VENDOR_DIR = Path(__file__).resolve().parents[1] / ".tools" / "osmium38"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import osmium
from shapely import wkb


HANGZHOU_RELATION_ID = 3221112


class AdministrativeBoundaryHandler(osmium.SimpleHandler):
    def __init__(self) -> None:
        super().__init__()
        self.factory = osmium.geom.WKBFactory()
        self.records: list[dict] = []
        self.geometry_errors: list[dict] = []

    def area(self, area: osmium.osm.Area) -> None:
        if area.from_way() or area.tags.get("boundary") != "administrative":
            return

        osm_id = area.orig_id()
        try:
            geometry = wkb.loads(self.factory.create_multipolygon(area), hex=True)
        except Exception as exc:  # malformed OSM relations are part of the QA result
            self.geometry_errors.append({"osm_id": osm_id, "error": str(exc)})
            return

        tags = {tag.k: tag.v for tag in area.tags}
        self.records.append(
            {
                "osm_id": osm_id,
                "name": tags.get("name"),
                "name_zh": tags.get("name:zh"),
                "admin_level": tags.get("admin_level"),
                "place": tags.get("place"),
                "type": tags.get("type"),
                "geometry": geometry,
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("pbf", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    handler = AdministrativeBoundaryHandler()
    handler.apply_file(str(args.pbf), locations=True)
    all_admin = gpd.GeoDataFrame(handler.records, geometry="geometry", crs="EPSG:4326")

    hz_matches = all_admin[all_admin["osm_id"] == HANGZHOU_RELATION_ID]
    if len(hz_matches) != 1:
        raise RuntimeError(
            f"Expected one Hangzhou relation {HANGZHOU_RELATION_ID}, found {len(hz_matches)}"
        )
    hz_geometry = hz_matches.geometry.iloc[0]

    # A representative point avoids including neighboring polygons that merely share a border.
    inside = all_admin.geometry.representative_point().within(hz_geometry)
    hangzhou = all_admin[inside | (all_admin["osm_id"] == HANGZHOU_RELATION_ID)].copy()
    hangzhou = hangzhou.sort_values(["admin_level", "name", "osm_id"], na_position="last")

    projected = hangzhou.to_crs("EPSG:32651")
    hangzhou["area_km2"] = projected.geometry.area / 1_000_000
    hangzhou["is_valid"] = hangzhou.geometry.is_valid
    hangzhou["within_hangzhou"] = hangzhou.geometry.apply(
        lambda geom: bool(geom.equals(hz_geometry) or geom.within(hz_geometry))
    )

    summary_by_level = []
    for level, group in hangzhou.groupby("admin_level", dropna=False):
        summary_by_level.append(
            {
                "admin_level": level,
                "count": int(len(group)),
                "valid_count": int(group["is_valid"].sum()),
                "within_count": int(group["within_hangzhou"].sum()),
                "area_km2_min": float(group["area_km2"].min()),
                "area_km2_median": float(group["area_km2"].median()),
                "area_km2_max": float(group["area_km2"].max()),
                "names": [
                    str(name)
                    for name in group["name"].dropna().astype(str).sort_values().tolist()
                ],
            }
        )

    report = {
        "source_pbf": str(args.pbf.resolve()),
        "hangzhou_osm_relation_id": HANGZHOU_RELATION_ID,
        "total_admin_relations_in_zhejiang_extract": int(len(all_admin)),
        "hangzhou_related_boundaries": int(len(hangzhou)),
        "admin_level_counts": dict(
            sorted(Counter(hangzhou["admin_level"].fillna("missing")).items())
        ),
        "invalid_geometry_count": int((~hangzhou["is_valid"]).sum()),
        "not_strictly_within_hangzhou_count": int((~hangzhou["within_hangzhou"]).sum()),
        "geometry_build_errors": handler.geometry_errors,
        "levels": summary_by_level,
    }

    hangzhou.to_file(args.output_dir / "hangzhou_osm_admin_boundaries.geojson", driver="GeoJSON")
    hangzhou.drop(columns="geometry").to_csv(
        args.output_dir / "hangzhou_osm_admin_inventory.csv", index=False, encoding="utf-8-sig"
    )
    (args.output_dir / "hangzhou_osm_admin_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
