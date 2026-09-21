# Data sources and provenance

## Spatial units

The 191 Hangzhou units are OpenStreetMap `admin_level=8` street, town and
township polygons extracted from the Zhejiang Geofabrik snapshot. Their union
covers approximately 99.128% of the Hangzhou municipality polygon. Detailed
coverage statistics are in `docs/boundary_reports/`.

The raw `zhejiang-latest.osm.pbf` file and its temporary download parts are not
included because they add about 186.7 MB and are not needed for inference. The
snapshot was downloaded from:

https://download.geofabrik.de/asia/china/zhejiang.html

## Population

`data_sources/hangzhou_worldpop_2020.csv` contains 191 polygon totals returned
by the WorldPop v2 population API. Every record identifies the data source as
`worldpop_R2025A_2020_1km` and includes the API task ID. These are modeled
population estimates, not official Hangzhou street-level census counts.

WorldPop API documentation:

https://api.worldpop.org/v2/

## Geographic features

The prepared dataset contains 18 OSM-derived features per area in the order
required by DeepGravity. The order and extraction categories are documented in
`data/hangzhou_inference/feature_schema.json`. The preparation scripts are kept
under `data_preparation/` for transparency. Rebuilding them requires the omitted
Geofabrik PBF plus PyOsmium and Shapely 2; the primary `environment.yml` is for
running inference against the already prepared files.

## Observed flows

No observed Hangzhou OD-flow matrix is included. The records in
`processed/flows_oa.csv.zip` are synthetic self-loop population carriers used
only because the original DeepGravity loader derives population from that file.
`processed/od2flow.pkl` is empty, so those carrier rows are not used as observed
commuting targets.
