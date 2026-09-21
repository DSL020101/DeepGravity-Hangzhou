"""Fetch WorldPop 2020 population estimates for Hangzhou level-8 OSM areas."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path

import aiohttp


API_ROOT = "https://api.worldpop.org/v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("boundaries", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--concurrency", type=int, default=8)
    return parser.parse_args()


async def submit_and_wait(session: aiohttp.ClientSession, feature: dict, semaphore: asyncio.Semaphore) -> dict:
    props = feature["properties"]
    async with semaphore:
        async with session.post(
            f"{API_ROOT}/population",
            json={"geojson": feature["geometry"], "year": 2020, "resolution": "1km"},
        ) as response:
            response.raise_for_status()
            submitted = await response.json()

        task_id = submitted["task_id"]
        for _ in range(120):
            async with session.get(f"{API_ROOT}/tasks/{task_id}") as response:
                response.raise_for_status()
                status = await response.json()
            if status["status"] == "success":
                result = status["result"]
                return {
                    "osm_id": str(props["osm_id"]),
                    "name": props.get("name") or props.get("name_zh") or "",
                    "population": float(result["total_population"]),
                    "api_area_km2": float(result["area_km2"]),
                    "data_year": int(result["data_year"]),
                    "data_source": result["data_source"],
                    "task_id": task_id,
                }
            if status["status"] == "failed":
                raise RuntimeError(f"WorldPop task {task_id} failed: {status.get('error')}")
            await asyncio.sleep(0.5)
        raise TimeoutError(f"WorldPop task {task_id} did not finish")


async def run(args: argparse.Namespace) -> None:
    collection = json.loads(args.boundaries.read_text(encoding="utf-8"))
    features = [
        feature
        for feature in collection["features"]
        if str(feature["properties"].get("admin_level")) == "8"
    ]
    timeout = aiohttp.ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    semaphore = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=False) as session:
        tasks = [submit_and_wait(session, feature, semaphore) for feature in features]
        rows = []
        for completed in asyncio.as_completed(tasks):
            row = await completed
            rows.append(row)
            print(f"WorldPop {len(rows)}/{len(features)}: {row['name']}", flush=True)

    rows.sort(key=lambda row: int(row["osm_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} population estimates to {args.output.resolve()}")


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
