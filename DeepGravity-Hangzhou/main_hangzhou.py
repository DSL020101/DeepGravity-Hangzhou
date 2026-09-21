"""Run DeepGravity probability inference on the prepared Hangzhou dataset.

This is a readable, one-click PyCharm entry point modeled after the data-loading
and model-inference sections of the repository's ``main.py``.  It intentionally
does not train or calculate CPC because no observed Hangzhou OD-flow matrix is
available.
"""

from __future__ import print_function

import argparse
import json
import os
import random
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "hangzhou_inference"
DEFAULT_DEEPGRAVITY_DIR = PROJECT_ROOT / "deepgravity"
DEFAULT_CHECKPOINT = DEFAULT_DEEPGRAVITY_DIR / "results" / "model_DG_new_york.pt"
DEFAULT_OUTPUT = DEFAULT_DATASET_DIR / "predicted_probabilities_DG_new_york_transfer_main.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use the New York DeepGravity checkpoint to infer Hangzhou OD probabilities"
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--deepgravity-dir", type=Path, default=DEFAULT_DEEPGRAVITY_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def require(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    deepgravity_dir = args.deepgravity_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()

    print("=" * 72)
    print("Hangzhou DeepGravity probability inference")
    print("=" * 72)
    print(f"Hangzhou dataset : {dataset_dir}")
    print(f"DeepGravity code : {deepgravity_dir}")
    print(f"Model checkpoint : {checkpoint_path}")
    print(f"Output CSV       : {output_path}")
    print()

    # These are the same three random seeds configured in the repository's
    # main.py.  In evaluation mode the model contains no active dropout, but
    # fixing them keeps the whole run deterministic.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    require(dataset_dir / "processed", "Hangzhou processed-data directory")
    require(dataset_dir / "population.csv", "Hangzhou population table")
    require(deepgravity_dir / "utils.py", "DeepGravity utils.py")
    require(deepgravity_dir / "data_loader.py", "DeepGravity data_loader.py")
    require(deepgravity_dir / "models" / "deepgravity.py", "DeepGravity model definition")
    require(checkpoint_path, "New York model checkpoint")

    use_cuda = args.device == "gpu" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[1/6] Runtime device: {device}")

    # utils.py and data_loader.py use repository-relative paths to import the
    # model files, so load them while the working directory is the original
    # DeepGravity source directory.
    original_cwd = Path.cwd()
    os.chdir(deepgravity_dir)
    try:
        utils = SourceFileLoader("hangzhou_main_utils", "./utils.py").load_module()
        dgd = SourceFileLoader("hangzhou_main_data_loader", "./data_loader.py").load_module()
        print("[2/6] Loaded original DeepGravity utils.py and data_loader.py")

        # This is the same load_data() path used by the repository's main.py.
        loaded = utils.load_data(
            str(dataset_dir),
            "tile_ID",
            "geometry",
            "oa_ID",
            "geometry",
            "origin",
            "destination",
            "flow",
        )
        (
            tileid2oa2features,
            oa_gdf,
            _flow_df,
            oa2pop,
            oa2features,
            od2flow,
            oa2centroid,
        ) = loaded

        # Match main.py exactly: prepend log(population) to each area's 18 OSM
        # features.  An OD pair then has 19 origin values, 19 destination
        # values, and one distance value: 39 inputs in total.
        oa2features = {
            oa: [float(np.log(oa2pop[oa]))] + [float(value) for value in features]
            for oa, features in oa2features.items()
        }
        print(
            f"[3/6] Loaded {len(oa_gdf)} areas, {len(next(iter(oa2features.values())))-1} "
            f"OSM features per area, population sum={sum(oa2pop.values()):.2f}"
        )
        print(f"      Observed Hangzhou OD rows used by the model: {len(od2flow)}")

        test_tiles = pd.read_csv(
            dataset_dir / "processed" / "test_tiles.csv", header=None, dtype=str
        )[0].tolist()
        origins = [
            oa for tile in test_tiles for oa in tileid2oa2features[str(tile)].keys()
        ]

        # No observed flows are supplied.  FlowDataset is used only for the
        # original get_features()/get_X_T() implementation.
        dataset = dgd.FlowDataset(
            origins,
            tileid2oa2features2vals=tileid2oa2features,
            o2d2flow={},
            oa2features=oa2features,
            oa2pop=oa2pop,
            oa2centroid=oa2centroid,
            dim_dests=int(1e9),
            frac_true_dest=0.0,
            model="DG",
        )
        input_width = len(dataset.get_features(origins[0], origins[0]))
        if input_width != 39:
            raise RuntimeError(f"Expected a 39-value DeepGravity input, got {input_width}")
        print(
            f"[4/6] Built inference dataset: {len(origins)} origins, "
            f"{len(origins)} candidate destinations, input width={input_width}"
        )

        # instantiate_model() constructs the original 15-hidden-layer
        # NN_MultinomialRegression from models/deepgravity.py.
        model = utils.instantiate_model(
            oa2centroid, oa2features, oa2pop, input_width, device=device
        )
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        print("[5/6] Created original DeepGravity network and loaded New York weights")

        metadata = pd.read_csv(dataset_dir / "population.csv", dtype={"oa_ID": str})
        metadata = metadata.set_index("oa_ID").to_dict(orient="index")
        rows = []

        # For each origin, score every candidate destination with DeepGravity.
        # predict_proba() applies Softmax over those scores, so the 191
        # probabilities for an origin sum to one.
        print("[6/6] Computing probabilities for every Hangzhou origin...")
        with torch.no_grad():
            for tile in test_tiles:
                destinations = sorted(tileid2oa2features[str(tile)].keys())
                for origin_number, origin in enumerate(destinations, start=1):
                    features, _targets = dataset.get_X_T([origin], [destinations])
                    probabilities = model.predict_proba(features.to(device))[0]
                    descending = np.argsort(-probabilities)
                    ranks = np.empty_like(descending)
                    ranks[descending] = np.arange(1, len(descending) + 1)
                    origin_meta = metadata[origin]

                    for index, destination in enumerate(destinations):
                        destination_meta = metadata[destination]
                        rows.append(
                            {
                                "tile": str(tile),
                                "origin": origin,
                                "origin_name": origin_meta["name"],
                                "origin_district": origin_meta["district"],
                                "destination": destination,
                                "destination_name": destination_meta["name"],
                                "destination_district": destination_meta["district"],
                                "probability": float(probabilities[index]),
                                "rank_within_origin": int(ranks[index]),
                            }
                        )

                    if origin_number == 1 or origin_number % 10 == 0 or origin_number == len(destinations):
                        print(
                            f"      processed origin {origin_number:3d}/{len(destinations)}: "
                            f"{origin_meta['name']}"
                        )
    finally:
        os.chdir(original_cwd)

    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    probability_sums = result.groupby("origin")["probability"].sum()
    report = {
        "dataset": "hangzhou_inference",
        "model": "DeepGravity NN_MultinomialRegression",
        "weights": "New York checkpoint transferred to Hangzhou",
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "device": str(device),
        "origins": int(result["origin"].nunique()),
        "destinations": int(result["destination"].nunique()),
        "od_probability_rows": len(result),
        "model_input_width": input_width,
        "observed_hangzhou_od_rows": len(od2flow),
        "probability_sum_min_by_origin": float(probability_sums.min()),
        "probability_sum_max_by_origin": float(probability_sums.max()),
        "warning": "These are New-York-to-Hangzhou transfer probabilities, not Hangzhou-calibrated flows.",
    }
    report_path = output_path.with_suffix(".summary.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 72)
    print("Run completed successfully")
    print(f"OD probability rows : {len(result)}")
    print(
        "Probability sums   : "
        f"{probability_sums.min():.9f} to {probability_sums.max():.9f} per origin"
    )
    print(f"Result CSV          : {output_path}")
    print(f"Summary JSON        : {report_path}")
    print("Important: this run used New York weights and no observed Hangzhou OD flows.")


if __name__ == "__main__":
    main()
