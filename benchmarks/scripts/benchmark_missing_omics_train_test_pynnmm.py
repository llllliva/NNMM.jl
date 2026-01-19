#!/usr/bin/env python3
"""
PyNNMM Benchmark Script: Missing Omics in Train vs Test (NNMM.jl parity)

This mirrors `benchmarks/scripts/benchmark_missing_omics_train_test.jl` and is intended
to compare PyNNMM vs NNMM.jl on the same:
  - train/test split (by individual)
  - nested missingness schedule (by individual)
  - MCMC configuration

Important
---------
NNMM.jl uses Julia's `MersenneTwister` shuffles to define the split and missingness
permutations. Python/NumPy RNGs do NOT match Julia's `MersenneTwister` stream, so this
script calls Julia once (stdlib only) to obtain the exact index permutations used by
the NNMM.jl benchmark.

Usage
-----
    # Example (requires a working `nnmm` import, e.g. via PYTHONPATH to a built PyNNMM)
    python benchmarks/scripts/benchmark_missing_omics_train_test_pynnmm.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nnmm import Equation, Layer, NNMMModel


@dataclass(frozen=True)
class JuliaIndexPlan:
    train_perm: list[int]  # 1-based indices into phenotype rows
    test_perm: list[int]   # 1-based indices into phenotype rows


def _parse_pct_list(value: str) -> list[float]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    out = []
    for p in parts:
        out.append(float(p))
    out = [min(max(v, 0.0), 1.0) for v in out]
    return sorted(set(out))


def _call_julia_for_perms(
    *,
    n_individuals: int,
    seed: int,
    test_frac: float,
    julia_bin: str,
) -> JuliaIndexPlan:
    code = f"""
using Random
n = {n_individuals}
seed = {seed}
test_frac = {test_frac}
perm = shuffle(MersenneTwister(seed), collect(1:n))
n_test = round(Int, n * test_frac)
test_idx = perm[1:n_test]
train_idx = perm[n_test+1:end]
train_perm = shuffle(MersenneTwister(seed + 1), train_idx)
test_perm = shuffle(MersenneTwister(seed + 2), test_idx)
println("train_perm=" * join(train_perm, ","))
println("test_perm=" * join(test_perm, ","))
"""

    env = os.environ.copy()
    # Avoid touching user/global depots; stdlib only here.
    env.setdefault("JULIA_DEPOT_PATH", str(Path.cwd() / ".julia_depot"))

    proc = subprocess.run(
        [julia_bin, "-e", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to run Julia to obtain split/missingness permutations.\n"
            f"julia_bin={julia_bin}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
            "Tip: set JULIA_BIN to a direct Julia binary (not the juliaup launcher)."
        )

    train_perm: list[int] | None = None
    test_perm: list[int] | None = None
    for line in proc.stdout.splitlines():
        if line.startswith("train_perm="):
            train_perm = [int(x) for x in line.split("=", 1)[1].split(",") if x]
        if line.startswith("test_perm="):
            test_perm = [int(x) for x in line.split("=", 1)[1].split(",") if x]

    if not train_perm or not test_perm:
        raise RuntimeError(f"Could not parse Julia output:\n{proc.stdout}")

    return JuliaIndexPlan(train_perm=train_perm, test_perm=test_perm)


def _apply_individual_missingness(
    omics_df: pd.DataFrame,
    omic_cols: list[str],
    indices_1based: list[int],
) -> int:
    if not indices_1based:
        return 0
    idx0 = [i - 1 for i in indices_1based]
    omics_df.loc[idx0, omic_cols] = np.nan
    return len(indices_1based) * len(omic_cols)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyNNMM train/test missing-omics benchmark (parity with NNMM.jl)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chain-length", type=int, default=1000)
    parser.add_argument("--burnin", type=int, default=200)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--train-missing-pcts", type=str, default="0,0.25,0.5,0.75,1")
    parser.add_argument("--test-missing-pcts", type=str, default="0,1")
    parser.add_argument("--missing-mode", type=str, default="individual", choices=("individual",))
    parser.add_argument("--estimate-pi", type=int, default=1)
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument("--julia-bin", type=str, default=os.environ.get("JULIA_BIN", "julia"))
    parser.add_argument("--geno-path", type=str, default="")
    parser.add_argument("--pheno-path", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    results_dir = repo_root / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    geno_path = Path(args.geno_path) if args.geno_path else (repo_root / "src" / "datasets" / "data" / "simulated_omics_data" / "genotypes_1000snps.txt")
    pheno_path = Path(args.pheno_path) if args.pheno_path else (repo_root / "src" / "datasets" / "data" / "simulated_omics_data" / "phenotypes_sim.txt")

    pheno_df = pd.read_csv(pheno_path, sep=",")
    pheno_df["ID"] = pheno_df["ID"].astype(str)
    n_individuals = len(pheno_df)

    if not (0.0 < args.test_frac < 1.0):
        raise ValueError(f"--test-frac must be in (0,1), got {args.test_frac}")

    train_pcts = _parse_pct_list(args.train_missing_pcts)
    test_pcts = _parse_pct_list(args.test_missing_pcts)

    # Obtain the exact split/missingness permutations used in the Julia benchmark.
    plan = _call_julia_for_perms(
        n_individuals=n_individuals,
        seed=args.seed,
        test_frac=args.test_frac,
        julia_bin=args.julia_bin,
    )

    test_idx_1based = plan.test_perm[:]  # complete set
    test_id_set = set(pheno_df.loc[[i - 1 for i in test_idx_1based], "ID"].tolist())
    n_test = len(test_idx_1based)
    n_train = n_individuals - n_test

    omic_cols = [f"omic{i}" for i in range(1, 11)]

    rows = []
    for train_missing_pct in train_pcts:
        for test_missing_pct in test_pcts:
            # Nested missingness like Julia: take the first round(n*pct) from the fixed permutation.
            n_train_miss = int(round(n_train * train_missing_pct))
            n_test_miss = int(round(n_test * test_missing_pct))
            train_miss_idx = plan.train_perm[:n_train_miss]
            test_miss_idx = plan.test_perm[:n_test_miss]

            tmp_dir = Path(tempfile.mkdtemp(prefix="pynnmm_missing_omics_train_test_"))
            try:
                omics_df = pheno_df[["ID"] + omic_cols].copy()
                train_missing_cells = _apply_individual_missingness(omics_df, omic_cols, train_miss_idx)
                test_missing_cells = _apply_individual_missingness(omics_df, omic_cols, test_miss_idx)

                omics_path = tmp_dir / "omics.csv"
                omics_df.to_csv(omics_path, index=False, na_rep="NA")

                pheno_out_df = pheno_df[["ID", "trait1"]].copy()
                pheno_out_df.loc[[i - 1 for i in test_idx_1based], "trait1"] = np.nan
                pheno_out_path = tmp_dir / "phenotypes.csv"
                pheno_out_df.to_csv(pheno_out_path, index=False, na_rep="NA")

                layers = [
                    Layer(name="geno", data_path=str(geno_path), separator=","),
                    Layer(name="omics", data_path=str(omics_path), separator=",", missing_value="NA"),
                    Layer(name="phenotypes", data_path=str(pheno_out_path), separator=",", missing_value="NA"),
                ]
                equations = [
                    Equation(
                        from_layer="geno",
                        to_layer="omics",
                        equation="omics = intercept + geno",
                        method="BayesC",
                    ),
                    Equation(
                        from_layer="omics",
                        to_layer="phenotypes",
                        equation="phenotypes = intercept + omics",
                        method="BayesC",
                        activation="linear",
                    ),
                ]

                model = NNMMModel(layers, equations)
                output_folder = str(tmp_dir / "output")
                start = time.time()
                result = model.run(
                    chain_length=args.chain_length,
                    burnin=args.burnin,
                    estimate_pi=bool(args.estimate_pi),
                    seed=args.seed,
                    output_folder=output_folder,
                )
                elapsed = time.time() - start

                ebv_ids = [str(x) for x in result.get("ID", [])]
                ebv_vals = np.asarray(result.get("EBV_NonLinear", []), dtype=float)
                if len(ebv_ids) != ebv_vals.size or ebv_vals.size == 0:
                    raise RuntimeError("PyNNMM returned empty EBV output.")
                ebv_df = pd.DataFrame({"ID": ebv_ids, "EBV": ebv_vals})

                truth_df = pheno_df[["ID", "genetic_total", "genetic_direct", "genetic_indirect", "trait1"]].copy()
                merged = ebv_df.merge(truth_df, on="ID", how="inner")
                merged_test = merged[merged["ID"].isin(test_id_set)]

                ebv_test_total = _corr(merged_test["EBV"].to_numpy(), merged_test["genetic_total"].to_numpy())
                ebv_test_direct = _corr(merged_test["EBV"].to_numpy(), merged_test["genetic_direct"].to_numpy())
                ebv_test_indirect = _corr(merged_test["EBV"].to_numpy(), merged_test["genetic_indirect"].to_numpy())

                # EPV on test individuals:
                # - if test omics are observed (0%), use observed omics
                # - if test omics are fully missing (100%), use EBV as a proxy (matches Julia EPV_Output intent)
                alpha2 = np.asarray(result.get("alpha2", []), dtype=float).reshape(-1)
                if alpha2.size != len(omic_cols):
                    raise RuntimeError(f"Unexpected alpha2 size: {alpha2.size}, expected {len(omic_cols)}")

                if test_missing_pct == 0.0:
                    omics_test = pheno_df.loc[pheno_df["ID"].isin(test_id_set), omic_cols].to_numpy(dtype=float)
                    epv_test_vals = omics_test @ alpha2
                    epv_truth = pheno_df.loc[pheno_df["ID"].isin(test_id_set), ["genetic_total", "genetic_direct", "genetic_indirect", "trait1"]].copy()
                else:
                    # Use EBV values for the matched test set order.
                    epv_test_vals = merged_test["EBV"].to_numpy(dtype=float)
                    epv_truth = merged_test[["genetic_total", "genetic_direct", "genetic_indirect", "trait1"]].copy()

                epv_test_total = _corr(epv_test_vals, epv_truth["genetic_total"].to_numpy())
                epv_test_direct = _corr(epv_test_vals, epv_truth["genetic_direct"].to_numpy())
                epv_test_indirect = _corr(epv_test_vals, epv_truth["genetic_indirect"].to_numpy())
                epv_test_trait = _corr(epv_test_vals, epv_truth["trait1"].to_numpy())

                rows.append(
                    {
                        "seed": args.seed,
                        "chain_length": args.chain_length,
                        "burnin": args.burnin,
                        "test_frac": args.test_frac,
                        "missing_mode": args.missing_mode,
                        "train_missing_pct": float(train_missing_pct),
                        "test_missing_pct": float(test_missing_pct),
                        "n_train": int(n_train),
                        "n_test": int(n_test),
                        "train_missing_cells": int(train_missing_cells),
                        "test_missing_cells": int(test_missing_cells),
                        "ebv_test_total": float(ebv_test_total),
                        "ebv_test_direct": float(ebv_test_direct),
                        "ebv_test_indirect": float(ebv_test_indirect),
                        "epv_test_total": float(epv_test_total),
                        "epv_test_direct": float(epv_test_direct),
                        "epv_test_indirect": float(epv_test_indirect),
                        "epv_test_trait": float(epv_test_trait),
                        "time_seconds": float(elapsed),
                    }
                )
                print(
                    f"train={int(round(train_missing_pct*100))}% test={int(round(test_missing_pct*100))}% "
                    f"EBV(test,total)={ebv_test_total:.4f} EPV(test,total)={epv_test_total:.4f} "
                    f"time={elapsed:.1f}s"
                )
            finally:
                # Always remove temp dirs; this benchmark is for timing only.
                try:
                    for child in tmp_dir.glob("**/*"):
                        pass
                finally:
                    import shutil

                    shutil.rmtree(tmp_dir, ignore_errors=True)

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["train_missing_pct", "test_missing_pct"]).reset_index(drop=True)

    if args.output_csv:
        out_path = Path(args.output_csv)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = results_dir / f"missing_omics_train_test_results_pynnmm_{ts}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved PyNNMM results to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

