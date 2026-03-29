"""
chunk_b.py
==========
SABR calibration module — calibrates (α, ρ, ν) per smile slice
using analytical gradients from sabr_math.py.

Appendix references: A.10 (SABR vol), A.5–A.6 (gradients).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm

from utils_math import sabr_vol, sabr_vol_dalpha, sabr_vol_drho, sabr_vol_dnu
from part2_a_pipeline import (
    load_and_clean,
    build_contract_series,
    merge_spx,
    compute_tau,
    compute_bs_delta,
    compute_hedge_move,
    compute_residuals,
)


# ══════════════════════════════════════════════════════════════════
# Function 1 — Build smile slices
# ══════════════════════════════════════════════════════════════════


def build_slices(
    df: pd.DataFrame,
    r: float,
    q: float,
    D_max: int = 200,
) -> list[dict]:
    """
    Extract (date, exdate) smile slices for SABR calibration.

    For each unique (date, exdate) pair:
      - Compute D = (exdate - date).days
      - Keep slices with 1 <= D <= D_max
      - Compute forward F = S · exp((r-q)·τ)
      - Collect strikes K and market implied vols σ_mkt
      - Drop slices with fewer than 3 valid strikes

    Parameters
    ----------
    df    : DataFrame from chunk_a pipeline (must contain S, tau, strike_price, impl_volatility)
    r     : risk-free rate
    q     : dividend yield
    D_max : maximum days-to-expiry to include

    Returns
    -------
    List of slice dicts with keys: date, exdate, D, tau, F, K, sigma_mkt, n_strikes
    """
    print("\n[5.1] Building smile slices for SABR calibration...")

    slices: list[dict] = []
    total_before = 0
    total_after = 0

    for (date, exdate), grp in df.groupby(["date", "exdate"]):
        D = (exdate - date).days
        if D < 1 or D > D_max:
            continue

        slice_data = grp
        total_before += len(slice_data)

        # Skip slice if fewer than 3 strikes remain after filtering
        if len(slice_data) < 3:
            continue

        total_after += len(slice_data)

        tau_val = slice_data["tau"].iloc[0]
        S_val = slice_data["S"].iloc[0]
        F_val = S_val * np.exp((r - q) * tau_val)

        K = slice_data["strike_price"].values.copy()
        sigma_mkt = slice_data["impl_volatility"].values.copy()

        slices.append(
            {
                "date": date,
                "exdate": exdate,
                "D": D,
                "tau": tau_val,
                "F": F_val,
                "K": K,
                "sigma_mkt": sigma_mkt,
                "n_strikes": len(K),
            }
        )

    n_removed = total_before - total_after
    pct_removed = 100.0 * n_removed / total_before if total_before > 0 else 0.0
    print(f"[5.1] Total slices: {len(slices)}")
    print(
        f"      D range: {min(s['D'] for s in slices)}–{max(s['D'] for s in slices)} days"
    )
    print(
        f"      Strikes per slice: "
        f"min={min(s['n_strikes'] for s in slices)}, "
        f"max={max(s['n_strikes'] for s in slices)}, "
        f"mean={np.mean([s['n_strikes'] for s in slices]):.1f}"
    )

    return slices


# ══════════════════════════════════════════════════════════════════
# Function 2 — Calibrate a single slice
# ══════════════════════════════════════════════════════════════════


def calibrate_slice(
    slice_dict: dict,
    r: float,
) -> Optional[dict]:
    """
    Calibrate SABR parameters (α, ρ, ν) for one smile slice via L-BFGS-B
    with analytical gradient.

    Objective (eq. A.10):
        L(α, ρ, ν) = Σ_K [σ_mkt(K) − σ_SABR(K; F, τ, α, ρ, ν)]²

    Uses 3 restarts with different initialisations; keeps the best result.

    Parameters
    ----------
    slice_dict : dict from build_slices
    r          : risk-free rate (kept for interface consistency)

    Returns
    -------
    Dict with calibrated parameters and diagnostics, or None on failure.
    """
    K = slice_dict["K"]
    sigma_mkt = slice_dict["sigma_mkt"]
    F = slice_dict["F"]
    tau = slice_dict["tau"]

    bounds = [(1e-4, 5.0), (-0.999, 0.999), (1e-4, 5.0)]

    alpha_0 = float(np.mean(sigma_mkt))

    # Multiple restarts
    init_params = [
        (alpha_0, -0.5, 0.4),
        (alpha_0, -0.3, 0.6),
        (alpha_0, -0.7, 0.3),
    ]

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        alpha, rho, nu = params
        sigma_sabr = sabr_vol(K, F, tau, alpha, rho, nu)
        residuals = sigma_mkt - sigma_sabr
        loss = float(np.sum(residuals**2))

        grad_alpha = -2.0 * np.sum(
            residuals * sabr_vol_dalpha(K, F, tau, alpha, rho, nu)
        )
        grad_rho = -2.0 * np.sum(residuals * sabr_vol_drho(K, F, tau, alpha, rho, nu))
        grad_nu = -2.0 * np.sum(residuals * sabr_vol_dnu(K, F, tau, alpha, rho, nu))

        return loss, np.array([grad_alpha, grad_rho, grad_nu])

    best_result = None
    best_loss = np.inf

    for x0 in init_params:
        try:
            res = minimize(
                objective,
                x0=np.array(x0),
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
            )
            if res.success and np.isfinite(res.fun) and res.fun < best_loss:
                best_loss = res.fun
                best_result = res
        except Exception:
            continue

    if best_result is None:
        return None

    alpha_hat, rho_hat, nu_hat = best_result.x

    # Compute RMSE at calibrated params
    sigma_fit = sabr_vol(K, F, tau, alpha_hat, rho_hat, nu_hat)
    rmse = float(np.sqrt(np.mean((sigma_mkt - sigma_fit) ** 2)))

    return {
        "date": slice_dict["date"],
        "exdate": slice_dict["exdate"],
        "D_days": slice_dict["D"],
        "tau": tau,
        "F": F,
        "n_strikes": slice_dict["n_strikes"],
        "alpha": alpha_hat,
        "rho": rho_hat,
        "nu": nu_hat,
        "vol_rmse": rmse,
    }


# ══════════════════════════════════════════════════════════════════
# Function 3 — Run calibration across all slices
# ══════════════════════════════════════════════════════════════════


def run_calibration(
    df: pd.DataFrame,
    r: float,
    q: float,
) -> pd.DataFrame:
    """
    Run SABR calibration on all valid smile slices.

    Parameters
    ----------
    df : DataFrame from chunk_a pipeline
    r  : risk-free rate
    q  : dividend yield

    Returns
    -------
    calib_df : DataFrame with one row per successfully calibrated slice
    """
    print("\n[5.2] Running SABR calibration...")

    slices = build_slices(df, r, q)
    results: list[dict] = []

    for s in tqdm(slices, desc="Calibrating slices"):
        res = calibrate_slice(s, r)
        if res is not None:
            results.append(res)

    calib_df = pd.DataFrame(results)

    n_total = len(slices)
    n_ok = len(results)
    n_fail = n_total - n_ok

    print(f"\n[5.2] Calibration summary:")
    print(f"      Total slices attempted:  {n_total}")
    print(f"      Successfully calibrated: {n_ok}")
    print(f"      Failed:                  {n_fail}")
    print(f"      Mean RMSE:               {calib_df['vol_rmse'].mean():.6f}")

    return calib_df


# ══════════════════════════════════════════════════════════════════
# Function 4 — Save calibration metrics
# ══════════════════════════════════════════════════════════════════


def save_calibration_metrics(
    calib_df: pd.DataFrame,
    path: str = "outputs/part2/CalibrationMetrics.csv",
) -> None:
    """
    Save calibration results to CSV with competition column names.

    Columns: date, D_days, tau, n_strikes, alpha, rho, nu, vol_rmse
    """
    cols = ["date", "D_days", "tau", "n_strikes", "alpha", "rho", "nu", "vol_rmse"]
    calib_df[cols].to_csv(path, index=False)
    print(f"\n[5.3] Calibration metrics saved → {path}")


# ══════════════════════════════════════════════════════════════════
# Function 5 — Plot smile fit
# ══════════════════════════════════════════════════════════════════


def plot_smile_fit(
    df: pd.DataFrame,
    calib_df: pd.DataFrame,
    r: float,
    q: float,
) -> None:
    """
    Plot market smile vs SABR fitted smile for the slice with the most strikes.

    Saves to outputs/part2/sabr_smile_fit.png.
    """
    # Pick the slice with the most strikes
    idx = calib_df["n_strikes"].idxmax()
    row = calib_df.loc[idx]

    date = row["date"]
    exdate = row["exdate"]
    alpha, rho, nu = row["alpha"], row["rho"], row["nu"]
    F = row["F"]
    tau = row["tau"]
    D = row["D_days"]

    # Extract market data for this slice
    mask = (df["date"] == date) & (df["exdate"] == exdate)
    slice_df = df[mask].sort_values("strike_price")

    K_mkt = slice_df["strike_price"].values
    sigma_mkt = slice_df["impl_volatility"].values

    # Filter valid vols
    valid = np.isfinite(sigma_mkt) & (sigma_mkt > 0)
    K_mkt = K_mkt[valid]
    sigma_mkt = sigma_mkt[valid]

    # SABR fitted smile on a fine grid
    K_grid = np.linspace(K_mkt.min(), K_mkt.max(), 200)
    sigma_sabr = sabr_vol(K_grid, F, tau, alpha, rho, nu)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        K_mkt,
        sigma_mkt,
        "o-",
        color="steelblue",
        markersize=4,
        linewidth=1.5,
        label="Market $\\sigma_{mkt}$",
        zorder=3,
    )
    ax.plot(
        K_grid,
        sigma_sabr,
        "--",
        color="tomato",
        linewidth=2,
        label="SABR fit",
        zorder=4,
    )

    ax.axvline(F, color="grey", linestyle=":", linewidth=1, label=f"F = {F:.0f}")

    ax.set_xlabel("Strike K")
    ax.set_ylabel("Implied Volatility")
    ax.set_title(
        f"SABR Smile Fit — {pd.Timestamp(date).date()} | "
        f"Expiry {pd.Timestamp(exdate).date()} ({D}d)\n"
        f"$\\alpha$={alpha:.4f}, $\\rho$={rho:.4f}, $\\nu$={nu:.4f}, "
        f"RMSE={row['vol_rmse']:.5f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/part2/sabr_smile_fit.png", dpi=150)
    plt.show()
    print(f"\n[5.4] Smile fit plot saved → outputs/part2/sabr_smile_fit.png")


# ══════════════════════════════════════════════════════════════════
# Function 6 — Plot parameter time series
# ══════════════════════════════════════════════════════════════════


def plot_params_timeseries(calib_df: pd.DataFrame) -> None:
    """
    Plot α and ν over time for the two most common maturities (D_days).

    Saves to outputs/sabr_params_timeseries.png.
    """
    # Find two most common D_days values
    top_D = calib_df["D_days"].value_counts().head(2).index.tolist()
    print(f"\n[5.5] Plotting parameter time series for D = {top_D}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for D_val in top_D:
        sub = calib_df[calib_df["D_days"] == D_val].sort_values("date")
        ax1.plot(
            sub["date"],
            sub["alpha"],
            "o-",
            markersize=4,
            linewidth=1.5,
            label=f"D = {D_val}d",
        )
        ax2.plot(
            sub["date"],
            sub["nu"],
            "o-",
            markersize=4,
            linewidth=1.5,
            label=f"D = {D_val}d",
        )

    ax1.set_ylabel("$\\alpha$ (vol level)")
    ax1.set_title("SABR $\\alpha$ over Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("$\\nu$ (vol-of-vol)")
    ax2.set_xlabel("Date")
    ax2.set_title("SABR $\\nu$ over Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/part2/sabr_params_timeseries.png", dpi=150)
    plt.show()
    print(
        f"[5.5] Parameter time series saved → outputs/part2/sabr_params_timeseries.png"
    )


# ══════════════════════════════════════════════════════════════════
# Function 7 — Plot calibration error distribution
# ══════════════════════════════════════════════════════════════════


def plot_calibration_error(calib_df: pd.DataFrame) -> None:
    """
    Plot histogram of RMSE across all calibrated slices.

    Saves to outputs/sabr_calibration_error.png.
    """
    rmse_vals = calib_df["vol_rmse"].values
    mean_rmse = float(np.mean(rmse_vals))
    median_rmse = float(np.median(rmse_vals))
    max_rmse = float(np.max(rmse_vals))
    pct_below_005 = float(np.mean(rmse_vals < 0.005) * 100)

    print(f"\n[5.6] Calibration error summary:")
    print(f"      Mean RMSE:             {mean_rmse:.6f}")
    print(f"      Median RMSE:           {median_rmse:.6f}")
    print(f"      Max RMSE:              {max_rmse:.6f}")
    print(f"      % slices RMSE < 0.005: {pct_below_005:.1f}%")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(rmse_vals, bins=40, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(
        mean_rmse,
        color="tomato",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_rmse:.5f}",
    )
    ax.axvline(
        median_rmse,
        color="darkorange",
        linestyle=":",
        linewidth=2,
        label=f"Median = {median_rmse:.5f}",
    )

    ax.set_xlabel("RMSE (implied vol)")
    ax.set_ylabel("Count")
    ax.set_title("SABR Calibration RMSE Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/part2/sabr_calibration_error.png", dpi=150)
    plt.show()
    print(
        f"[5.6] Calibration error plot saved → outputs/part2/sabr_calibration_error.png"
    )


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════


def main():
    """
    Main function executing the SABR calibration pipeline (Part 2-B).
    It loads the processed data from Pipeline A, runs the calibration
    across all slices, and saves the resulting parameters.
    """
    # Ensure output directory exists
    os.makedirs("outputs/part2", exist_ok=True)

    # Market parameters
    r, q = 0.045, 0.015

    # 1. Load the processed data generated in Part 2-A
    # IMPORTANT: We load the CSV saved by the previous script
    input_path = "outputs/part2/processed_options_base.csv"
    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found. Run part2_a_pipeline.py first.")
        return

    df = pd.read_csv(input_path, parse_dates=["date", "exdate"])
    print(f"✅ Data loaded successfully from {input_path}")

    # 2. Run SABR calibration across all valid smile slices
    calib_df = run_calibration(df, r, q)

    # 3. Save calibrated parameters (α, ρ, ν) to CSV
    save_calibration_metrics(calib_df)

    # 4. Generate diagnostic plots
    plot_smile_fit(df, calib_df, r, q)
    plot_params_timeseries(calib_df)
    plot_calibration_error(calib_df)

    print("\n✅ Pipeline B completed. Calibration results saved to outputs/part2/.")

    # Optional: return results for programmatic use
    return calib_df


# Standard boilerplate for standalone or imported execution
if __name__ == "__main__":
    main()
