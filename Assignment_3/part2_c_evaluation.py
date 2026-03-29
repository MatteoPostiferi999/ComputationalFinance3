"""
chunk_c.py
==========
Step 6 — SABR / Bartlett hedging evaluation and bucketed scoring.
Step 7 — HedgingScoreboard.csv output.

Depends on:
  - chunk_a.py  (data pipeline producing df with BS hedging columns)
  - chunk_b.py  (SABR calibration producing calib_df)
  - sabr_math.py (analytical SABR formulas: delta_sabr, delta_bartlett)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm

from utils_math import (
    delta_sabr,
    delta_bartlett,
    black_delta_F,
    black_vega,
)
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
# Standardised bin edges (mandatory, fixed for competition)
# ══════════════════════════════════════════════════════════════════

# 9 moneyness bins — equal-width over (0.05, 0.95)
DELTA_BINS = np.linspace(0.05, 0.95, 10)

# 7 maturity bins — equal-width over (14, 200)
MATURITY_BINS = np.linspace(14, 200, 8)


# ══════════════════════════════════════════════════════════════════
# Function 1 — Merge calibration parameters onto option panel
# ══════════════════════════════════════════════════════════════════


def merge_calibration(
    df: pd.DataFrame,
    calib_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join calibrated SABR parameters (α, ρ, ν) onto the option panel
    using (date, exdate) as key.  [Step 6.a]

    Parameters
    ----------
    df       : Option panel from chunk_a pipeline.
    calib_df : Calibration results from chunk_b (must contain date, exdate,
               alpha, rho, nu).

    Returns
    -------
    Enriched DataFrame with alpha, rho, nu columns.
    """
    df = df.copy()
    n_before = len(df)

    df = df.merge(
        calib_df[["date", "exdate", "alpha", "rho", "nu"]],
        on=["date", "exdate"],
        how="left",
    )

    n_missing = df["alpha"].isna().sum()
    print(f"\n[6.a] Merged SABR parameters onto panel:")
    print(f"      Rows: {n_before:,}")
    print(
        f"      Missing SABR params: {n_missing:,} ({100 * n_missing / len(df):.1f}%)"
    )

    return df


# ══════════════════════════════════════════════════════════════════
# Function 2 — Compute SABR and Bartlett deltas (row-wise)
# ══════════════════════════════════════════════════════════════════


def compute_sabr_deltas(
    df: pd.DataFrame,
    r: float,
    q: float,
) -> pd.DataFrame:
    """
    Compute Δ_SABR and Δ_Bartlett for each row using analytical
    formulas from sabr_math.py.  [Step 6.b]

    sabr_math functions take scalar F, α, ρ, ν but vectorised K.
    Since these parameters vary per row, we apply row-by-row.
    Rows with missing SABR parameters get NaN deltas.

    Parameters
    ----------
    df : DataFrame with strike_price, S, tau, alpha, rho, nu.
    r  : Risk-free rate.
    q  : Dividend yield.

    Returns
    -------
    DataFrame with new columns: F, delta_sabr, delta_bartlett.
    """
    df = df.copy()

    # Forward price
    df["F"] = df["S"] * np.exp((r - q) * df["tau"])

    # Identify rows with valid SABR params
    has_sabr = df[["alpha", "rho", "nu"]].notna().all(axis=1)

    # Pre-allocate NaN
    df["delta_sabr"] = np.nan
    df["delta_bartlett"] = np.nan

    # Row-wise computation on valid rows only
    valid_idx = df.index[has_sabr]
    print(f"\n[6.b] Computing SABR/Bartlett deltas for {len(valid_idx):,} rows...")

    tqdm.pandas(desc="SABR deltas")

    def _compute_row(row):
        K_arr = np.array([row["strike_price"]])
        d_sabr = delta_sabr(
            K_arr,
            row["F"],
            row["tau"],
            row["alpha"],
            row["rho"],
            row["nu"],
            r,
            q,
        )[0]
        d_bart = delta_bartlett(
            K_arr,
            row["F"],
            row["tau"],
            row["alpha"],
            row["rho"],
            row["nu"],
            r,
            q,
        )[0]
        return pd.Series({"delta_sabr": d_sabr, "delta_bartlett": d_bart})

    results = df.loc[valid_idx].progress_apply(_compute_row, axis=1)
    df.loc[valid_idx, "delta_sabr"] = results["delta_sabr"].values
    df.loc[valid_idx, "delta_bartlett"] = results["delta_bartlett"].values

    # Summary
    for col in ["delta_sabr", "delta_bartlett"]:
        valid = df[col].dropna()
        print(
            f"      {col}: min={valid.min():.4f}  max={valid.max():.4f}  "
            f"mean={valid.mean():.4f}  n_valid={len(valid):,}"
        )

    return df


# ══════════════════════════════════════════════════════════════════
# Function 3 — Apply standardised filters
# ══════════════════════════════════════════════════════════════════


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standardised filters in exact order.  [Step 6.d]

    Filter 1: delta_bs <= 0.05  → drop
    Filter 2: delta_bs >= 0.95  → drop
    Filter 3: D_days  <= 14     → drop
    Filter 4: D_days  >  200    → drop
    Filter 5: SABR missing      → drop rows where alpha/rho/nu is NaN

    Returns
    -------
    Filtered DataFrame.
    """
    df = df.copy()

    # Compute D_days if not present
    if "D_days" not in df.columns:
        df["D_days"] = (df["exdate"] - df["date"]).dt.days

    print(f"\n[6.d] Standardised filters:")
    print(f"      Starting rows: {len(df):,}")

    # Filter 1
    df = df[df["delta_bs"] > 0.05]
    print(f"      After delta_bs > 0.05:    {len(df):,} rows")

    # Filter 2
    df = df[df["delta_bs"] < 0.95]
    print(f"      After delta_bs < 0.95:    {len(df):,} rows")

    # Filter 3
    df = df[df["D_days"] > 14]
    print(f"      After D_days > 14:        {len(df):,} rows")

    # Filter 4
    df = df[df["D_days"] <= 200]
    print(f"      After D_days <= 200:      {len(df):,} rows")

    # Filter 5
    df = df[df[["alpha", "rho", "nu"]].notna().all(axis=1)]
    print(
        f"      After SABR available:     {len(df):,} rows  \u2190 final filtered sample"
    )

    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
# Function 4 — Compute residuals for all three deltas
# ══════════════════════════════════════════════════════════════════


def compute_residuals_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute hedging residuals for BS, SABR, and Bartlett deltas.  [Step 6.e]

        ε = ΔV − Δ · ΔS_hedge

    Returns
    -------
    DataFrame with eps_bs, eps_sabr, eps_bartlett columns.
    """
    df = df.copy()

    df["eps_bs"] = df["dV"] - df["delta_bs"] * df["dS_hedge"]
    df["eps_sabr"] = df["dV"] - df["delta_sabr"] * df["dS_hedge"]
    df["eps_bartlett"] = df["dV"] - df["delta_bartlett"] * df["dS_hedge"]

    print(f"\n[6.e] Residuals computed on {len(df):,} rows:")
    for col in ["eps_bs", "eps_sabr", "eps_bartlett"]:
        print(f"      {col}: mean={df[col].mean():.6f}  std={df[col].std():.4f}")

    return df


# ══════════════════════════════════════════════════════════════════
# Function 5 — Total SSE, MSE, and Gain
# ══════════════════════════════════════════════════════════════════


def compute_sse_mse_gain(df: pd.DataFrame) -> dict:
    """
    Compute total SSE, MSE, and Gains on the final filtered sample.  [Step 6.f]

    Returns
    -------
    Dict with keys: n, SSE_bs, SSE_sabr, SSE_bart, MSE_bs, MSE_sabr,
    MSE_bart, Gain_sabr_vs_bs, Gain_bart_vs_bs, RelGain_bart_vs_sabr.
    """
    n = len(df)

    SSE_bs = float(np.sum(df["eps_bs"].values ** 2))
    SSE_sabr = float(np.sum(df["eps_sabr"].values ** 2))
    SSE_bart = float(np.sum(df["eps_bartlett"].values ** 2))

    MSE_bs = SSE_bs / n
    MSE_sabr = SSE_sabr / n
    MSE_bart = SSE_bart / n

    Gain_sabr_vs_bs = 1.0 - SSE_sabr / SSE_bs
    Gain_bart_vs_bs = 1.0 - SSE_bart / SSE_bs
    RelGain_bart_vs_sabr = 1.0 - SSE_bart / SSE_sabr

    print(f"\n[6.f] Total SSE/MSE/Gain:")
    print(f"      n obs:          {n:,}")
    print(f"      SSE(BS):        {SSE_bs:,.4f}")
    print(f"      SSE(SABR):      {SSE_sabr:,.4f}")
    print(f"      SSE(Bartlett):  {SSE_bart:,.4f}")
    print(f"      MSE(BS):        {MSE_bs:.8f}")
    print(f"      MSE(SABR):      {MSE_sabr:.8f}")
    print(f"      MSE(Bartlett):  {MSE_bart:.8f}")
    print(f"      Gain(SABR vs BS):       {Gain_sabr_vs_bs:.6f}")
    print(
        f"      Gain(Bart vs BS):       {Gain_bart_vs_bs:.6f}   \u2190 competition score"
    )
    print(f"      RelGain(Bart vs SABR):  {RelGain_bart_vs_sabr:.6f}")

    return {
        "n": n,
        "SSE_bs": SSE_bs,
        "SSE_sabr": SSE_sabr,
        "SSE_bart": SSE_bart,
        "MSE_bs": MSE_bs,
        "MSE_sabr": MSE_sabr,
        "MSE_bart": MSE_bart,
        "Gain_sabr_vs_bs": Gain_sabr_vs_bs,
        "Gain_bart_vs_bs": Gain_bart_vs_bs,
        "RelGain_bart_vs_sabr": RelGain_bart_vs_sabr,
    }


# ══════════════════════════════════════════════════════════════════
# Function 6 — Assign moneyness / maturity buckets
# ══════════════════════════════════════════════════════════════════


def assign_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create moneyness and maturity bins using the fixed competition edges.  [Step 6.g]

    Returns
    -------
    DataFrame with bin_delta (1-9) and bin_maturity (1-7) columns.
    Rows outside the bin edges are dropped.
    """
    df = df.copy()

    df["bin_delta"] = pd.cut(
        df["delta_bs"],
        bins=DELTA_BINS,
        labels=range(1, 10),
        include_lowest=True,
    )
    df["bin_maturity"] = pd.cut(
        df["D_days"],
        bins=MATURITY_BINS,
        labels=range(1, 8),
        include_lowest=True,
    )

    n_before = len(df)
    df = df.dropna(subset=["bin_delta", "bin_maturity"]).copy()
    n_dropped = n_before - len(df)

    # Cast to int for clean groupby
    df["bin_delta"] = df["bin_delta"].astype(int)
    df["bin_maturity"] = df["bin_maturity"].astype(int)

    print(f"\n[6.g] Bucket assignment:")
    print(f"      Rows before: {n_before:,}")
    print(f"      Dropped (outside bin edges): {n_dropped:,}")
    print(f"      Rows after:  {len(df):,}")

    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
# Function 7 — Bucketed metrics
# ══════════════════════════════════════════════════════════════════


def compute_bucketed_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SSE, MSE, and Gains per (bin_delta, bin_maturity) bucket.  [Step 6.h]

    Returns
    -------
    DataFrame with one row per (i, j) bucket.
    """

    def _bucket_stats(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        sse_bs = float(np.sum(g["eps_bs"].values ** 2))
        sse_sabr = float(np.sum(g["eps_sabr"].values ** 2))
        sse_bart = float(np.sum(g["eps_bartlett"].values ** 2))

        mse_bs = sse_bs / n if n > 0 else np.nan
        mse_sabr = sse_sabr / n if n > 0 else np.nan
        mse_bart = sse_bart / n if n > 0 else np.nan

        gain_sabr = (1.0 - sse_sabr / sse_bs) if sse_bs > 0 else np.nan
        gain_bart = (1.0 - sse_bart / sse_bs) if sse_bs > 0 else np.nan
        rel_gain = (1.0 - sse_bart / sse_sabr) if sse_sabr > 0 else np.nan

        return pd.Series(
            {
                "n_obs": n,
                "sse_bs": sse_bs,
                "sse_sabr": sse_sabr,
                "sse_bartlett": sse_bart,
                "mse_bs": mse_bs,
                "mse_sabr": mse_sabr,
                "mse_bartlett": mse_bart,
                "gain_sabr_vs_bs": gain_sabr,
                "gain_bart_vs_bs": gain_bart,
                "relgain_bart_vs_sabr": rel_gain,
            }
        )

    grouped = df.groupby(["bin_delta", "bin_maturity"], observed=True)
    bucketed = grouped.apply(_bucket_stats).reset_index()

    print(
        f"\n[6.h] Bucketed metrics: {len(bucketed)} non-empty buckets "
        f"(out of {9 * 7} = 63 possible)"
    )

    return bucketed


# ══════════════════════════════════════════════════════════════════
# Function 8 — Plot delta comparison (BS vs SABR vs Bartlett)
# ══════════════════════════════════════════════════════════════════


def plot_delta_comparison(
    df: pd.DataFrame,
    r: float,
    q: float,
    calib_df: pd.DataFrame,
) -> None:
    """
    Plot K → Δ_BS, Δ_SABR, Δ_Bartlett for one date/expiry slice.  [Step 6.i]

    Uses date=2023-02-14, expiry closest to 30 days.
    Saves to outputs/delta_comparison.png.
    """
    TARGET_DATE = pd.Timestamp("2023-02-14")
    TARGET_D = 30

    day_df = df[df["date"] == TARGET_DATE].copy()
    if day_df.empty:
        print(f"\n[6.i] WARNING: no data for {TARGET_DATE.date()}, skipping plot.")
        return

    # Pick expiry closest to 30 calendar days
    if "D_days" not in day_df.columns:
        day_df["D_days"] = (day_df["exdate"] - day_df["date"]).dt.days

    expiry_D = day_df.groupby("exdate")["D_days"].first()
    chosen_expiry = (expiry_D - TARGET_D).abs().idxmin()
    actual_D = int(expiry_D[chosen_expiry])

    slice_df = day_df[day_df["exdate"] == chosen_expiry].sort_values("strike_price")

    # Must have SABR params
    slice_df = slice_df.dropna(subset=["alpha", "rho", "nu"])
    if slice_df.empty:
        print(f"\n[6.i] WARNING: no SABR data for chosen slice, skipping plot.")
        return

    K_arr = slice_df["strike_price"].values
    F_val = slice_df["F"].iloc[0]

    print(f"\n[6.i] Delta comparison plot:")
    print(
        f"      Date: {TARGET_DATE.date()}, Expiry: {chosen_expiry.date()} ({actual_D}d)"
    )
    print(f"      Strikes: {len(K_arr)}")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        K_arr,
        slice_df["delta_bs"].values,
        "-",
        color="steelblue",
        linewidth=2,
        label="$\\Delta_{BS}$",
        zorder=3,
    )
    ax.plot(
        K_arr,
        slice_df["delta_sabr"].values,
        "--",
        color="tomato",
        linewidth=2,
        label="$\\Delta_{SABR}$",
        zorder=4,
    )
    ax.plot(
        K_arr,
        slice_df["delta_bartlett"].values,
        ":",
        color="forestgreen",
        linewidth=2,
        label="$\\Delta_{Bartlett}$",
        zorder=5,
    )

    ax.axvline(
        F_val, color="grey", linestyle="--", linewidth=1, label=f"ATM (F={F_val:.0f})"
    )
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)

    ax.set_xlabel("Strike K")
    ax.set_ylabel("Delta")
    ax.set_title(
        f"Delta Comparison — {TARGET_DATE.date()} | "
        f"Expiry {chosen_expiry.date()} ({actual_D}d)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/part2/delta_comparison.png", dpi=150)
    plt.show()
    print(f"      Saved \u2192 outputs/part2/delta_comparison.png")


# ══════════════════════════════════════════════════════════════════
# Function 9 — Plot beta comparison
# ══════════════════════════════════════════════════════════════════


def plot_beta_comparison(
    calib_df: pd.DataFrame,
    r: float,
    q: float,
    df: pd.DataFrame,
) -> None:
    """
    Show effect of β ∈ {0, 0.5, 1} on Δ_SABR and Δ_Bartlett.  [Step 6.j]

    For β ≠ 1, a local general SABR vol function is implemented using
    numerical central differences for derivatives, since sabr_math.py
    only implements the β=1 case analytically.

    Uses date=2023-02-14, D closest to 30 days.
    Saves to outputs/beta_comparison.png.
    """
    TARGET_DATE = pd.Timestamp("2023-02-14")
    TARGET_D = 30

    # Find the calibrated slice closest to target
    day_calib = calib_df[calib_df["date"] == TARGET_DATE].copy()
    if day_calib.empty:
        print(f"\n[6.j] WARNING: no calibration for {TARGET_DATE.date()}, skipping.")
        return

    idx = (day_calib["D_days"] - TARGET_D).abs().idxmin()
    row = day_calib.loc[idx]
    alpha, rho, nu = row["alpha"], row["rho"], row["nu"]
    tau = row["tau"]
    D = int(row["D_days"])

    # F is not stored in CalibrationMetrics.csv — reconstruct from the panel
    S_day = df.loc[df["date"] == TARGET_DATE, "S"]
    if S_day.empty:
        print(f"\n[6.j] WARNING: no S value for {TARGET_DATE.date()}, skipping.")
        return
    S_val = float(S_day.iloc[0])
    F = S_val * np.exp((r - q) * tau)

    print(f"\n[6.j] Beta comparison plot:")
    print(f"      Date: {TARGET_DATE.date()}, D={D}d")
    print(f"      \u03b1={alpha:.4f}, \u03c1={rho:.4f}, \u03bd={nu:.4f}")

    # Strike grid
    K_grid = np.linspace(F * 0.85, F * 1.15, 200)

    # ── General SABR vol for arbitrary β (numerical implementation) ──
    # This is needed because sabr_math.py only implements β=1 analytically.
    # We use the Hagan et al. (2002) general-β approximation with the same
    # ATM threshold and numerical stability guards as sabr_math.py.
    ATM_THRESHOLD = 1e-8

    def _general_sabr_vol(K, F, tau, alpha, rho, nu, beta):
        """General SABR implied vol for arbitrary β (Hagan et al. 2002)."""
        K = np.asarray(K, dtype=float)
        log_FK = np.log(F / K)
        atm = np.abs(log_FK) < ATM_THRESHOLD

        FK_mid = (F * K) ** ((1.0 - beta) / 2.0)

        # z and chi
        z = (nu / alpha) * FK_mid * log_FK
        xi = np.sqrt(1.0 - 2.0 * rho * z + z**2)
        log_arg = (xi + z - rho) / (1.0 - rho)
        chi_full = np.log(np.maximum(log_arg, 1e-300))
        chi = np.where(np.abs(z) < 1e-10, z, chi_full)
        chi = np.where(atm, 0.0, chi)

        # Denominator D(K) for non-ATM
        log2 = log_FK**2
        log4 = log_FK**4
        D_denom = FK_mid * (
            1.0 + (1.0 - beta) ** 2 / 24.0 * log2 + (1.0 - beta) ** 4 / 1920.0 * log4
        )

        # I0B
        safe_chi = np.where(atm, 1.0, chi)
        I0B_non_atm = alpha / D_denom * z / safe_chi
        I0B_atm = alpha / F ** (1.0 - beta)
        I0B = np.where(atm, I0B_atm, I0B_non_atm)

        # Higher-order correction I1H
        I1H = (
            (1.0 - beta) ** 2 * alpha**2 / (24.0 * (F * K) ** (1.0 - beta))
            + 0.25 * rho * beta * nu * alpha / FK_mid
            + (2.0 - 3.0 * rho**2) * nu**2 / 24.0
        )

        sigma = I0B * (1.0 + tau * I1H)
        return sigma

    def _general_delta_sabr(K, F, tau, alpha, rho, nu, beta, r, q):
        """SABR delta for general β via numerical ∂σ/∂F."""
        sigma = _general_sabr_vol(K, F, tau, alpha, rho, nu, beta)
        # Numerical Σ_f = ∂σ/∂F via central differences
        h = 1e-4 * F
        sig_up = _general_sabr_vol(K, F + h, tau, alpha, rho, nu, beta)
        sig_dn = _general_sabr_vol(K, F - h, tau, alpha, rho, nu, beta)
        Sigma_f = (sig_up - sig_dn) / (2.0 * h)

        Delta_B = black_delta_F(F, K, tau, sigma, r)
        V_B = black_vega(F, K, tau, sigma, r)

        Delta_SABR_F = Delta_B + V_B * Sigma_f
        return np.exp((r - q) * tau) * Delta_SABR_F

    def _general_delta_bartlett(K, F, tau, alpha, rho, nu, beta, r, q):
        """Bartlett delta for general β via numerical ∂σ/∂F and ∂σ/∂α."""
        sigma = _general_sabr_vol(K, F, tau, alpha, rho, nu, beta)
        h_F = 1e-4 * F
        Sigma_f = (
            _general_sabr_vol(K, F + h_F, tau, alpha, rho, nu, beta)
            - _general_sabr_vol(K, F - h_F, tau, alpha, rho, nu, beta)
        ) / (2.0 * h_F)
        h_a = 1e-4 * alpha
        Sigma_alpha = (
            _general_sabr_vol(K, F, tau, alpha + h_a, rho, nu, beta)
            - _general_sabr_vol(K, F, tau, alpha - h_a, rho, nu, beta)
        ) / (2.0 * h_a)

        Delta_B = black_delta_F(F, K, tau, sigma, r)
        V_B = black_vega(F, K, tau, sigma, r)

        bartlett_adj = Sigma_alpha * rho * nu / F
        Delta_Bart_F = Delta_B + V_B * (Sigma_f + bartlett_adj)
        return np.exp((r - q) * tau) * Delta_Bart_F

    betas = [0, 0.5, 1]
    colors = ["steelblue", "darkorange", "forestgreen"]
    styles = ["-", "--", ":"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for beta, color, ls in zip(betas, colors, styles):
        if beta == 1:
            # Use the exact analytical formulas from sabr_math.py
            d_sabr = delta_sabr(K_grid, F, tau, alpha, rho, nu, r, q)
            d_bart = delta_bartlett(K_grid, F, tau, alpha, rho, nu, r, q)
        else:
            # Use numerical general-β implementation
            d_sabr = _general_delta_sabr(K_grid, F, tau, alpha, rho, nu, beta, r, q)
            d_bart = _general_delta_bartlett(K_grid, F, tau, alpha, rho, nu, beta, r, q)

        ax1.plot(K_grid, d_sabr, ls, color=color, linewidth=2, label=f"$\\beta={beta}$")
        ax2.plot(K_grid, d_bart, ls, color=color, linewidth=2, label=f"$\\beta={beta}$")

    for ax, title in [(ax1, "$\\Delta_{SABR}$"), (ax2, "$\\Delta_{Bartlett}$")]:
        ax.axvline(F, color="grey", linestyle="--", linewidth=1, label=f"F={F:.0f}")
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Delta")
        ax.set_title(
            f"{title} — Effect of $\\beta$\n" f"t={TARGET_DATE.date()}, D={D}d"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/part2/beta_comparison.png", dpi=150)
    plt.show()
    print(f"      Saved \u2192 outputs/part2/beta_comparison.png")


# ══════════════════════════════════════════════════════════════════
# Function 10 — Gain heatmaps
# ══════════════════════════════════════════════════════════════════


def plot_gain_heatmaps(bucketed_df: pd.DataFrame) -> None:
    """
    Two heatmaps: Gain(Bart vs BS) and RelGain(Bart vs SABR) per bucket.  [Step 6.k]

    x-axis: maturity bins 1-7, y-axis: moneyness bins 1-9.
    Diverging colormap centred at 0 (RdYlGn). Empty buckets shown in grey.
    Saves to outputs/gain_heatmaps.png.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    metrics = [
        ("gain_bart_vs_bs", "Gain(Bartlett vs BS)", ax1),
        ("relgain_bart_vs_sabr", "RelGain(Bartlett vs SABR)", ax2),
    ]

    for col, title, ax in metrics:
        # Pivot to (bin_delta × bin_maturity) matrix
        pivot = bucketed_df.pivot(index="bin_delta", columns="bin_maturity", values=col)
        # Reindex to full grid so empty cells show as NaN
        pivot = pivot.reindex(index=range(1, 10), columns=range(1, 8))

        # Diverging norm centred at 0
        vals = pivot.values[~np.isnan(pivot.values)]
        if len(vals) > 0:
            vmax = max(abs(vals.min()), abs(vals.max()), 0.01)
            norm_obj = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        else:
            norm_obj = None

        # Set NaN background to grey
        cmap = plt.cm.RdYlGn.copy()
        cmap.set_bad(color="lightgrey")

        im = ax.imshow(
            pivot.values,
            aspect="auto",
            cmap=cmap,
            norm=norm_obj,
            origin="lower",
        )

        # Annotate cells
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.3f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                    )

        ax.set_xticks(range(7))
        ax.set_xticklabels(range(1, 8))
        ax.set_yticks(range(9))
        ax.set_yticklabels(range(1, 10))
        ax.set_xlabel("Maturity Bin")
        ax.set_ylabel("Moneyness Bin (delta)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig("outputs/part2/gain_heatmaps.png", dpi=150)
    plt.show()
    print(f"\n[6.k] Gain heatmaps saved \u2192 outputs/part2/gain_heatmaps.png")


# ══════════════════════════════════════════════════════════════════
# Function 11 — Save HedgingScoreboard.csv
# ══════════════════════════════════════════════════════════════════


def save_hedging_scoreboard(
    bucketed_df: pd.DataFrame,
    total_metrics: dict,
    path: str = "outputs/HedgingScoreboard.csv",
) -> None:
    """
    Save HedgingScoreboard.csv in exact competition format.  [Step 7]

    Columns: section, bin1, bin2, n_obs,
             sse_bs, sse_sabr, sse_bartlett,
             mse_bs, mse_sabr, mse_bartlett,
             gain_sabr_vs_bs, gain_bart_vs_bs, relgain_bart_vs_sabr
    """
    required_cols = [
        "section",
        "bin1",
        "bin2",
        "n_obs",
        "sse_bs",
        "sse_sabr",
        "sse_bartlett",
        "mse_bs",
        "mse_sabr",
        "mse_bartlett",
        "gain_sabr_vs_bs",
        "gain_bart_vs_bs",
        "relgain_bart_vs_sabr",
    ]

    # BUCKET rows
    bucket_rows = bucketed_df.copy()
    bucket_rows["section"] = "BUCKET"
    bucket_rows = bucket_rows.rename(
        columns={
            "bin_delta": "bin1",
            "bin_maturity": "bin2",
        }
    )

    # TOTAL row
    total_row = pd.DataFrame(
        [
            {
                "section": "TOTAL",
                "bin1": "TOTAL",
                "bin2": "TOTAL",
                "n_obs": total_metrics["n"],
                "sse_bs": total_metrics["SSE_bs"],
                "sse_sabr": total_metrics["SSE_sabr"],
                "sse_bartlett": total_metrics["SSE_bart"],
                "mse_bs": total_metrics["MSE_bs"],
                "mse_sabr": total_metrics["MSE_sabr"],
                "mse_bartlett": total_metrics["MSE_bart"],
                "gain_sabr_vs_bs": total_metrics["Gain_sabr_vs_bs"],
                "gain_bart_vs_bs": total_metrics["Gain_bart_vs_bs"],
                "relgain_bart_vs_sabr": total_metrics["RelGain_bart_vs_sabr"],
            }
        ]
    )

    scoreboard = pd.concat(
        [bucket_rows[required_cols], total_row[required_cols]],
        ignore_index=True,
    )
    scoreboard.to_csv(path, index=False)
    print(f"\n[7] HedgingScoreboard saved \u2192 {path}")
    print(f"    Rows: {len(scoreboard)} ({len(bucketed_df)} buckets + 1 TOTAL)")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def main():
    """
    Main function executing the SABR/Bartlett hedging evaluation (Part 2-C).
    It merges calibration parameters onto the option panel, computes
    advanced deltas, applies standardized filters, and generates the
    final competition scoreboard and heatmaps.
    """
    # Ensure output directory exists
    os.makedirs("outputs/part2", exist_ok=True)

    # Market parameters
    r, q = 0.045, 0.015

    # 1. Load the processed option data from Pipeline A
    input_path = "outputs/part2/processed_options_base.csv"
    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found. Run part2_a_pipeline.py first.")
        return
    df = pd.read_csv(input_path, parse_dates=["date", "exdate"])
    print(f"✅ Option data loaded from {input_path}")

    # 2. Load the SABR calibration parameters from Pipeline B
    calib_path = "outputs/part2/CalibrationMetrics.csv"
    if not os.path.exists(calib_path):
        print(f"❌ Error: {calib_path} not found. Run part2_b_calibration.py first.")
        return
    calib_df = pd.read_csv(calib_path, parse_dates=["date"])

    # Reconstruct exdate for merging purposes
    calib_df["exdate"] = calib_df["date"] + pd.to_timedelta(
        calib_df["D_days"], unit="D"
    )
    print(f"✅ Calibration metrics loaded from {calib_path}")

    # 3. Perform SABR/Bartlett Delta calculations
    df = merge_calibration(df, calib_df)
    df = compute_sabr_deltas(df, r, q)

    # 4. Apply standardized competition filters (D > 14, etc.)
    df = apply_filters(df)
    df = compute_residuals_all(df)

    # 5. Compute aggregate performance metrics (SSE/MSE/Gain)
    total_metrics = compute_sse_mse_gain(df)

    # 6. Assign observations to standardized buckets
    df = assign_buckets(df)
    bucketed_df = compute_bucketed_metrics(df)

    # 7. Generate final report plots
    plot_delta_comparison(df, r, q, calib_df)
    plot_beta_comparison(calib_df, r, q, df)
    plot_gain_heatmaps(bucketed_df)

    # 8. SAVE THE COMPETITION SCOREBOARD
    scoreboard_path = "outputs/part2/HedgingScoreboard.csv"
    save_hedging_scoreboard(bucketed_df, total_metrics, path=scoreboard_path)

    print(f"\n✅ Pipeline C completed. Final scoreboard saved to: {scoreboard_path}")


# Boilerplate for standalone or imported execution
if __name__ == "__main__":
    main()
