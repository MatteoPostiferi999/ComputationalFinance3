import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

# ── PATHS ──────────────────────────────────────────────────────────────────
OPTIONS_PATH = "data/option20230201_20230228.csv"
SPX_PATH = "data/spx_close.csv"

# ── STEP 1: DATA LOADING AND CLEANING ──────────────────────────────────────


def load_and_clean(path: str = OPTIONS_PATH) -> pd.DataFrame:
    """
    Carica il panel di opzioni SPX e applica il preprocessing standard.
    Ritorna un DataFrame pulito con solo le colonne necessarie.
    """

    # ── 1.1 Carica raw data ─────────────────────────────────────────────────
    raw = pd.read_csv(path)
    print(f"[1.1] Raw rows: {len(raw):,}")

    # ── 1.2 Keep calls only ─────────────────────────────────────────────────
    df = raw[raw["cp_flag"] == "C"].copy()
    print(
        f"[1.2] After call filter: {len(df):,} rows "
        f"(dropped {len(raw) - len(df):,})"
    )

    # ── 1.3 Convert dates ───────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"])
    df["exdate"] = pd.to_datetime(df["exdate"])

    # ── 1.4 Rescale strikes: divide by 1000 ─────────────────────────────────
    df["strike_price"] = df["strike_price"] / 1000

    # ── 1.5 Costruisci midquote Vt = (bid + ask) / 2 ────────────────────────
    df["V"] = (df["best_bid"] + df["best_offer"]) / 2

    # ── 1.6 Reporting: missing rates PRIMA del filtro IV ────────────────────
    n = len(df)
    missing_V = df["V"].isna().sum()
    missing_sigma = df["impl_volatility"].isna().sum()
    missing_delta = df["delta"].isna().sum()

    print(f"\n[1.6] Missing rates (before IV filter, n={n:,}):")
    print(f"      V (midquote):       {missing_V:,}  ({100*missing_V/n:.2f}%)")
    print(f"      impl_volatility:    {missing_sigma:,}  ({100*missing_sigma/n:.2f}%)")
    print(f"      delta (vendor):     {missing_delta:,}  ({100*missing_delta/n:.2f}%)")

    # ── 1.7 Unique dates and expiries PRIMA del filtro IV ───────────────────
    print(f"\n[1.7] Unique trading dates: {df['date'].nunique()}")
    print(f"      Unique expiries:       {df['exdate'].nunique()}")

    # ── 1.8 Drop missing o non-positive impl_volatility ─────────────────────
    df = df[df["impl_volatility"].notna() & (df["impl_volatility"] > 0)].copy()
    print(f"\n[1.8] After IV filter: {len(df):,} rows " f"(dropped {n - len(df):,})")

    # ── 1.9 Tieni solo le colonne necessarie ────────────────────────────────
    cols = [
        "date",
        "exdate",
        "symbol",
        "strike_price",
        "best_bid",
        "best_offer",
        "V",
        "impl_volatility",
        "delta",
    ]
    df = df[cols].reset_index(drop=True)

    print(f"\n[1.9] Final columns: {list(df.columns)}")
    print(f"      Final shape:   {df.shape}")

    return df


# ── STEP 2: CONTRACT TIME SERIES AND ONE-DAY OPTION CHANGES ────────────────


def build_contract_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Costruisce le serie temporali per contratto e calcola ΔVt = V(t+1) - V(t).
    Tiene solo le osservazioni dove ΔVt è definito.
    """

    # ── 2.1 Ordina per contratto e data ─────────────────────────────────────
    df = df.sort_values(["symbol", "date"]).copy()

    # ── 2.2 Calcola ΔVt dentro ogni contratto ───────────────────────────────
    # groupby symbol: per ogni opzione, shift(1) prende il giorno precedente
    df["dV"] = df.groupby("symbol")["V"].diff()

    # ── 2.3 Reporting PRIMA del filtro ──────────────────────────────────────
    print(f"\n[2.3] Summary statistics of Vt (before ΔV filter):")
    print(df["V"].describe().round(4))

    n_before = len(df)
    n_contracts_before = df["symbol"].nunique()
    print(f"\n      Rows before ΔV filter:     {n_before:,}")
    print(f"      Contracts before ΔV filter: {n_contracts_before:,}")

    # ── 2.4 Tieni solo righe dove ΔVt è definito ────────────────────────────
    # La prima osservazione di ogni contratto non ha ΔV → droppa
    df = df[df["dV"].notna()].copy()

    # ── 2.5 Reporting DOPO il filtro ────────────────────────────────────────
    n_after = len(df)
    n_contracts_after = df["symbol"].nunique()

    print(f"\n[2.5] After ΔV filter:")
    print(f"      Rows retained:      {n_after:,}")
    print(f"      Contracts retained: {n_contracts_after:,}")

    print(f"\n[2.5] Summary statistics of ΔVt:")
    print(df["dV"].describe().round(4))

    return df


# ── STEP 3: SPX MERGE AND ONE-DAY UNDERLYING CHANGES ───────────────────────


def merge_spx(df: pd.DataFrame, spx_path: str = "data/spx_close.csv") -> pd.DataFrame:
    """
    Carica SPX close, calcola ΔSt, merge nel panel opzioni per data.
    """

    # ── 3.1 Carica SPX ──────────────────────────────────────────────────────
    spx = (
        pd.read_csv(spx_path, parse_dates=["Date"])
        .set_index("Date")["SPX_close"]
        .rename("S")
    )

    # ── 3.2 Calcola ΔSt = S(t+1) - S(t) ────────────────────────────────────
    spx_df = spx.to_frame()
    spx_df["dS"] = spx_df["S"].diff().shift(-1)  # dS(t) = S(t+1) - S(t)

    print(f"\n[3.2] Summary statistics of St:")
    print(spx_df["S"].describe().round(2))
    print(f"\n[3.2] Summary statistics of ΔSt:")
    print(spx_df["dS"].describe().round(2))

    # ── 3.3 Merge nel panel opzioni per data ────────────────────────────────
    spx_df = spx_df.reset_index().rename(columns={"Date": "date"})
    n_before = len(df)

    df = df.merge(spx_df[["date", "S", "dS"]], on="date", how="left")

    # ── 3.4 Merge coverage ──────────────────────────────────────────────────
    matched = df["S"].notna().sum()
    print(f"\n[3.4] Merge coverage:")
    print(f"      Rows before merge: {n_before:,}")
    print(
        f"      Matched to valid (St, ΔSt): {matched:,} ({100*matched/n_before:.2f}%)"
    )
    print(f"      Unmatched: {n_before - matched:,}")

    # ── 3.5 Plot St e ΔSt ───────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    spx_plot = spx_df.dropna(subset=["S"])
    ax1.plot(spx_plot["date"], spx_plot["S"], color="steelblue", linewidth=1.5)
    ax1.set_ylabel("SPX Close (St)")
    ax1.set_title("SPX Close Price — February 2023")
    ax1.grid(True, alpha=0.3)

    spx_plot2 = spx_df.dropna(subset=["dS"])
    ax2.bar(spx_plot2["date"], spx_plot2["dS"], color="steelblue", alpha=0.7, width=0.6)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("ΔSt (one-day change)")
    ax2.set_title("SPX One-Day Changes — February 2023")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/part2/spx_series.png", dpi=150)
    plt.show()
    print("\n[3.5] Plot salvato in outputs/part2/spx_series.png")

    return df


# ── STEP 4: DELTA HEDGING ───────────────────────────────────────────────────

_TRADING_DAYS = 252  # standard convention for one-day Δt
_TAU_FLOOR = 1 / 365  # floor at 1 calendar day to avoid division by zero


def compute_tau(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute time-to-expiry τ in year fractions for each row.

    τ = (exdate - date).days / 365, floored at 1/365 to prevent
    division-by-zero in Black-Scholes formulas when expiry == trade date.

    Adds column: tau
    """
    # ── 4.1 Raw calendar day fraction ───────────────────────────────────────
    df = df.copy()
    df["tau"] = (df["exdate"] - df["date"]).dt.days / 365.0

    # Apply floor: options expiring today get τ = 1 day
    df["tau"] = df["tau"].clip(lower=_TAU_FLOOR)

    print(
        f"\n[4.1] tau computed — min: {df['tau'].min():.6f}  "
        f"max: {df['tau'].max():.4f}  mean: {df['tau'].mean():.4f}"
    )
    return df


def compute_bs_delta(df: pd.DataFrame, r: float, q: float) -> pd.DataFrame:
    """
    Compute the Black-Scholes spot delta for European calls using market IV.

    Forward price:  F = S · exp((r - q) · τ)
    d1 = [ln(F/K) + 0.5 · σ² · τ] / (σ · √τ)
    Δ_BS = exp(-q · τ) · Φ(d1)

    Uses impl_volatility (per-option market IV) as σ.

    Parameters
    ----------
    r : float  Continuously-compounded risk-free rate (annualised, e.g. 0.045)
    q : float  Continuous dividend yield (annualised, e.g. 0.015)

    Adds column: delta_bs
    """
    df = df.copy()

    S = df["S"].values
    K = df["strike_price"].values
    sigma = df["impl_volatility"].values
    tau = df["tau"].values

    # ── 4.2 Forward price ────────────────────────────────────────────────────
    F = S * np.exp((r - q) * tau)

    # ── 4.3 d1 ──────────────────────────────────────────────────────────────
    sigma_sqrt_tau = sigma * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * tau) / sigma_sqrt_tau

    # ── 4.4 Spot delta ───────────────────────────────────────────────────────
    df["delta_bs"] = np.exp(-q * tau) * norm.cdf(d1)

    print(
        f"\n[4.2] delta_bs computed — "
        f"min: {df['delta_bs'].min():.4f}  "
        f"max: {df['delta_bs'].max():.4f}  "
        f"mean: {df['delta_bs'].mean():.4f}"
    )
    print(
        f"      Correlation with vendor delta: "
        f"{df['delta_bs'].corr(df['delta']):.6f}"
    )
    return df


def compute_hedge_move(df: pd.DataFrame, q: float) -> pd.DataFrame:
    """
    Compute the one-day total-return hedge move.

    A delta-hedged portfolio must account for dividends received on the
    underlying position. The total return increment is:

        ΔS_hedge = ΔS + S · (exp(q · Δt) - 1)

    where Δt = 1/252 (one trading day).

    Parameters
    ----------
    q : float  Continuous dividend yield (annualised)

    Adds column: dS_hedge
    """
    df = df.copy()

    dt = 1 / _TRADING_DAYS  # one trading day as year fraction

    # Dividend accrual over one day: S·(e^{qΔt} - 1) ≈ S·q·Δt for small q
    df["dS_hedge"] = df["dS"] + df["S"] * (np.exp(q * dt) - 1)

    print(f"\n[4.3] dS_hedge computed (Δt = 1/{_TRADING_DAYS}):")
    print(f"      mean dS       = {df['dS'].mean():.4f}")
    print(
        f"      mean dS_hedge = {df['dS_hedge'].mean():.4f}  "
        f"(dividend correction ≈ {df['S'].mean() * (np.exp(q * dt) - 1):.4f})"
    )
    return df


def compute_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the delta-hedging residual for each observation.

        ε = ΔV - Δ_BS · ΔS_hedge

    A perfect linear delta hedge would give ε = 0 everywhere; deviations
    capture gamma P&L, vega exposure, and microstructure noise.

    Adds column: eps_bs
    Prints summary statistics of eps_bs.
    """
    df = df.copy()

    df["eps_bs"] = df["dV"] - df["delta_bs"] * df["dS_hedge"]

    print(f"\n[4.4] Hedging residuals (eps_bs) summary:")
    print(df["eps_bs"].describe().round(6))
    return df


def compute_sse_mse(df: pd.DataFrame, eps_col: str) -> tuple[float, float]:
    """
    Compute SSE and MSE for a given residual column.

    Parameters
    ----------
    eps_col : str  Name of the residual column in df (e.g. 'eps_bs')

    Returns
    -------
    sse : float  Sum of squared errors
    mse : float  Mean squared error  (SSE / n)
    """
    eps = df[eps_col].dropna().values
    n = len(eps)

    sse = float(np.sum(eps**2))
    mse = sse / n

    print(f"\n[4.5] SSE/MSE on '{eps_col}' (n={n:,}):")
    print(f"      SSE = {sse:,.4f}")
    print(f"      MSE = {mse:.8f}")
    print(f"      RMSE = {np.sqrt(mse):.6f}")
    return sse, mse


def plot_delta_vs_strike(df: pd.DataFrame) -> None:
    """
    Plot BS delta vs strike for one date / maturity slice and overlay the
    vendor delta for visual comparison.

    Selects trading date 2023-02-14 and the expiry whose τ is closest to
    30 calendar days. Saves to outputs/delta_vs_strike.png.
    """
    TARGET_DATE = pd.Timestamp("2023-02-14")
    TARGET_TAU_DAYS = 30  # calendar days, ~1-month slice

    # ── 4.6a Slice one trading date ──────────────────────────────────────────
    day_df = df[df["date"] == TARGET_DATE].copy()
    if day_df.empty:
        print(f"\n[4.6] WARNING: no data for {TARGET_DATE.date()}, skipping plot.")
        return

    # ── 4.6b Select expiry closest to 30 calendar days ──────────────────────
    expiry_tau = day_df.groupby("exdate")["tau"].first() * 365  # back to calendar days
    chosen_expiry = (expiry_tau - TARGET_TAU_DAYS).abs().idxmin()
    slice_df = day_df[day_df["exdate"] == chosen_expiry].sort_values("strike_price")

    actual_days = int((chosen_expiry - TARGET_DATE).days)
    print(f"\n[4.6] Plotting delta vs strike:")
    print(f"      Date:    {TARGET_DATE.date()}")
    print(f"      Expiry:  {chosen_expiry.date()}  ({actual_days} calendar days)")
    print(f"      Options: {len(slice_df)}")

    # ── 4.6c Plot ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        slice_df["strike_price"],
        slice_df["delta_bs"],
        color="steelblue",
        linewidth=2,
        label="BS delta (impl. vol)",
        zorder=3,
    )
    ax.scatter(
        slice_df["strike_price"],
        slice_df["delta"],
        color="tomato",
        s=20,
        alpha=0.7,
        label="Vendor delta",
        zorder=4,
    )

    # Mark ATM (S ≈ K)
    S_val = slice_df["S"].iloc[0]
    ax.axvline(
        S_val, color="grey", linestyle="--", linewidth=1, label=f"ATM (S={S_val:.0f})"
    )
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)

    ax.set_xlabel("Strike K (index points)")
    ax.set_ylabel("Delta")
    ax.set_title(
        f"BS Delta vs Strike — {TARGET_DATE.date()} | "
        f"Expiry {chosen_expiry.date()} ({actual_days}d)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/part2/delta_vs_strike.png", dpi=150)
    plt.show()
    print("      Saved → outputs/part2/delta_vs_strike.png")


def misspecification_experiment(df: pd.DataFrame) -> None:
    """
    Illustrate the cost of using a wrong flat volatility in the delta hedge.

    For each σ in a grid [0.05, 0.60] (step 0.025):
      1. Recompute d1 and Δ_BS using that flat σ (all options, all dates)
      2. Recompute residuals ε = ΔV - Δ(σ) · ΔS_hedge
      3. Compute SSE(σ)

    Plots SSE(σ), marks the minimiser σ*, and overlays the mean market IV.
    Saves to outputs/misspecification.png.
    """
    # ── 4.7a Pre-extract arrays for speed ───────────────────────────────────
    mask = df[["S", "strike_price", "tau", "dV", "dS_hedge"]].notna().all(axis=1)
    sub = df[mask]

    S = sub["S"].values
    K = sub["strike_price"].values
    tau = sub["tau"].values
    dV = sub["dV"].values
    dS_hedge = sub["dS_hedge"].values

    # Use the same r, q as the main pipeline (stored implicitly via delta_bs
    # already existing; recompute F from scratch with the same constants)
    r, q = 0.045, 0.015
    F = S * np.exp((r - q) * tau)

    # ── 4.7b Grid search ─────────────────────────────────────────────────────
    sigma_grid = np.arange(0.05, 3.5, 0.025)  # 0.05 → 1  .60 inclusive
    sse_values: list[float] = []

    for sigma_flat in sigma_grid:
        sigma_sqrt_tau = sigma_flat * np.sqrt(tau)
        d1 = (np.log(F / K) + 0.5 * sigma_flat**2 * tau) / sigma_sqrt_tau
        delta_flat = np.exp(-q * tau) * norm.cdf(d1)
        eps = dV - delta_flat * dS_hedge
        sse_values.append(float(np.sum(eps**2)))

    sse_arr = np.array(sse_values)
    sigma_star = sigma_grid[np.argmin(sse_arr)]
    mean_iv = df["impl_volatility"].mean()

    print(f"\n[4.7] Misspecification experiment:")
    print(f"      σ* (minimises SSE) = {sigma_star:.3f}")
    print(f"      Mean market IV     = {mean_iv:.3f}")
    print(f"      SSE at σ*          = {sse_arr.min():,.2f}")
    print(
        f"      SSE at mean IV     = {sse_arr[np.argmin(np.abs(sigma_grid - mean_iv))]:,.2f}"
    )

    # ── 4.7c Plot ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(sigma_grid, sse_arr, color="steelblue", linewidth=2, label="SSE(σ)")

    ax.axvline(
        sigma_star,
        color="tomato",
        linestyle="--",
        linewidth=1.5,
        label=f"σ* = {sigma_star:.3f}  (SSE min)",
    )
    ax.axvline(
        mean_iv,
        color="darkorange",
        linestyle=":",
        linewidth=1.5,
        label=f"Mean market IV = {mean_iv:.3f}",
    )

    ax.set_xlabel("Flat volatility σ")
    ax.set_ylabel("SSE")
    ax.set_title("Delta-Hedge SSE vs Flat Volatility (misspecification)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/part2/misspecification.png", dpi=150)
    plt.show()
    print("      Saved → outputs/part2/misspecification.png")


def main():
    """
    Main function executing the entire Part 2-A pipeline.
    This is called by run_full_analysis.py or can be executed directly.
    """
    # Ensure the output directory exists
    os.makedirs("outputs/part2", exist_ok=True)

    # 1. Initial data loading and cleaning
    df = load_and_clean()

    # 2. Contract time series and option delta changes
    df = build_contract_series(df)

    # 3. Merge with underlying SPX data
    df = merge_spx(df)

    # 4. Black-Scholes Delta calculations (Baseline)
    r, q = 0.045, 0.015  # Standard assignment parameters
    df = compute_tau(df)
    df = compute_bs_delta(df, r, q)
    df = compute_hedge_move(df, q)
    df = compute_residuals(df)

    # 5. Global error metrics calculation (SSE/MSE)
    sse, mse = compute_sse_mse(df, "eps_bs")

    # 6. Generation of required plots
    plot_delta_vs_strike(df)
    misspecification_experiment(df)

    # 7. DATA SAVING (Crucial step for the pipeline)
    # Save the processed dataframe so that part2_b can load it for calibration
    output_path = "outputs/part2/processed_options_base.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✅ Pipeline A completed. Data saved to: {output_path}")

    # Return the dataframe for convenience if called programmatically
    return df


# Boilerplate to allow the file to run standalone or be imported as a module
if __name__ == "__main__":
    main()
