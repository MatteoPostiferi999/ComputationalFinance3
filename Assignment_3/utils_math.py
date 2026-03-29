"""
sabr_math.py
============
Analytical SABR formulas for β=1, θ=0 (lognormal SABR, no shift).

All equation references are to the assignment Appendix B (A.x.y notation).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

# ──────────────────────────────────────────────────────────────────
# Fixed model parameters
# ──────────────────────────────────────────────────────────────────
BETA  = 1   # lognormal SABR
THETA = 0   # no shift

# Threshold below which |log(F/K)| is treated as ATM
ATM_THRESHOLD = 1e-8


# ══════════════════════════════════════════════════════════════════
# Group 0 — Auxiliary quantities
# ══════════════════════════════════════════════════════════════════

def _compute_auxiliary(
    K: np.ndarray | float,
    F: float,
    alpha: float,
    rho: float,
    nu: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute shared auxiliary quantities used across all SABR formulas.

    Parameters
    ----------
    K     : array-like or float — strike(s)
    F     : float               — forward price
    alpha : float               — initial vol level α
    rho   : float               — correlation ρ
    nu    : float               — vol-of-vol ν

    Returns
    -------
    x   : log(F/K)                         — log-moneyness
    z   : (ν/α)·x                          — β=1 case, eq.(31)
    xi  : sqrt(1 − 2ρz + z²)
    chi : log((ξ + z − ρ) / (1 − ρ))
    I1H : ρναT/4 + (2−3ρ²)ν²/24           — A.10 higher-order term, β=1

    ATM handling
    ------------
    |x| < ATM_THRESHOLD  →  z=0, ξ=1, χ=0  (limiting values).
    |z| < 1e-10          →  χ ≈ z  (first-order Taylor, prevents log(0)).
    """
    K = np.asarray(K, dtype=float)

    # ── log-moneyness and SABR z-transform ──────────────────────
    x  = np.log(F / K)        # log-moneyness
    z  = (nu / alpha) * x     # β=1 specialisation of eq.(31)

    # ── ξ = sqrt(1 − 2ρz + z²) ─────────────────────────────────
    xi = np.sqrt(1.0 - 2.0 * rho * z + z**2)

    # ── χ with numerical guard for small |z| ────────────────────
    # Exact formula: χ = log((ξ + z − ρ) / (1 − ρ))
    # Taylor limit:  χ ≈ z  as  z → 0
    log_arg  = (xi + z - rho) / (1.0 - rho)
    chi_full = np.log(np.maximum(log_arg, 1e-300))   # guard against log(0)
    chi = np.where(np.abs(z) < 1e-10, z, chi_full)

    # ── ATM override: impose limiting values ─────────────────────
    atm = np.abs(x) < ATM_THRESHOLD
    xi  = np.where(atm, 1.0, xi)
    chi = np.where(atm, 0.0, chi)

    # ── I¹_H (higher-order correction), A.10, β=1 ───────────────
    I1H = rho * nu * alpha / 4.0 + (2.0 - 3.0 * rho**2) * nu**2 / 24.0

    return x, z, xi, chi, I1H


# ══════════════════════════════════════════════════════════════════
# Group 1 — SABR implied volatility
# ══════════════════════════════════════════════════════════════════

def sabr_vol(
    K: np.ndarray | float,
    F: float,
    tau: float,
    alpha: float,
    rho: float,
    nu: float,
) -> np.ndarray | float:
    """
    SABR implied volatility for β=1, θ=0.  (A.10)

    Non-ATM  (|x| > ATM_THRESHOLD):
        σ = α · (z/χ) · (1 + τ·I¹_H)

    ATM  (|x| ≤ ATM_THRESHOLD):
        σ = α · (1 + τ·I¹_H)

    Parameters
    ----------
    K     : array-like or float — strike(s)
    F     : float               — forward
    tau   : float               — time to expiry (years)
    alpha : float               — SABR α
    rho   : float               — SABR ρ
    nu    : float               — SABR ν

    Returns
    -------
    Implied vol, scalar or array matching shape of K.
    """
    K = np.asarray(K, dtype=float)
    x, z, xi, chi, I1H = _compute_auxiliary(K, F, alpha, rho, nu)

    atm = np.abs(x) < ATM_THRESHOLD

    # I⁰_B: α·(z/χ) for non-ATM; α for ATM (L'Hôpital: z/χ → 1)
    safe_chi = np.where(atm, 1.0, chi)        # prevent 0/0 in ATM branch
    I0B = np.where(atm, alpha, alpha * z / safe_chi)

    sigma = I0B * (1.0 + tau * I1H)

    return float(sigma) if sigma.ndim == 0 else sigma


# ══════════════════════════════════════════════════════════════════
# Group 2 — Derivatives of I⁰_B  (private building blocks)
# ══════════════════════════════════════════════════════════════════

def _dI0B_dalpha(
    x: np.ndarray,
    z: np.ndarray,
    xi: np.ndarray,
    chi: np.ndarray,
    alpha: float,
    nu: float,
) -> np.ndarray:
    """
    ∂I⁰_B/∂α for β=1.  (A.5.3 eq.(65))

    Derivation: I⁰_B = ν·x/χ; differentiating through z = ν·x/α yields
        ∂(I⁰_B)/∂α = ν²x² / (α²·ξ·χ²)  ≡  ν·x·z / (α·ξ·χ²)

    Non-ATM:  ν·x·z / (α·ξ·χ²)
    ATM:      1   (A.5.1 eq.(55), K^(β−1) = 1 for β=1)
    """
    atm = np.abs(x) < ATM_THRESHOLD

    # Safe denominators — replaced with 1 in ATM branch so np.where
    # does not produce NaN even though we discard the non-ATM result.
    safe_chi2 = np.where(atm, 1.0, chi**2)
    safe_xi   = np.where(atm, 1.0, xi)

    non_atm = nu * x * z / (alpha * safe_xi * safe_chi2)
    return np.where(atm, 1.0, non_atm)


def _dI0B_drho(
    x: np.ndarray,
    z: np.ndarray,
    xi: np.ndarray,
    chi: np.ndarray,
    rho: float,
    nu: float,
) -> np.ndarray:
    """
    ∂I⁰_B/∂ρ for β=1.  (A.5.3 eq.(68))

        Non-ATM:  ν·x · [(ρ−1)·(z+ξ) + (ξ+z−ρ)·ξ]
                  ──────────────────────────────────────
                  (ρ−1)·(ξ+z−ρ)·ξ·χ²

        ATM:      0   (A.5.1 eq.(55))

    The denominator factor (ρ−1) ≠ 0 for |ρ| < 1; (ξ+z−ρ) > 0
    whenever χ is real (which is guaranteed by construction).
    """
    atm = np.abs(x) < ATM_THRESHOLD

    safe_chi2   = np.where(atm, 1.0, chi**2)
    safe_xi     = np.where(atm, 1.0, xi)
    safe_xizrho = np.where(atm, 1.0, xi + z - rho)  # ξ + z − ρ

    numerator   = nu * x * ((rho - 1.0) * (z + xi) + (xi + z - rho) * xi)
    denominator = (rho - 1.0) * safe_xizrho * safe_xi * safe_chi2

    non_atm = numerator / denominator
    return np.where(atm, 0.0, non_atm)


def _dI0B_dnu(
    x: np.ndarray,
    z: np.ndarray,
    xi: np.ndarray,
    chi: np.ndarray,
    alpha: float,
    nu: float,
) -> np.ndarray:
    """
    ∂I⁰_B/∂ν for β=1.  (A.5.3 eq.(67))

    Derivation: differentiating I⁰_B = ν·x/χ through z = ν·x/α gives
        ∂χ/∂ν = x/(α·ξ),  hence:
        ∂(I⁰_B)/∂ν = x/χ − ν·x²/(α·ξ·χ²) = x·(α·ξ·χ − ν·x) / (α·ξ·χ²)

    Non-ATM:  x · (α·ξ·χ − ν·x) / (α·ξ·χ²)
    ATM:      0   (A.5.1 eq.(55))
    """
    atm = np.abs(x) < ATM_THRESHOLD

    safe_chi  = np.where(atm, 1.0, chi)
    safe_chi2 = np.where(atm, 1.0, chi**2)
    safe_xi   = np.where(atm, 1.0, xi)

    non_atm = x * (alpha * safe_xi * safe_chi - nu * x) / (alpha * safe_xi * safe_chi2)
    return np.where(atm, 0.0, non_atm)


def _dI0B_dF(
    x: np.ndarray,
    z: np.ndarray,
    xi: np.ndarray,
    chi: np.ndarray,
    alpha: float,
    nu: float,
    F: float,
) -> np.ndarray:
    """
    ∂I⁰_B/∂f for β=1.  (A.5.3 eq.(69))

    Derivation: ∂x/∂F = 1/F → ∂χ/∂F = ν/(α·F·ξ), hence:
        ∂(I⁰_B)/∂F = ν/(F·χ) − ν²·x/(α·F·ξ·χ²)
                    = ν·(α·ξ·χ − ν·x) / (α·F·ξ·χ²)

    Non-ATM:  ν · (α·ξ·χ − ν·x) / (α·F·ξ·χ²)
    ATM:      0   (A.5.1 eq.(56))
    """
    atm = np.abs(x) < ATM_THRESHOLD

    safe_chi  = np.where(atm, 1.0, chi)
    safe_chi2 = np.where(atm, 1.0, chi**2)
    safe_xi   = np.where(atm, 1.0, xi)

    non_atm = nu * (alpha * safe_xi * safe_chi - nu * x) / (alpha * F * safe_xi * safe_chi2)
    return np.where(atm, 0.0, non_atm)


# ══════════════════════════════════════════════════════════════════
# Group 3 — Full derivatives of σ_imp  (for calibration)
# ══════════════════════════════════════════════════════════════════
#
# Product rule from A.6 eq.(83):
#   ∂σ/∂p = (∂I⁰_B/∂p)·(1 + τ·I¹_H) + I⁰_B · τ · (∂I¹_H/∂p)
#
# β=1 derivatives of I¹_H  (A.4 specialised to β=1):
#   ∂I¹_H/∂α = ρν/4
#   ∂I¹_H/∂ρ = αν/4 − ρν²/4
#   ∂I¹_H/∂ν = αρ/4 + ν(2−3ρ²)/12


def sabr_vol_dalpha(
    K: np.ndarray | float,
    F: float,
    tau: float,
    alpha: float,
    rho: float,
    nu: float,
) -> np.ndarray | float:
    """
    ∂σ_imp/∂α — used in Bartlett delta (Σ_α) and calibration gradient.
    From A.6 eq.(84) and A.10 eq.(113).

        Σ_α = (1 + τ·I¹_H) · ∂I⁰_B/∂α  +  τ · I⁰_B · (ρν/4)

    Vectorized over K.
    """
    K = np.asarray(K, dtype=float)
    x, z, xi, chi, I1H = _compute_auxiliary(K, F, alpha, rho, nu)

    atm = np.abs(x) < ATM_THRESHOLD

    # I⁰_B
    safe_chi = np.where(atm, 1.0, chi)
    I0B = np.where(atm, alpha, alpha * z / safe_chi)

    d_I0B      = _dI0B_dalpha(x, z, xi, chi, alpha, nu)
    dI1H_dalph = rho * nu / 4.0                          # ∂I¹_H/∂α

    result = (1.0 + tau * I1H) * d_I0B + tau * I0B * dI1H_dalph
    return float(result) if result.ndim == 0 else result


def sabr_vol_drho(
    K: np.ndarray | float,
    F: float,
    tau: float,
    alpha: float,
    rho: float,
    nu: float,
) -> np.ndarray | float:
    """
    ∂σ_imp/∂ρ — used in calibration gradient.
    From A.6 eq.(87).

        ∂σ/∂ρ = (1 + τ·I¹_H) · ∂I⁰_B/∂ρ  +  τ · I⁰_B · (αν/4 − ρν²/4)

    Vectorized over K.
    """
    K = np.asarray(K, dtype=float)
    x, z, xi, chi, I1H = _compute_auxiliary(K, F, alpha, rho, nu)

    atm = np.abs(x) < ATM_THRESHOLD
    safe_chi = np.where(atm, 1.0, chi)
    I0B = np.where(atm, alpha, alpha * z / safe_chi)

    d_I0B     = _dI0B_drho(x, z, xi, chi, rho, nu)
    dI1H_drho = alpha * nu / 4.0 - rho * nu**2 / 4.0    # ∂I¹_H/∂ρ

    result = (1.0 + tau * I1H) * d_I0B + tau * I0B * dI1H_drho
    return float(result) if result.ndim == 0 else result


def sabr_vol_dnu(
    K: np.ndarray | float,
    F: float,
    tau: float,
    alpha: float,
    rho: float,
    nu: float,
) -> np.ndarray | float:
    """
    ∂σ_imp/∂ν — used in calibration gradient.
    From A.6 eq.(86).

        ∂σ/∂ν = (1 + τ·I¹_H) · ∂I⁰_B/∂ν  +  τ · I⁰_B · (αρ/4 + ν(2−3ρ²)/12)

    Vectorized over K.
    """
    K = np.asarray(K, dtype=float)
    x, z, xi, chi, I1H = _compute_auxiliary(K, F, alpha, rho, nu)

    atm = np.abs(x) < ATM_THRESHOLD
    safe_chi = np.where(atm, 1.0, chi)
    I0B = np.where(atm, alpha, alpha * z / safe_chi)

    d_I0B    = _dI0B_dnu(x, z, xi, chi, alpha, nu)
    dI1H_dnu = alpha * rho / 4.0 + nu * (2.0 - 3.0 * rho**2) / 12.0  # ∂I¹_H/∂ν

    result = (1.0 + tau * I1H) * d_I0B + tau * I0B * dI1H_dnu
    return float(result) if result.ndim == 0 else result


# ══════════════════════════════════════════════════════════════════
# Group 4 — Derivative of σ_imp w.r.t. F  (for SABR delta)
# ══════════════════════════════════════════════════════════════════

def sabr_vol_dF(
    K: np.ndarray | float,
    F: float,
    tau: float,
    alpha: float,
    rho: float,
    nu: float,
) -> np.ndarray | float:
    """
    Σ_f = ∂σ_imp/∂F — used in SABR delta and Bartlett delta.
    From A.10 eq.(112).

        Σ_f = (1 + τ·I¹_H) · ∂I⁰_B/∂f

    Note: ∂I¹_H/∂f = 0 for β=1 (A.10 simplification — I¹_H is f-independent).
    Vectorized over K.
    """
    K = np.asarray(K, dtype=float)
    x, z, xi, chi, I1H = _compute_auxiliary(K, F, alpha, rho, nu)

    d_I0B = _dI0B_dF(x, z, xi, chi, alpha, nu, F)

    result = (1.0 + tau * I1H) * d_I0B
    return float(result) if result.ndim == 0 else result


# ══════════════════════════════════════════════════════════════════
# Group 5 — Black formula and Greeks
# ══════════════════════════════════════════════════════════════════

def black_price(
    F: float,
    K: np.ndarray | float,
    tau: float,
    sigma: np.ndarray | float,
    r: float,
) -> np.ndarray | float:
    """
    Black call price.  (A.7 eq.(93))

        B  = e^(−rτ) · [F·Φ(d₁) − K·Φ(d₂)]
        d₁ = [log(F/K) + ½σ²τ] / (σ√τ)
        d₂ = d₁ − σ√τ
    """
    K     = np.asarray(K,     dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau

    price = np.exp(-r * tau) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return float(price) if price.ndim == 0 else price


def black_delta_F(
    F: float,
    K: np.ndarray | float,
    tau: float,
    sigma: np.ndarray | float,
    r: float,
) -> np.ndarray | float:
    """
    Forward-based Black delta ∂B/∂F.  (A.7 eq.(95))

        Δ_B = e^(−rτ) · Φ(d₁)
    """
    K     = np.asarray(K,     dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * tau) / (sigma * sqrt_tau)

    delta = np.exp(-r * tau) * norm.cdf(d1)
    return float(delta) if delta.ndim == 0 else delta


def black_vega(
    F: float,
    K: np.ndarray | float,
    tau: float,
    sigma: np.ndarray | float,
    r: float,
) -> np.ndarray | float:
    """
    Black vega ∂B/∂σ.  (A.7 eq.(96))

        V_B = e^(−rτ) · F · √τ · φ(d₁)
    """
    K     = np.asarray(K,     dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * tau) / (sigma * sqrt_tau)

    vega = np.exp(-r * tau) * F * sqrt_tau * norm.pdf(d1)
    return float(vega) if vega.ndim == 0 else vega


# ══════════════════════════════════════════════════════════════════
# Group 6 — SABR delta and Bartlett delta (spot-based)
# ══════════════════════════════════════════════════════════════════

def delta_sabr(
    K: np.ndarray | float,
    F: float,
    tau: float,
    alpha: float,
    rho: float,
    nu: float,
    r: float,
    q: float,
) -> np.ndarray | float:
    """
    Spot-based SABR delta.  (A.8 eq.(100) and A.10)

    Steps
    -----
    1. σ       = sabr_vol(K, F, τ, α, ρ, ν)
    2. Σ_f     = sabr_vol_dF(K, F, τ, α, ρ, ν)
    3. Δ_B     = black_delta_F(F, K, τ, σ, r)
    4. V_B     = black_vega(F, K, τ, σ, r)
    5. Δ_SABR_F = Δ_B + V_B · Σ_f          (chain rule: dσ/dF term)
    6. Δ_SABR   = e^{(r−q)τ} · Δ_SABR_F    (forward → spot conversion)

    Returns spot delta w.r.t. S_t.
    """
    sigma   = sabr_vol(K, F, tau, alpha, rho, nu)
    Sigma_f = sabr_vol_dF(K, F, tau, alpha, rho, nu)
    Delta_B = black_delta_F(F, K, tau, sigma, r)
    V_B     = black_vega(F, K, tau, sigma, r)

    Delta_SABR_F = Delta_B + V_B * Sigma_f
    Delta_SABR   = np.asarray(np.exp((r - q) * tau) * Delta_SABR_F)

    return float(Delta_SABR) if Delta_SABR.ndim == 0 else Delta_SABR


def delta_bartlett(
    K: np.ndarray | float,
    F: float,
    tau: float,
    alpha: float,
    rho: float,
    nu: float,
    r: float,
    q: float,
) -> np.ndarray | float:
    """
    Spot-based Bartlett delta.  (A.8 eq.(103) and A.10)

    Steps 1–4 identical to delta_sabr, then:

    5. Σ_α       = sabr_vol_dalpha(K, F, τ, α, ρ, ν)
    6. Δ_Bart_F  = Δ_B + V_B · (Σ_f + Σ_α · ρν/F)
                         ↑ Bartlett correction: captures α–F covariation
    7. Δ_Bart    = e^{(r−q)τ} · Δ_Bart_F

    The Bartlett correction term  V_B · Σ_α · ρν/F  accounts for the
    negative correlation between α (stochastic vol) and F.  For equity
    (ρ < 0) this lowers the delta relative to the plain SABR delta.

    Returns spot delta w.r.t. S_t.
    """
    sigma   = sabr_vol(K, F, tau, alpha, rho, nu)
    Sigma_f = sabr_vol_dF(K, F, tau, alpha, rho, nu)
    Delta_B = black_delta_F(F, K, tau, sigma, r)
    V_B     = black_vega(F, K, tau, sigma, r)

    Sigma_alpha  = sabr_vol_dalpha(K, F, tau, alpha, rho, nu)
    bartlett_adj = Sigma_alpha * rho * nu / F   # covariance correction term

    Delta_Bart_F = Delta_B + V_B * (Sigma_f + bartlett_adj)
    Delta_Bart   = np.asarray(np.exp((r - q) * tau) * Delta_Bart_F)

    return float(Delta_Bart) if Delta_Bart.ndim == 0 else Delta_Bart


# ══════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test parameters
    F, K_atm, K_otm = 4000.0, 4000.0, 4200.0
    tau = 30 / 365
    alpha, rho, nu = 0.20, -0.40, 0.40
    r, q = 0.045, 0.015

    # 1. Test ATM vol
    vol_atm = sabr_vol(K_atm, F, tau, alpha, rho, nu)
    print(f"ATM vol: {vol_atm:.6f}  (expected ≈ alpha*(1+tau*I1H) ≈ 0.20x)")

    # 2. Test smile is skewed (OTM call should have lower vol than ATM for rho<0)
    vol_otm = sabr_vol(K_otm, F, tau, alpha, rho, nu)
    print(f"OTM vol: {vol_otm:.6f}  (expected < ATM for rho<0)")

    # 3. Test Sigma_f
    sf = sabr_vol_dF(K_otm, F, tau, alpha, rho, nu)
    print(f"Sigma_f: {sf:.6f}  (expected sign depends on moneyness)")

    # 4. Test delta_sabr is between 0 and 1
    ds = delta_sabr(K_otm, F, tau, alpha, rho, nu, r, q)
    print(f"SABR delta OTM: {ds:.6f}  (expected in (0,1))")

    # 5. Test Bartlett delta differs from SABR delta
    db = delta_bartlett(K_otm, F, tau, alpha, rho, nu, r, q)
    print(f"Bartlett delta OTM: {db:.6f}")
    print(f"Difference Bart-SABR: {db - ds:.6f}  (expected nonzero for rho != 0)")

    # 6. Validate dalpha numerically
    h = 1e-4 * alpha
    vol_up = sabr_vol(K_otm, F, tau, alpha + h, rho, nu)
    vol_dn = sabr_vol(K_otm, F, tau, alpha - h, rho, nu)
    dalpha_numerical   = (vol_up - vol_dn) / (2 * h)
    dalpha_analytical  = sabr_vol_dalpha(K_otm, F, tau, alpha, rho, nu)
    print(f"\nValidation ∂σ/∂α:")
    print(f"  Numerical:  {dalpha_numerical:.8f}")
    print(f"  Analytical: {dalpha_analytical:.8f}")
    print(f"  Error:      {abs(dalpha_numerical - dalpha_analytical):.2e}")

    # 7. Validate dF numerically
    h = 1e-4 * F
    vol_up = sabr_vol(K_otm, F + h, tau, alpha, rho, nu)
    vol_dn = sabr_vol(K_otm, F - h, tau, alpha, rho, nu)
    dF_numerical  = (vol_up - vol_dn) / (2 * h)
    dF_analytical = sabr_vol_dF(K_otm, F, tau, alpha, rho, nu)
    print(f"\nValidation ∂σ/∂F:")
    print(f"  Numerical:  {dF_numerical:.8f}")
    print(f"  Analytical: {dF_analytical:.8f}")
    print(f"  Error:      {abs(dF_numerical - dF_analytical):.2e}")

    # 8. Extra: validate drho and dnu numerically
    h = 1e-5
    drho_numerical  = (sabr_vol(K_otm, F, tau, alpha, rho + h, nu) -
                       sabr_vol(K_otm, F, tau, alpha, rho - h, nu)) / (2 * h)
    drho_analytical = sabr_vol_drho(K_otm, F, tau, alpha, rho, nu)
    print(f"\nValidation ∂σ/∂ρ:")
    print(f"  Numerical:  {drho_numerical:.8f}")
    print(f"  Analytical: {drho_analytical:.8f}")
    print(f"  Error:      {abs(drho_numerical - drho_analytical):.2e}")

    dnu_numerical  = (sabr_vol(K_otm, F, tau, alpha, rho, nu + h) -
                      sabr_vol(K_otm, F, tau, alpha, rho, nu - h)) / (2 * h)
    dnu_analytical = sabr_vol_dnu(K_otm, F, tau, alpha, rho, nu)
    print(f"\nValidation ∂σ/∂ν:")
    print(f"  Numerical:  {dnu_numerical:.8f}")
    print(f"  Analytical: {dnu_analytical:.8f}")
    print(f"  Error:      {abs(dnu_numerical - dnu_analytical):.2e}")

    # 9. Array test
    strikes = np.array([3600.0, 3800.0, 4000.0, 4200.0, 4400.0])
    vols    = sabr_vol(strikes, F, tau, alpha, rho, nu)
    print(f"\nSmile across strikes: {strikes}")
    print(f"Vols:                 {np.round(vols, 6)}")
    assert vols[2] >= vols[3], "Smile should be downward-sloping for rho<0"
    print("All checks passed.")
