import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import zeta


def mc_binary(S0, K, r, sigma, T, N=10000):
    payoff = sum(
        1
        for _ in range(N)
        if S0
        * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.normal())
        >= K
    )
    return np.exp(-r * T) * payoff / N


# Implicit Scheme


def implicit_binary_solve(S, K, dS, dt, sigma, r, M, N):
    C = np.where(S > K, 1.0, 0.0)

    for _ in range(N):
        A = np.zeros((M + 1, M + 1))
        RHS = C.copy()

        A[0, 0] = 1.0
        A[M, M] = 1.0
        RHS[0] = 0.0
        RHS[M] = 1.0

        for i in range(1, M):
            Si = S[i]
            diff = sigma**2 * Si**2 / dS**2
            drift = r * Si / (2 * dS)

            A[i, i - 1] = -dt * (0.5 * diff - 0.5 * drift)
            A[i, i] = 1 + dt * (diff + r)
            A[i, i + 1] = -dt * (0.5 * diff + 0.5 * drift)

        C = np.linalg.solve(A, RHS)

    return C


def implicit_binary_solve_store(S, K, dS, dt, sigma, r, M, N):
    C = np.where(S > K, 1.0, 0.0)
    C_store = np.zeros((N, M + 1))

    for n in range(N):
        A = np.zeros((M + 1, M + 1))
        RHS = C.copy()
        A[0, 0] = 1.0
        A[M, M] = 1.0
        RHS[0] = 0.0
        RHS[M] = 1.0

        for i in range(1, M):
            Si = S[i]
            diff = sigma**2 * Si**2 / dS**2
            drift = r * Si / (2 * dS)
            A[i, i - 1] = -dt * (0.5 * diff - 0.5 * drift)
            A[i, i] = 1 + dt * (diff + r)
            A[i, i + 1] = -dt * (0.5 * diff + 0.5 * drift)

        C = np.linalg.solve(A, RHS)
        C_store[n, :] = C

    return C, C_store


# Crank Nicolson Scheme


def build_cn_matrices(S, dS, dt, sigma, r, M):
    A = np.zeros((M + 1, M + 1))
    B = np.zeros((M + 1, M + 1))
    A[0, 0] = A[M, M] = B[0, 0] = B[M, M] = 1.0
    for i in range(1, M):
        Si = S[i]
        diff = sigma**2 * Si**2 / dS**2
        drift = r * Si / (2 * dS)
        A[i, i - 1] = -0.25 * dt * (diff - drift)
        A[i, i] = 1 + 0.5 * dt * (diff + r)
        A[i, i + 1] = -0.25 * dt * (diff + drift)
        B[i, i - 1] = 0.25 * dt * (diff - drift)
        B[i, i] = 1 - 0.5 * dt * (diff + r)
        B[i, i + 1] = 0.25 * dt * (diff + drift)
    return A, B


def cn_solve(S, K, A, B_mat, M, N):
    C = np.where(S > K, 1.0, 0.0)
    for _ in range(N):
        RHS = B_mat @ C
        RHS[0], RHS[M] = 0.0, 1.0
        C = np.linalg.solve(A, RHS)
    return C


def cn_solve_store(S, K, A, B_mat, M, N):
    C = np.where(S > K, 1.0, 0.0)
    C_store = np.zeros((N, M + 1))
    for n in range(N):
        RHS = B_mat @ C
        RHS[0], RHS[M] = 0.0, 1.0
        C = np.linalg.solve(A, RHS)
        C_store[n, :] = C
    return C, C_store


def cn_barrier_solve(S, K, B, A, B_mat, M, N):
    # terminal payoff
    C = np.maximum(S - K, 0.0)
    C[S >= B] = 0.0
    C_store = np.zeros((N, M + 1))
    for n in range(N):
        RHS = B_mat @ C

        RHS[0] = 0.0
        RHS[M] = 0.0

        C = np.linalg.solve(A, RHS)
        C[S >= B] = 0.0

        C_store[n, :] = C

    return C, C_store


def compute_delta(C, dS, M):
    delta = np.zeros(M + 1)
    for i in range(1, M):
        delta[i] = (C[i + 1] - C[i - 1]) / (2 * dS)
    delta[0], delta[M] = delta[1], delta[M - 1]
    return delta


def d_pm(z, tau, sign, r, sigma):
    return (np.log(z) + (r + sign * 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))


def analytical_ko_price(S, K, B, tau, r, sigma):
    Phi = norm.cdf
    d = lambda z, s: d_pm(z, tau, s, r, sigma)
    t1 = S * (Phi(d(S / K, +1)) - Phi(d(S / B, +1)))
    t2 = (
        -S
        * (B / S) ** (1 + 2 * r / sigma**2)
        * (Phi(d(B**2 / (K * S), +1)) - Phi(d(B / S, +1)))
    )
    t3 = -np.exp(-r * tau) * K * (Phi(d(S / K, -1)) - Phi(d(S / B, -1)))
    t4 = (
        np.exp(-r * tau)
        * K
        * (S / B) ** (1 - 2 * r / sigma**2)
        * (Phi(d(B**2 / (K * S), -1)) - Phi(d(B / S, -1)))
    )
    return t1 + t2 + t3 + t4


def mc_ko_adjusted(S0, K, B, r, sigma, T, m, beta1, N_paths=100000):
    B_adj = B * np.exp(beta1 * sigma * np.sqrt(T / m))
    dt = T / m
    S = np.full(N_paths, float(S0))
    max_S = S.copy()
    Z = np.random.randn(N_paths, m)
    for j in range(m):
        S = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, j])
        max_S = np.maximum(max_S, S)
    payoff = np.maximum(S - K, 0) * (max_S < B_adj)
    return np.exp(-r * T) * np.mean(payoff)


if __name__ == "__main__":

    # Part 1: Binary Option
    # Monte Carlo

    price_mc = mc_binary(S0=1.0, K=1.2, r=0.05, sigma=0.2, T=1.0)
    print(f"MC binary option price: {price_mc:.4f}")

    S_max, S_min = 200, 0
    K_base = 100
    T, sigma, r = 1.0, 0.2, 0.05
    M, N = 200, 200
    dS = (S_max - S_min) / M
    dt = T / N
    S = np.linspace(S_min, S_max, M + 1)

    # Implicit scheme price
    C_impl = implicit_binary_solve(S, K_base, dS, dt, sigma, r, M, N)
    print(f"Implicit scheme binary price at S=100: {np.interp(100, S, C_impl):.4f}")

    # Crank Nicolson price
    A_cn, B_cn = build_cn_matrices(S, dS, dt, sigma, r, M)
    C_cn = cn_solve(S, K_base, A_cn, B_cn, M, N)
    print(f"Crank-Nicolson binary price at S=100:  {np.interp(100, S, C_cn):.4f}")

    A, B_mat = build_cn_matrices(S, dS, dt, sigma, r, M)

    # sensitivity to strike K
    K_values = [80, 90, 100, 110, 120]
    for K in K_values:
        C = cn_solve(S, K, A, B_mat, M, N)
        plt.plot(S, C, label=f"K={K}")
    plt.title("Sensitivity to Strike K")
    plt.xlabel("S")
    plt.ylabel("Option Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/part1/binary_sensitivity_K.png", dpi=150)
    plt.show()

    # price surface over (S, t)
    C, C_store = cn_solve_store(S, K_values[1], A, B_mat, M, N)
    S_grid, T_grid = np.meshgrid(S, np.linspace(0, T, N))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(S_grid, T_grid, C_store)
    ax.set_xlabel("S")
    ax.set_ylabel("t")
    ax.set_zlabel("C(S,t)")
    plt.title("Binary Option Surface")
    plt.savefig("outputs/part1/binary_surface.png", dpi=150)
    plt.show()

    # delta
    delta = compute_delta(C, dS, M)
    plt.plot(S, delta)
    plt.title("Delta of Digital Option")
    plt.xlabel("S")
    plt.ylabel("Delta")
    plt.tight_layout()
    plt.savefig("outputs/part1/binary_delta.png", dpi=150)
    plt.show()

    # Part 2: Knock-out option
    S0, K, B, r, sigma, T = 100, 100, 120, 0.05, 0.2, 1.0
    beta1 = -zeta(0.5) / np.sqrt(2 * np.pi)

    C_exact = analytical_ko_price(S0, K, B, T, r, sigma)
    print(f"\nAnalytical KO price: {C_exact:.4f}")

    # Monte Carlo error - convergence
    print(f"\n{'m':>6} {'MC adjusted':>14} {'Error':>10}")
    for m in [10, 20, 50, 100, 250, 500]:
        C_mc = mc_ko_adjusted(S0, K, B, r, sigma, T, m, beta1)
        print(f"{m:>6} {C_mc:>14.4f} {abs(C_mc - C_exact):>10.4f}")

    # sensitivity — price vs S, sigma, B (Analytical formula)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    S_range = np.linspace(80, B - 0.01, 200)
    plt.plot(S_range, [analytical_ko_price(s, K, B, T, r, sigma) for s in S_range])
    plt.xlabel("S")
    plt.ylabel("C")
    plt.title("Price vs S")

    plt.subplot(1, 3, 2)
    sig_range = np.linspace(0.05, 0.5, 100)
    plt.plot(sig_range, [analytical_ko_price(S0, K, B, T, r, s) for s in sig_range])
    plt.xlabel("σ")
    plt.ylabel("C")
    plt.title("Price vs σ")

    plt.subplot(1, 3, 3)
    B_range = np.linspace(K + 1, 150, 100)
    plt.plot(B_range, [analytical_ko_price(S0, K, b, T, r, sigma) for b in B_range])
    plt.xlabel("B")
    plt.ylabel("C")
    plt.title("Price vs Barrier B")

    plt.tight_layout()
    plt.savefig("outputs/part1/barrier_sensitivity_analytical.png", dpi=150)
    plt.show()

    S0, K, B, r, sigma, T = 100, 100, 120, 0.05, 0.2, 1.0

    S_min, S_max = 0, B
    M, N = 200, 200
    dS, dt = (S_max - S_min) / M, T / N
    S = np.linspace(S_min, S_max, M + 1)

    A, B_mat = build_cn_matrices(S, dS, dt, sigma, r, M)

    # Solve
    C, C_store = cn_barrier_solve(S, K, B, A, B_mat, M, N)

    S_grid, T_grid = np.meshgrid(S, np.linspace(0, T, N))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(S_grid, T_grid, C_store)
    ax.set_xlabel("S")
    ax.set_ylabel("t")
    ax.set_zlabel("C(S,t)")
    plt.title("Barrier Option Surface")
    plt.savefig("outputs/part1/barrier_surface.png", dpi=150)
    plt.show()

    # Sensitivity - price vs S, sigma, B (Crank Nicolson)

    plt.figure(figsize=(12, 4))

    # 1. Price vs S
    plt.subplot(1, 3, 1)
    plt.plot(S, C)
    plt.title("Price vs S")
    plt.xlabel("S")
    plt.ylabel("C")

    # 2. Price vs sigma
    plt.subplot(1, 3, 2)
    sig_range = np.linspace(0.1, 0.5, 20)
    prices_sigma = []

    for sig in sig_range:
        A, B_mat = build_cn_matrices(S, dS, dt, sig, r, M)
        C_tmp, _ = cn_barrier_solve(S, K, B, A, B_mat, M, N)
        prices_sigma.append(np.interp(S0, S, C_tmp))

    plt.plot(sig_range, prices_sigma)
    plt.title("Price vs σ")
    plt.xlabel("σ")

    # 3. Price vs Barrier B
    plt.subplot(1, 3, 3)
    B_range = np.linspace(105, 150, 20)
    prices_B = []

    for b in B_range:
        S_tmp = np.linspace(0, b, M + 1)
        dS_tmp = (b - 0) / M
        A_tmp, B_tmp = build_cn_matrices(S_tmp, dS_tmp, dt, sigma, r, M)
        C_tmp, _ = cn_barrier_solve(S_tmp, K, b, A_tmp, B_tmp, M, N)
        prices_B.append(np.interp(S0, S_tmp, C_tmp))

    plt.plot(B_range, prices_B)
    plt.title("Price vs Barrier B")
    plt.xlabel("B")
    plt.tight_layout()
    plt.savefig("outputs/part1/barrier_price_sensitivity_cn.png", dpi=150)
    plt.show()

    # Delta
    delta_barrier = compute_delta(C, dS, M)

    plt.plot(S, delta_barrier)
    plt.title("Delta of Barrier Option")
    plt.xlabel("S")
    plt.ylabel("Delta")
    plt.tight_layout()
    plt.savefig("outputs/part1/delta_barrier.png", dpi=150)
    plt.show()
