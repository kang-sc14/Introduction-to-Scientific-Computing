import numpy as np
from scipy.fftpack import dct   
import matplotlib.pyplot as plt
import time

# ========================================
# Chebyshev nodes & transforms
# ========================================

def chebyshev_nodes(N):
    """Chebyshev extrema nodes, x_k = cos(pi*k/N)"""
    return np.cos(np.pi * np.arange(N + 1) / N)


def cheb_coeff(values):
    """
    Compute Chebyshev coefficients from function values at nodes (length N+1).
    Uses DCT-I and halves the endpoints coefficients.
    """
    N = len(values) - 1
    a = dct(values, type=1) / N     # DCT-I
    a[0] /= 2
    a[-1] /= 2
    return a


def cheb_val(a, x):
    """
    Vectorized Clenshaw recursion.
    Assumes a[0] is already halved (consistent with cheb_coeff definition)
    """
    x_arr = np.array(x, copy=False)
    scalar_input = False
    if x_arr.ndim == 0:
        x_arr = x_arr[np.newaxis]
        scalar_input = True

    N = len(a) - 1
    bkp2 = np.zeros_like(x_arr)
    bkp1 = np.zeros_like(x_arr)

    for k in range(N, 0, -1):
        bk = 2 * x_arr * bkp1 - bkp2 + a[k]
        bkp2, bkp1 = bkp1, bk

    result = a[0] + bkp1 * x_arr - bkp2

    if scalar_input:
        return float(result[0])
    return result


# ========================================
# Spectral Integration
# ========================================

def spectral_integration(a):
    """
    Spectral integration (Formulas 11-12)
    Length of a = N+1
    d_k = (a_{k-1} - a_{k+1})/(2k),  k>=1
    d_0 = sum((-1)^(k+1) * d_k)
    """
    M = len(a)
    N = M - 1
    d = np.zeros_like(a)

    # k = 1..N
    for k in range(1, N + 1):
        a_prev = a[k - 1]
        a_next = a[k + 1] if (k + 1 < M) else 0.0
        d[k] = (a_prev - a_next) / (2 * k)

    # k = 0
    d[0] = sum(((-1) ** (k + 1)) * d[k] for k in range(1, N + 1))

    return d


# ========================================
# Solve Poisson BVP: u'' = f, u(-1)=u(1)=0
# ========================================

def solve_bvp_poisson(N, f_func, exact_func):
    start = time.time()

    x = chebyshev_nodes(N)
    f = f_func(x)

    f_coeff = cheb_coeff(f)
    u1_coeff = spectral_integration(f_coeff)
    u_coeff  = spectral_integration(u1_coeff)

    # Values without constants
    u_minus1 = cheb_val(u_coeff, -1)
    u_plus1  = cheb_val(u_coeff, 1)

    # Adjust constants
    C0 = -(u_plus1 + u_minus1) / 2
    C1 = -(u_plus1 - u_minus1) / 2

    u_final = cheb_val(u_coeff, x) + (C0 + C1 * x)

    exact = exact_func(x)
    mse = np.mean((u_final - exact)**2)
    cpu = time.time() - start

    return mse, cpu, u_final, x


# ========================================
# Test problem: u'' = sin(pi x)
# exact solution: u = -sin(pi x)/pi^2
# ========================================

def f_func(x):
    return np.sin(np.pi * x)

def exact_func(x):
    return -np.sin(np.pi * x) / (np.pi**2)


# ========================================
# TABLE 1
# ========================================

Ns = [4, 8, 16, 32, 64]

print("="*70)
print("TABLE 1")
print("CPU times (in seconds) and mean square errors")
print("Problem: u''(x) = sin(pi x),   u(-1)=u(1)=0")
print("="*70)
print(f"{'N':<10} {'CPU Time (s)':<22} {'Mean Square Error':<30}")
print("-"*70)

results = []
for N in Ns:
    mse, cpu, _, _ = solve_bvp_poisson(N, f_func, exact_func)
    results.append((N, mse, cpu))
    print(f"{N:<10} {cpu:<22.6f} {mse:<30.3e}")

print("="*70)


# ========================================
# Convergence plot (paper style)
# ========================================

N_plot = [r[0] for r in results]
mse_plot = [r[1] for r in results]

plt.figure(figsize=(8, 6))

plt.plot(N_plot, mse_plot, 'o-', color='tab:blue', linewidth=2, markersize=8, label='Spectral Integration MSE')

plt.xscale('log')
plt.yscale('log')

plt.xlabel("Number of Chebyshev nodes (N)", fontsize=12)
plt.ylabel("Mean Square Error (MSE)", fontsize=12)
plt.title("Spectral Integration Convergence", fontsize=14)

plt.grid(which='major', linestyle='-', linewidth=0.8, alpha=0.7)
plt.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.5)

plt.xticks(N_plot, labels=[str(n) for n in N_plot], fontsize=10)
plt.yticks(fontsize=10)

plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig("C:\\Users\\asus\\Downloads\\convergence.pdf", dpi=300)
plt.show()
