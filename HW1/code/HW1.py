import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, BarycentricInterpolator, KroghInterpolator

# 定義 Runge function
def f(x):
    return 1 / (1 + x**2)

# 區間
x_interval = [-5, 5]
x_plot = np.linspace(x_interval[0], x_interval[1], 300)
y_original = f(x_plot)

# 要比較的節點數
n_list = [10, 100]

# 建立 subplot
fig, axes = plt.subplots(len(n_list), 1, figsize=(12, 8))

for idx, n in enumerate(n_list):

    # 等距節點
    x_nodes = np.linspace(x_interval[0], x_interval[1], n)
    y_nodes = f(x_nodes)

    # 1. Lagrange
    poly_lagrange = lagrange(x_nodes, y_nodes)
    y_lagrange = poly_lagrange(x_plot)

    # 2. Barycentric
    interp_bary = BarycentricInterpolator(x_nodes, y_nodes)
    y_bary = interp_bary(x_plot)

    # 3. Newton (KroghInterpolator)
    interp_newton = KroghInterpolator(x_nodes, y_nodes)
    y_newton = interp_newton(x_plot)

    # ===== 繪圖 =====
    ax = axes[idx]
    ax.plot(x_plot, y_original, 'k-', linewidth=2, label='Original f(x)')
    ax.plot(x_plot, y_lagrange, '--', label='Lagrange', color='green')
    ax.plot(x_plot, y_bary, '--', label='Barycentric', color='red')
    ax.plot(x_plot, y_newton, '--', label='Newton', color='gray')
    ax.plot(x_nodes, y_nodes, 'o', label='Nodes', color='blue', markersize=4)
    ax.set_title(f'Comparison Polynomial Interpolation (n={n} points)')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_ylim(-2, 3)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()


