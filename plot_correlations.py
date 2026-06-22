import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

csv_path = Path(__file__).parent / "correlations.csv"
df = pd.read_csv(csv_path)

t = df["tJ2"].to_numpy()
gxx = df["gxx"].to_numpy()
gxy = df["gxy"].to_numpy()
gzz = df["gzz"].to_numpy()

def gaussian(t, sigma):
    return 0.25 * np.exp(-t**2 / (2 * sigma**2))

(sigma,), _ = curve_fit(gaussian, t, gxx, p0=[1.5])
print(f"Fitted σ = {sigma:.4f} (paper: 1.46/J₂)")

t_fit = np.linspace(0, t.max(), 500)

fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(t, 4 * gzz, "-", label=r"$g^{zz}$")
ax.plot(t, 4 * gxx, "-", label=r"$g^{xx}$")
ax.plot(t, 4 * gxy, "-", label=r"$g^{xy}$")
ax.plot(t_fit, 4 * gaussian(t_fit, sigma), "k--", label=rf"Gaussian $\sigma={sigma:.2f}$")

ax.set_xlabel(r"$t$ (units of $\frac{1}{\mathcal{J}_2}$)", fontsize=12)
ax.set_ylabel(r"$4g^{\alpha\beta}$", fontsize=12)
ax.set_xlim(0, t.max())
ax.axhline(0, color="k", linewidth=0.5)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = Path(__file__).parent / "correlations.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")
plt.show()
