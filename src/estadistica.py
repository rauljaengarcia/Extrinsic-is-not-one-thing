"""
Estadistica del experimento: bootstrap, los dos indices, y el guardian.

Decisiones fijadas ANTES de correr nada (pre-registro, seccion 7 del README):

  * el resultado primario son los contrastes CRUDOS `Phi_B`, `Phi_R`, `Phi_S`
    con intervalos bootstrap; `delta` y `rho` son derivados;
  * `n_boot = 10_000`, percentil 2.5 / 97.5;
  * el guardian `epsilon` se fija en el 5% de `Phi_B` de cada manipulacion;
  * donde el guardian muerde, la salida es `NA`, NUNCA cero.  Asignar cero
    registraria una regla parcial como exchangeable, que es exactamente el
    error que la desigualdad `parcialidad != dependencia` existe para impedir.

El bootstrap se siembra de forma explicita para que el analisis sea tan
reproducible como la simulacion.
"""
from __future__ import annotations

import numpy as np

N_BOOT = 10_000
ALPHA = 0.05
BOOT_SEED = 20260911


def boot_mean(x, n_boot=N_BOOT, alpha=ALPHA, seed=BOOT_SEED):
    """Media con intervalo percentil bootstrap."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, (n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


def boot_stat(fn, *samples, n_boot=N_BOOT, alpha=ALPHA, seed=BOOT_SEED):
    """Intervalo bootstrap de un estadistico de varias muestras independientes.

    Cada muestra se remuestrea por separado, que es lo correcto cuando los
    brazos son runs independientes con semillas distintas.
    """
    rng = np.random.default_rng(seed)
    samples = [np.asarray(s, dtype=np.float64) for s in samples]
    samples = [s[np.isfinite(s)] for s in samples]
    if any(s.size == 0 for s in samples):
        return np.nan, np.nan, np.nan
    vals = np.empty(n_boot)
    for b in range(n_boot):
        draw = [s[rng.integers(0, s.size, s.size)].mean() for s in samples]
        vals[b] = fn(*draw)
    point = fn(*[s.mean() for s in samples])
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return float(point), np.nan, np.nan
    lo, hi = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def delta(phi_b, phi_r):
    """Contraste de dependencia, acotado y con signo.

        delta = (Phi_B - Phi_R) / max(Phi_B, Phi_R)   en [-1, 1]

    `delta < 0` significa que el soporte era una RESTRICCION y no un apoyo.
    Es una prediccion sustantiva y facil de falsar, no un caso degenerado.
    """
    m = max(phi_b, phi_r)
    if m <= 0:
        return np.nan
    return (phi_b - phi_r) / m


def rho(phi_b, phi_r, phi_s, eps):
    """Dependencia del suministro parcial.

        rho = (Phi_B - Phi_S) / (Phi_B - Phi_R)

    Es un estimador de razon, inestable cerca del cero de su denominador.
    Donde el guardian muerde devuelve NaN (que se informa como `NA`), nunca
    cero.
    """
    den = phi_b - phi_r
    if not np.isfinite(den) or den < eps:
        return np.nan
    return (phi_b - phi_s) / den


def resumen_brazos(phi_b, phi_r, phi_s, eps_frac=0.05, seed=BOOT_SEED):
    """Los tres brazos con intervalos, mas `delta` y `rho` propagados.

    Devuelve tambien si el guardian mordio y por que, para que el informe
    nunca presente un `NA` sin explicacion.
    """
    b = boot_mean(phi_b, seed=seed)
    r = boot_mean(phi_r, seed=seed + 1)
    s = boot_mean(phi_s, seed=seed + 2)
    eps = eps_frac * b[0] if np.isfinite(b[0]) else np.nan

    d = boot_stat(delta, phi_b, phi_r, seed=seed + 3)
    rr = boot_stat(lambda x, y, z: rho(x, y, z, eps), phi_b, phi_r, phi_s,
                   seed=seed + 4)

    den_point = b[0] - r[0]
    den_ci = boot_stat(lambda x, y: x - y, phi_b, phi_r, seed=seed + 5)
    guard = None
    if not np.isfinite(den_point) or den_point < eps:
        guard = f"Phi_B - Phi_R = {den_point:,.0f} < epsilon = {eps:,.0f}"
    elif den_ci[1] <= 0 <= den_ci[2]:
        guard = "el intervalo de Phi_B - Phi_R cubre 0"

    return {
        "Phi_B": b, "Phi_R": r, "Phi_S": s,
        "epsilon": eps, "den": den_ci,
        "delta": d, "rho": (np.nan, np.nan, np.nan) if guard else rr,
        "guardian": guard,
    }


def fisher_exact_2x2(a, b, c, d):
    """p bilateral de Fisher para una tabla 2x2, sin depender de scipy.

    Se usa para contrastar proporciones de supervivencia entre brazos, que es
    donde el prototipo anterior confundio una media con una probabilidad.
    """
    from math import comb
    n = a + b + c + d
    r1, r2 = a + b, c + d
    c1 = a + c

    def p_of(x):
        lo = max(0, c1 - r2)
        if x < lo or x > min(r1, c1):
            return 0.0
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)

    p_obs = p_of(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    total = sum(p_of(x) for x in range(lo, hi + 1)
                if p_of(x) <= p_obs + 1e-12)
    return float(min(1.0, total))


def prop_ci(k, n, alpha=ALPHA):
    """Intervalo de Wilson para una proporcion.  Correcto con k = 0, que es
    justo donde el intervalo normal falla."""
    if n == 0:
        return np.nan, np.nan, np.nan
    from math import sqrt
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else _z(alpha)
    p = k / n
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, cen - half), min(1.0, cen + half)


def _z(alpha):
    """Cuantil normal por la aproximacion racional de Abramowitz-Stegun 26.2.23.

    Error absoluto < 4.5e-4, sobradamente suficiente para un intervalo de
    Wilson informado a tres decimales.
    """
    from math import sqrt, log
    p = alpha / 2.0                      # cola superior
    t = sqrt(-2.0 * log(p))
    num = 2.515517 + 0.802853 * t + 0.010328 * t * t
    den = 1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t
    return t - num / den
