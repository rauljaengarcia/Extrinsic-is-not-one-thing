"""
Estadistica de la R1: bootstrap emparejado por semilla, percentil y BCa.

`percentil_pareado` replica exactamente `boot_pareado` de
`experiments/boot_emparejado.py`, que es el estimador publicado: por replica se
remuestrea UNA vez el vector de semillas y se aplica el mismo a todos los
brazos; `N_BOOT = 10 000` replicas, percentiles 2,5 / 97,5, `BOOT_SEED` de
`estadistica.py`.  Las replicas en que el estadistico no se define (el
guardian de rho devuelve NaN) se descartan antes de tomar los percentiles.

`bca_pareado` es el segundo estimador que pide el Revisor 1 (R1.Q3): BCa
(bias-corrected and accelerated; Efron 1987) sobre las MISMAS replicas, con

    z0 = Phi^-1( (#{theta* < theta^} + #{theta* = theta^}/2) / B )
    a  = sum (m - theta_(i))^3 / (6 [sum (m - theta_(i))^2]^(3/2))

donde theta_(i) es el estadistico sin la semilla i en ningun brazo (jackknife
emparejado sobre semillas) y m su media.  Los extremos son los percentiles
Phi(z0 + (z0 + z_q) / (1 - a (z0 + z_q))) de las replicas, con z_q en 2,5 y
97,5 %.  El guardian se trata igual que en el percentil publicado: las replicas
NaN se descartan, tanto para z0 como para los percentiles, y los theta_(i)
NaN se descartan del jackknife; los dos recuentos se devuelven.  Si todas las
replicas son iguales (cocientes exactos, como el 1.000 de C = I) el intervalo
es ese valor, sin BCa que calcular.

Solo NumPy y la biblioteca estandar: la normal, con math.erf.
"""
from __future__ import annotations

import math

import numpy as np

from estadistica import ALPHA, BOOT_SEED, N_BOOT


# --------------------------------------------------------------------------
# la normal
# --------------------------------------------------------------------------

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p):
    """Inversa de la normal por biseccion sobre erf: lenta y exacta."""
    if not 0.0 < p < 1.0:
        return math.copysign(math.inf, p - 0.5) if p in (0.0, 1.0) else math.nan
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def t_ppf(p, gl):
    """Cuantil de la t de Student por Cornish-Fisher (Abramowitz-Stegun 26.7.5).

    Solo para la prueba de referencia de BCa: con gl = 99 el error es < 1e-6.
    """
    z = norm_ppf(p)
    g1 = (z ** 3 + z) / 4
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384
    g4 = (79 * z ** 9 + 776 * z ** 7 + 1482 * z ** 5 - 1920 * z ** 3 - 945 * z) / 92160
    return z + g1 / gl + g2 / gl ** 2 + g3 / gl ** 3 + g4 / gl ** 4


# --------------------------------------------------------------------------
# bootstrap emparejado
# --------------------------------------------------------------------------

def _alinear(muestras):
    ms = [np.asarray(s, dtype=np.float64) for s in muestras]
    n = {s.size for s in ms}
    if len(n) != 1:
        raise ValueError(f"los brazos no estan emparejados: tamanos {n}")
    # una semilla solo entra si esta viva en todos los brazos
    ok = np.all([np.isfinite(s) for s in ms], axis=0)
    return [s[ok] for s in ms]


def replicas_pareadas(fn, *muestras, n_boot=N_BOOT, seed=BOOT_SEED):
    """(theta^, replicas, muestras alineadas).  Mismo consumo del generador que
    `boot_emparejado.boot_pareado`: un `integers(0, n, n)` por replica."""
    ms = _alinear(muestras)
    n = ms[0].size
    if n == 0:
        return np.nan, np.full(n_boot, np.nan), ms
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = fn(*[s[idx].mean() for s in ms])
    return float(fn(*[s.mean() for s in ms])), vals, ms


def percentil_pareado(fn, *muestras, n_boot=N_BOOT, alpha=ALPHA, seed=BOOT_SEED,
                      _replicas=None):
    """(punto, lo, hi): el percentil emparejado publicado."""
    punto, vals, _ = _replicas or replicas_pareadas(fn, *muestras, n_boot=n_boot, seed=seed)
    fin = vals[np.isfinite(vals)]
    if fin.size == 0:
        return float(punto), np.nan, np.nan
    lo, hi = np.percentile(fin, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(punto), float(lo), float(hi)


def jackknife(fn, ms):
    """theta_(i): el estadistico sin la semilla i, en todos los brazos a la vez."""
    n = ms[0].size
    sumas = [s.sum() for s in ms]
    return np.array([fn(*[(S - s[i]) / (n - 1) for S, s in zip(sumas, ms)])
                     for i in range(n)], dtype=np.float64)


def bca_pareado(fn, *muestras, n_boot=N_BOOT, alpha=ALPHA, seed=BOOT_SEED,
                _replicas=None):
    """BCa emparejado.  Devuelve un dict con el intervalo y su diagnostico."""
    punto, vals, ms = _replicas or replicas_pareadas(fn, *muestras, n_boot=n_boot, seed=seed)
    fin = vals[np.isfinite(vals)]
    out = {"punto": float(punto), "lo": np.nan, "hi": np.nan, "z0": np.nan, "a": np.nan,
           "alfa_lo": np.nan, "alfa_hi": np.nan, "replicas_nan": int(vals.size - fin.size),
           "jackknife_nan": 0, "nota": ""}
    if not np.isfinite(punto) or fin.size == 0:
        out["nota"] = "sin punto o sin replicas finitas: NA"
        return out
    if np.all(fin == fin[0]):
        out.update(lo=float(fin[0]), hi=float(fin[0]), z0=0.0, a=0.0,
                   alfa_lo=alpha / 2, alfa_hi=1 - alpha / 2,
                   nota="todas las replicas iguales: intervalo degenerado")
        return out
    p0 = (np.sum(fin < punto) + 0.5 * np.sum(fin == punto)) / fin.size
    if not 0.0 < p0 < 1.0:
        out["nota"] = f"z0 infinito (fraccion por debajo del punto = {p0}): NA"
        return out
    z0 = norm_ppf(p0)
    jack = jackknife(fn, ms)
    jf = jack[np.isfinite(jack)]
    out["jackknife_nan"] = int(jack.size - jf.size)
    d = jf.mean() - jf
    den = 6.0 * float(np.sum(d ** 2)) ** 1.5
    a = float(np.sum(d ** 3)) / den if den > 0 else 0.0
    alfas = []
    for q in (alpha / 2, 1 - alpha / 2):
        zq = norm_ppf(q)
        den_q = 1.0 - a * (z0 + zq)
        if den_q <= 0:
            out.update(z0=z0, a=a, nota="1 - a(z0 + z) <= 0: NA")
            return out
        alfas.append(norm_cdf(z0 + (z0 + zq) / den_q))
    lo, hi = np.percentile(fin, [100 * alfas[0], 100 * alfas[1]])
    out.update(lo=float(lo), hi=float(hi), z0=float(z0), a=a,
               alfa_lo=float(alfas[0]), alfa_hi=float(alfas[1]))
    return out


def ambos(fn, *muestras, n_boot=N_BOOT, alpha=ALPHA, seed=BOOT_SEED):
    """Percentil y BCa emparejados sobre las mismas replicas."""
    rep = replicas_pareadas(fn, *muestras, n_boot=n_boot, seed=seed)
    return (percentil_pareado(fn, n_boot=n_boot, alpha=alpha, _replicas=rep),
            bca_pareado(fn, n_boot=n_boot, alpha=alpha, _replicas=rep))


# --------------------------------------------------------------------------
# lectura de intervalos
# --------------------------------------------------------------------------

def signo(iv, ref=0.0):
    """'negativo' / 'positivo' si el intervalo excluye `ref`, si no 'compatible'."""
    lo, hi = iv[1], iv[2]
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "NA"
    if hi < ref:
        return "negativo"
    if lo > ref:
        return "positivo"
    return "compatible"


def cubre(iv, x):
    return bool(np.isfinite(iv[1]) and np.isfinite(iv[2]) and iv[1] <= x <= iv[2])


def solapan(a, b):
    return bool(a[1] <= b[2] and b[1] <= a[2])


def cociente(x, y):
    """y / x, para los cocientes de medias: Phi_R/Phi_B, L1/L0."""
    return y / x if x != 0 else np.nan
