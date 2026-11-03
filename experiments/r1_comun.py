"""
Comun a los scripts de la R1: rutas, procedencia y lectura de lo publicado.

Las rutas salen de la posicion de este fichero, como en `reproduce.py`: la raiz
es la carpeta que contiene `experiments/`, y las salidas van a `results/` y
`figures/` de esa raiz, no a `resultados/` y `figuras/`, que son los nombres
que `src/runner.py` deriva de si mismo.  `preparar()` redirige `runner` antes de
usarlo, igual que `reproduce.preparar()`, sin tocar `src/runner.py`.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
SRC = RAIZ / "src"
RESULTS = RAIZ / "results"
FIGURES = RAIZ / "figures"

#: los cuatro ficheros de `src/` que produjeron los resultados publicados, y su
#: hash tal como lo calcula `runner.code_hash()` (manifiestos de 2026-09-12/13)
SRC_PUBLICADO = ("ecologia.py", "estadistica.py", "estilo.py", "runner.py")
HASH_SRC_PUBLICADO = "2f4ecfd3aa636719"
#: el codigo nuevo de la R1, para el manifiesto
CODIGO_R1 = ("src/ecologia_graduada.py", "src/estadistica_r1.py",
             "experiments/r1_comun.py", "experiments/r1_bootstrap.py",
             "experiments/r1_sensibilidad_coste.py", "experiments/r1_topologia.py",
             "experiments/r1_graduada.py", "tests/test_r1.py")


def preparar():
    """Redirige las salidas de `runner` a results/ y figures/, como reproduce.py."""
    import runner
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    runner.RESULTADOS = RESULTS
    runner.FIGURAS = FIGURES
    return runner


def sha256_16(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()[:16]


def hash_src_publicado():
    """`runner.code_hash()` restringido a los cuatro ficheros publicados."""
    h = hashlib.sha256()
    for nombre in sorted(SRC_PUBLICADO):
        p = SRC / nombre
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def procedencia(script, fuentes=()):
    """Lo que el manifiesto anade a `runner.save`: que codigo y que datos.

    `code_sha256_16` (lo pone `runner.save`) resume TODO `src/*.py`, incluidos
    los dos modulos nuevos; aqui se anade el hash de los cuatro publicados, el
    del script y el de cada fichero de la R1, y de cada fuente leida su hash y
    el `code_sha256_16` de su propio manifiesto.
    """
    import matplotlib
    import runner
    out = {
        "script": Path(script).name,
        "script_sha256_16": sha256_16(script),
        "src_publicado_sha256_16": hash_src_publicado(),
        "src_publicado_intacto": hash_src_publicado() == HASH_SRC_PUBLICADO,
        "code_sha256_16_src_completo": runner.code_hash(),
        "codigo_r1_sha256_16": {f: sha256_16(RAIZ / f) for f in CODIGO_R1
                                if (RAIZ / f).is_file()},
        "matplotlib": matplotlib.__version__,
        "python": platform.python_version(),
    }
    fs = {}
    for nombre in fuentes:
        p = RESULTS / nombre
        d = {"sha256_16": sha256_16(p)}
        if p.suffix == ".json":
            m = json.loads(p.read_text(encoding="utf-8")).get("manifiesto", {})
            d["code_sha256_16"] = m.get("code_sha256_16")
            d["fecha_utc"] = m.get("fecha_utc")
        fs[nombre] = d
    out["fuentes"] = fs
    return out


def cargar(nombre):
    return json.loads((RESULTS / f"{nombre}.json").read_text(encoding="utf-8"))


def iv(t):
    """Intervalo como lista de tres floats, para el JSON."""
    return [float(t[0]), float(t[1]), float(t[2])]


def fmt(t, d=3):
    """'0.989 [0.980, 0.999]', o 'NA'."""
    if t is None or not all(isinstance(x, (int, float)) for x in t[:3]):
        return "NA"
    p, lo, hi = t[:3]
    if p != p:
        return "NA"
    if lo != lo or hi != hi:
        return f"{p:.{d}f} [NA]"
    return f"{p:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def iguales(a, b):
    """Igualdad exacta de dos valores de una fila, con NaN == NaN."""
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    return a == b


def misma_fila(a, b, ignorar=("label", "arm", "cond")):
    """Claves en que dos filas difieren (las de `ignorar` no cuentan)."""
    claves = (set(a) | set(b)) - set(ignorar)
    return sorted(k for k in claves if k not in a or k not in b or not iguales(a[k], b[k]))
