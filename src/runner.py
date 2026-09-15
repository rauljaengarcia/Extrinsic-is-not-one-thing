"""
Ejecucion reproducible de lotes de simulaciones.

Garantias que ofrece este modulo, y que se comprueban en `tests.py`:

  * una `RunSpec` determina el run por completo: misma spec -> misma salida,
    bit a bit, en cualquier maquina con la misma version de numpy;
  * el resultado no depende del numero de workers ni del orden de ejecucion,
    porque cada run lleva su propia semilla y no se comparte estado;
  * cada lote guarda un manifiesto con las especificaciones completas, el
    hash del codigo fuente, las versiones del entorno y la fecha, de modo que
    un fichero de resultados nunca queda huerfano de las condiciones que lo
    produjeron.

Limitar el threading de numpy ANTES de importarlo es obligatorio: sin eso los
workers compiten entre si por los mismos nucleos y el speedup se desploma.
Por eso este modulo lo hace en su cabecera y los scripts de experimento deben
importarlo antes que numpy.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import atexit
import hashlib
import json
import multiprocessing as mp
import platform
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from ecologia import Ecologia, Config

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
RESULTADOS = ROOT / "resultados"
FIGURAS = ROOT / "figuras"


# --------------------------------------------------------------------------
# Especificacion de un run
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RunSpec:
    """Todo lo que determina un run.  Nada fuera de aqui influye en el
    resultado."""
    cond: str = "Ix"
    seed: int = 0
    T: int = 20_000
    cfg: tuple = ()                     # pares (clave, valor) para Config
    remove_operator: bool = False
    inject_at: int | None = None
    n_inject: int = 0
    access_from: int | None = None
    access_until: int | None = None
    arm: str = ""                       # etiqueta libre: 'B', 'R', 'S', ...
    label: str = ""                     # etiqueta libre del experimento

    def config(self) -> Config:
        return Config(**dict(self.cfg))

    def to_json(self):
        d = asdict(self)
        d["cfg"] = dict(self.cfg)
        return d


def spec(**kw) -> RunSpec:
    """Constructor comodo: acepta `cfg` como dict y lo congela."""
    if "cfg" in kw and isinstance(kw["cfg"], dict):
        kw["cfg"] = tuple(sorted(kw["cfg"].items()))
    return RunSpec(**kw)


# --------------------------------------------------------------------------
# Ejecucion
# --------------------------------------------------------------------------

def run_one(s: RunSpec) -> dict:
    e = Ecologia(cond=s.cond, seed=s.seed, cfg=s.config(),
                 remove_operator=s.remove_operator, log_every=0)
    out = e.run(s.T, inject_at=s.inject_at, n_inject=s.n_inject,
                access_from=s.access_from, access_until=s.access_until)
    if s.access_from is not None:
        acc = e.access()
        for tag in (0, 1):
            out[f"recv_pc_L{tag}"] = float(acc["recv_pc"][tag])
            out[f"from_L0_pc_L{tag}"] = float(acc["from_L0_pc"][tag])
            out[f"prov_pc_L{tag}"] = float(acc["prov_pc"][tag])
            out[f"born_pc_L{tag}"] = float(acc["born_pc"][tag])
            out[f"expo_L{tag}"] = float(acc["expo"][tag])
        f = acc["flow"]
        out["flow_00"], out["flow_01"] = float(f[0, 0]), float(f[0, 1])
        out["flow_10"], out["flow_11"] = float(f[1, 0]), float(f[1, 1])
        cruzado = f[0, 1] + f[1, 0]
        out["flow_cross_frac"] = float(cruzado / f.sum()) if f.sum() > 0 else np.nan
    out["arm"] = s.arm
    out["label"] = s.label
    out["T"] = s.T
    return out


def _worker(payload):
    return run_one(RunSpec(**payload))


def code_hash() -> str:
    """Hash del codigo fuente que produce los resultados."""
    h = hashlib.sha256()
    for p in sorted(SRC.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }


_POOL = {"n": None, "p": None}


def _pool(workers):
    """Pool persistente.

    Crear y destruir un Pool en cada llamada es carisimo bajo `spawn`: cada
    worker reimporta numpy.  Con decenas de llamadas cortas -- que es el caso
    del calibrado, donde se barre un parametro -- el arranque domina el tiempo
    total y el proceso padre pasa casi todo el rato bloqueado.  El pool se
    reutiliza mientras no cambie el numero de workers.

    Esto no afecta a la reproducibilidad: cada run lleva su propia semilla y no
    comparte estado con los demas.  `tests.py` lo comprueba ejecutando la misma
    rejilla con 1 y con 4 workers.
    """
    if _POOL["p"] is None or _POOL["n"] != workers:
        cerrar_pool()
        _POOL["n"] = workers
        _POOL["p"] = mp.Pool(workers)
    return _POOL["p"]


def cerrar_pool():
    if _POOL["p"] is not None:
        _POOL["p"].terminate()
        _POOL["p"].join()
        _POOL["p"] = None
        _POOL["n"] = None


def run_grid(specs, workers=None, quiet=False):
    """Ejecuta una lista de RunSpec.  El orden de la salida coincide con el de
    la entrada, independientemente del reparto entre workers."""
    specs = list(specs)
    workers = workers or min(12, os.cpu_count() or 1)
    t0 = time.perf_counter()
    payloads = [asdict(s) for s in specs]
    if workers == 1:
        rows = [_worker(p) for p in payloads]
    else:
        rows = _pool(workers).map(_worker, payloads, chunksize=1)
    dt = time.perf_counter() - t0
    if not quiet:
        print(f"  {len(specs)} runs en {dt:,.1f} s con {workers} workers "
              f"({len(specs)/dt:,.1f} runs/s)")
    return rows


def save(name, rows, specs, extra=None):
    """Guarda resultados y manifiesto juntos.  Nunca uno sin el otro."""
    RESULTADOS.mkdir(exist_ok=True)
    payload = {
        "manifiesto": {
            "nombre": name,
            "fecha_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code_sha256_16": code_hash(),
            "entorno": environment(),
            "n_runs": len(rows),
            "specs": [s.to_json() for s in specs],
            **(extra or {}),
        },
        "filas": rows,
    }
    path = RESULTADOS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, default=_jsonable),
                    encoding="utf-8")
    print(f"  -> {path.relative_to(ROOT)}  ({len(rows)} filas)")
    return path


def load(name):
    path = RESULTADOS / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"no serializable: {type(o)}")


atexit.register(cerrar_pool)


def col(rows, key):
    """Extrae una columna como array, con NaN donde falte."""
    return np.array([r.get(key, np.nan) for r in rows], dtype=np.float64)


def banner(txt):
    print("\n" + "=" * 74)
    print(txt)
    print("=" * 74)
