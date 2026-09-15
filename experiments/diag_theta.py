#!/usr/bin/env python3
"""Diagnostico: por que el cociente de acceso se aparta de 1 con theta=7.0.

La hipotesis es falta de potencia, no un efecto sistematico: con el umbral
reproductivo por debajo del de aportacion casi nadie llega a ser donante, se
producen muy pocas asignaciones en la ventana, y el cociente queda dominado
por si un individuo extremo cayo o no del lado de los clones.

NOTA: cualquier script que use run_grid tiene que ser un FICHERO con guarda
`__main__`.  Bajo `spawn` los workers reimportan el modulo principal, y un
script pasado por stdin no tiene ruta que reimportar.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import runner                                    # noqa: E402
import numpy as np                               # noqa: E402
from runner import spec, run_grid, col           # noqa: E402

T = 15_000
INY = dict(inject_at=T // 2, n_inject=150, access_from=T // 2,
           access_until=T // 2 + 600)


def main():
    print(f"  {'theta':>6} {'asignaciones':>13} {'minimo':>8} {'cociente':>9} "
          f"{'IC 95%':>20} {'cubre 1':>8}")
    for th in (7.0, 8.5, 10.0, 14.0):
        rows = run_grid([spec(cond="Ex", seed=s, T=T, cfg={"theta": th}, **INY)
                         for s in range(60)], quiet=True)
        f00, f01 = col(rows, "flow_00"), col(rows, "flow_01")
        a0, a1 = col(rows, "from_L0_pc_L0"), col(rows, "from_L0_pc_L1")
        n_ev = (f00 + f01) / 3.0                       # q = 3 por asignacion
        rng = np.random.default_rng(0)
        v = []
        for _ in range(4000):
            i = rng.integers(0, len(rows), len(rows))
            d = a0[i].mean()
            v.append(a1[i].mean() / d if d > 0 else np.nan)
        v = np.array(v)
        v = v[np.isfinite(v)]
        lo, hi = np.percentile(v, [2.5, 97.5])
        rat = a1.mean() / a0.mean()
        print(f"  {th:>6.1f} {n_ev.mean():>13.0f} {n_ev.min():>8.0f} "
              f"{rat:>9.3f}   [{lo:.3f}, {hi:.3f}] {str(lo <= 1 <= hi):>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
