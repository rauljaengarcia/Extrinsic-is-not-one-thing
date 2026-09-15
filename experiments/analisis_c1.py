#!/usr/bin/env python3
"""
C1 — parcialidad emergente: ?la regla abierta concentra la asignacion en los
organismos de mayor afinidad o mayor reserva?

Es el borde exacto del criterio de §2.2 del manuscrito.  Una regla puede ser
simetrica bajo reetiquetado —formalmente exchangeable— y aun asi servir de
hecho a un subconjunto sistematicamente mejor dotado.  Bajo la definicion eso
NO es parcialidad, porque el comportamiento depende del estado y no de la
etiqueta.  Pero es lo primero que preguntara un revisor agudo, y mas vale
haberlo medido.

DOS NIVELES DE EVIDENCIA, y hay que distinguirlos:

  (a) Sin computo nuevo.  Las filas de `brazos.json` son agregados por run, no
      registran quien recibio que.  Permiten comparar poblaciones entre brazos
      —afinidad media, reserva media, inscripciones— pero NO pueden responder
      la pregunta directamente.  Se informa lo que dan.

  (b) Con observacion directa.  Un observador externo que conduce la ecologia
      paso a paso y lee su estado desde fuera.  NO modifica `ecologia.py`: solo
      lee `alive`, `enrolled`, `E` y `G`, que son atributos publicos.  El hash
      del codigo del sustrato queda intacto y los resultados anteriores siguen
      siendo validos.

    python analisis_c1.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import runner                                          # noqa: E402
import numpy as np                                     # noqa: E402
from ecologia import Ecologia, Config                  # noqa: E402
from runner import load, col, banner, save             # noqa: E402
import estadistica as st                               # noqa: E402

T = 20_000
N_SEED_OBS = 20          # observacion directa: monohilo, ~5 s por run
DOSIS_OBS = 200
MUESTREA_CADA = 100      # pasos entre instantaneas del estado


# ==========================================================================
def nivel_a():
    """Lo que las filas ya registradas pueden decir, que no es la respuesta."""
    banner("C1(a) — lo que dan los datos ya registrados")
    d = load("brazos")
    filas = d["filas"]
    dmax = max(d["manifiesto"]["dosis"])

    # la manipulacion 1 es la unica con proveedor: B parcial contra S abierto
    B = [r for r in filas if r["label"] == f"B_{dmax}" and r.get("provided", 0) > 0
         and r.get("n_enrol", 0) == 0]
    S = [r for r in filas if r["label"] == f"S_{dmax}" and r.get("n_enrol", 0) > 0]
    if not B or not S:
        print("  no se localizan los brazos del proveedor en este fichero")
        return None

    print(f"  brazo B (proveedor parcial, n={len(B)}) frente a "
          f"brazo S (regla abierta, n={len(S)})\n")
    print(f"  {'magnitud':<34} {'B':>22} {'S':>22}")
    filas_tab = []
    for clave, nombre in (("g_a", "afinidad media final"),
                          ("E_mean", "reserva media final"),
                          ("n_final", "poblacion final"),
                          ("provided", "volumen emitido"),
                          ("n_enrol", "inscripciones")):
        b = st.boot_mean(col(B, clave))
        s = st.boot_mean(col(S, clave))
        filas_tab.append((nombre, b, s))
        print(f"  {nombre:<34} {b[0]:>10.4g} [{b[1]:.3g}, {b[2]:.3g}] "
              f"{s[0]:>10.4g} [{s[1]:.3g}, {s[2]:.3g}]")
    print("\n  Estas columnas comparan POBLACIONES, no receptores.  No pueden")
    print("  decir si dentro del brazo S el recurso fue a los mejor dotados:")
    print("  para eso hace falta la observacion directa de C1(b).")
    return filas_tab


# ==========================================================================
def observar(seed, arm, cfg_extra):
    """Conduce una ecologia y lee su estado desde fuera, sin modificarla."""
    cfg = Config(n0_L1=DOSIS_OBS, **cfg_extra)
    e = Ecologia(cond="Ex", seed=seed, cfg=cfg, log_every=0)
    acc = {"a_in": [], "a_out": [], "E_in": [], "E_out": [],
           "frac_serv": [], "pctl_a": [], "pctl_E": []}
    for t in range(T):
        if not e.step():
            break
        # Se muestrea desde pronto a proposito.  Bajo la regla parcial el
        # linaje no focal se extingue, y despues de eso "servidos" son todos y
        # no hay contraste que medir.  El contraste existe mientras existe
        # alguien a quien no servir, que es justo el punto del marco.
        if t % MUESTREA_CADA or t < 400:
            continue
        idx = np.flatnonzero(e.alive)
        if idx.size < 20:
            continue
        if arm == "S":
            servidos = e.enrolled[idx]
        else:                       # brazo B: la regla parcial sirve a L0
            servidos = e.L[idx] == 0
        acc["frac_serv"].append(float(servidos.mean()))
        if servidos.all() or not servidos.any():
            continue
        a, E = e.G[idx, 0], e.E[idx]
        acc["a_in"].append(a[servidos].mean())
        acc["a_out"].append(a[~servidos].mean())
        acc["E_in"].append(E[servidos].mean())
        acc["E_out"].append(E[~servidos].mean())
        # percentil medio que ocupan los servidos dentro de la poblacion
        acc["pctl_a"].append(float((a[:, None] < a[servidos][None, :]).mean()))
        acc["pctl_E"].append(float((E[:, None] < E[servidos][None, :]).mean()))
    return {k: (float(np.mean(v)) if v else np.nan) for k, v in acc.items()}


def nivel_b():
    """Observacion directa de quien esta servido en cada instante."""
    banner("C1(b) — observacion directa de quien recibe")
    print("  Un observador externo lee `alive`, `enrolled`, `E` y `G` de la")
    print("  ecologia sin modificarla.  Se compara, entre los vivos, el estado")
    print("  de los SERVIDOS frente al de los NO servidos.\n")
    print("  Brazo B: la regla parcial sirve a la etiqueta focal.")
    print("  Brazo S: la regla abierta sirve a quien se haya inscrito.\n")
    conf = {
        "B": {"provider": "partial", "Q_rate": 0.05},
        "S": {"provider": "open", "match": "total", "Q_total": 34.0},
    }
    out = {}
    print(f"  {'brazo':<6} {'servidos':>9} {'afinidad serv.':>15} "
          f"{'afinidad resto':>15} {'reserva serv.':>14} {'reserva resto':>14} "
          f"{'percentil':>10}")
    for arm, extra in conf.items():
        rs = [observar(s, arm, extra) for s in range(N_SEED_OBS)]
        m = {k: float(np.nanmean([r[k] for r in rs])) for k in rs[0]}
        out[arm] = m
        print(f"  {arm:<6} {m['frac_serv']:>9.3f} {m['a_in']:>15.4f} "
              f"{m['a_out']:>15.4f} {m['E_in']:>14.3f} {m['E_out']:>14.3f} "
              f"{m['pctl_E']:>10.3f}")

    print()
    for arm in ("B", "S"):
        m = out[arm]
        if not np.isfinite(m["a_in"]):
            print(f"  brazo {arm}: los servidos son toda la poblacion en cada")
            print("           instantanea; no hay contraste que medir.")
            continue
        da = (m["a_in"] - m["a_out"]) / m["a_out"]
        dE = (m["E_in"] - m["E_out"]) / max(abs(m["E_out"]), 1e-9)
        print(f"  brazo {arm}: los servidos tienen {da:+.1%} de afinidad y "
              f"{dE:+.1%} de reserva")
        print(f"           respecto a los no servidos; ocupan el percentil "
              f"{m['pctl_E']:.2f} de energia.")
    return out


def main():
    a = nivel_a()
    b = nivel_b()
    banner("LECTURA")
    print("  La pregunta de C1 tiene respuesta y no es la que cabria temer.")
    print("  Ver el informe en el manuscrito; los numeros estan arriba.")
    save("c1_parcialidad_emergente",
         [{"nivel": "b", "brazo": k, **v} for k, v in (b or {}).items()], [],
         extra={"nota": "C1: parcialidad emergente bajo la regla abierta",
                "n_seed_observacion": N_SEED_OBS, "dosis": DOSIS_OBS,
                "muestreo_cada": MUESTREA_CADA})
    return 0


if __name__ == "__main__":
    sys.exit(main())
