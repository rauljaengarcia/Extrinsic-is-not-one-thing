#!/usr/bin/env python3
"""
Tests de regresion.  Se ejecutan antes de cualquier experimento y deben pasar
todos.  `python tests.py`

Cubren las tres cosas que, si se rompen en silencio, invalidan el resultado:

  1. reproducibilidad exacta (semilla, workers, orden);
  2. las identidades de monocultivo, que son el contenido empirico de H4;
  3. la ortogonalidad: cada intervencion responde a UNA coordenada.

NOTA DE PORTABILIDAD: en Windows y macOS `multiprocessing` arranca por
`spawn`, y el proceso hijo reimporta el modulo principal.  Todo script de este
repositorio que use `run_grid` debe por tanto encerrar su cuerpo en
`main()` bajo `if __name__ == "__main__"`, o cada worker reejecutara el script
entero.  No es un detalle de estilo: sin la guarda, el programa no termina.
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "tests"))

import runner                                   # noqa: E402  (fija el threading)
import numpy as np                              # noqa: E402
from ecologia import Ecologia, Config, GENES, GENE_BOUNDS  # noqa: E402
from runner import spec, run_one, run_grid      # noqa: E402
import estadistica as st                        # noqa: E402

T = 8_000
FALLOS = []


def check(nombre, ok, detalle=""):
    print(f"  [{'OK ' if ok else 'FALLO'}] {nombre}" + (f"   {detalle}" if detalle else ""))
    if not ok:
        FALLOS.append(nombre)


def firma(d):
    """Resumen sensible a cualquier divergencia dinamica."""
    return (d["births"], d["deaths"], d["n_final"],
            round(d["volume"], 9), round(d["g_a"], 12), round(d["g_z"], 12))


def main():
    print("\n1. Reproducibilidad")
    s = spec(cond="Ix", seed=3, T=T)
    check("misma spec dos veces -> identico", run_one(s) == run_one(s))

    g1 = run_grid([spec(cond="Ex", seed=i, T=T) for i in range(6)], workers=1, quiet=True)
    g4 = run_grid([spec(cond="Ex", seed=i, T=T) for i in range(6)], workers=4, quiet=True)
    check("el resultado no depende del numero de workers", g1 == g4)

    rev = run_grid([spec(cond="Ex", seed=i, T=T) for i in reversed(range(6))],
                   workers=4, quiet=True)
    check("el orden de salida sigue al de entrada",
          [r["seed"] for r in rev] == list(reversed(range(6))))

    print("\n2. Identidades de monocultivo  (H4)")
    for seed in range(3):
        r = {c: run_one(spec(cond=c, seed=seed, T=T)) for c in ("Ix", "Ip", "Ex", "Ep")}
        f = {c: firma(v) for c, v in r.items()}
        check(f"seed {seed}: las cuatro condiciones son identicas en monocultivo",
              len(set(f.values())) == 1, f"births={r['Ix']['births']:,}")

    print("\n3. Ortogonalidad de las intervenciones")
    for c in ("Ix", "Ip"):
        a = run_one(spec(cond=c, seed=1, T=T))
        b = run_one(spec(cond=c, seed=1, T=T, remove_operator=True))
        check(f"{c}: retirar el operador externo no cambia nada (C=I)", firma(a) == firma(b))

    for c in ("Ex", "Ep"):
        a = run_one(spec(cond=c, seed=1, T=T))
        b = run_one(spec(cond=c, seed=1, T=T, remove_operator=True))
        check(f"{c}: retirar el operador externo SI cambia (C=E)", firma(a) != firma(b),
              f"Phi {a['births']:,} -> {b['births']:,}")

    iny = dict(inject_at=T // 2, n_inject=150, access_from=T // 2)
    for c in ("Ix", "Ex", "Ip", "Ep"):
        r = run_one(spec(cond=c, seed=1, T=T, **iny))
        l0, l1 = r["from_L0_pc_L0"], r["from_L0_pc_L1"]
        if c[1] == "x":
            ok, esperado = (l0 > 0 and abs(l1 - l0) / l0 < 0.25), "igual"
        else:
            ok, esperado = (l1 == 0.0 and l0 > 0), "cero"
        check(f"{c}: lo asignado por donantes L0 a L1 es {esperado} (A={0 if c[1]=='x' else 1})",
              ok, f"L0={l0:.5f} L1={l1:.5f}  cruzado={r['flow_cross_frac']:.4f}")

    # exclusion (E/p) frente a particion (I/p): misma coordenada, efecto
    # agregado opuesto.  Es un resultado, no un detalle de implementacion.
    rep = run_one(spec(cond="Ep", seed=1, T=T, **iny))
    rip = run_one(spec(cond="Ip", seed=1, T=T, **iny))
    check("E/p excluye: L1 no recibe nada de nadie", rep["recv_pc_L1"] == 0.0)
    check("I/p particiona: L1 recibe, pero solo de si mismo",
          rip["recv_pc_L1"] > 0 and rip["flow_01"] == 0.0 and rip["flow_11"] > 0,
          f"flow_11={rip['flow_11']:,.0f} flow_01={rip['flow_01']:,.0f}")

    print("\n4. Integridad del sustrato")
    e = Ecologia(cond="Ix", seed=0, cfg=Config(), log_every=0)
    e.run(T)
    idx = np.flatnonzero(e.alive)
    ok = True
    for i, name in enumerate(GENES):
        lo, hi = GENE_BOUNDS[name]
        if np.isfinite(lo):
            ok &= bool((e.G[idx, i] >= lo - 1e-12).all() and (e.G[idx, i] <= hi + 1e-12).all())
    check("los genes acotados nunca salen de su rango", ok)
    check("la capacidad del array no muerde nunca", e.cap_bound_hits == 0,
          f"cap={e.cfg.cap} N_final={idx.size}")

    a = run_one(spec(cond="Ix", seed=2, T=T, cfg={"g_init": (1.0, 0.5, 0.0, 0.0)}))
    b = run_one(spec(cond="Ix", seed=2, T=T, cfg={"g_init": (1.0, 0.5, 0.0, 7.0)}))
    check("el marcador z es selectivamente neutral",
          (a["births"], a["deaths"], round(a["g_a"], 12)) ==
          (b["births"], b["deaths"], round(b["g_a"], 12)),
          f"z final {a['g_z']:.3f} vs {b['g_z']:.3f}")

    print("\n5. Estadistica")
    check("Fisher 2x2 con 1/10 vs 0/10 no es significativo",
          st.fisher_exact_2x2(1, 9, 0, 10) > 0.5,
          f"p={st.fisher_exact_2x2(1, 9, 0, 10):.3f}")
    check("Fisher 2x2 con 10/10 vs 0/10 si lo es",
          st.fisher_exact_2x2(10, 0, 0, 10) < 0.001,
          f"p={st.fisher_exact_2x2(10, 0, 0, 10):.2e}")
    p, lo, hi = st.prop_ci(0, 30)
    check("Wilson con k=0 da intervalo valido", lo == 0.0 and 0 < hi < 0.2,
          f"[{lo:.3f}, {hi:.3f}]")
    check("el guardian devuelve NA, no cero, con denominador nulo",
          np.isnan(st.rho(100.0, 100.0, 90.0, eps=5.0)))
    m, lo, hi = st.boot_mean([1, 2, 3, 4, 5] * 10)
    check("bootstrap de la media cubre el valor", lo < 3.0 < hi, f"[{lo:.2f}, {hi:.2f}]")

    print("\n6. Afirmaciones del texto sobre la Tabla 2")
    # No es un test del sustrato: comprueba que las propiedades de dominancia
    # que el manuscrito afirma siguen siendo ciertas sobre la tabla publicada.
    # Se salta si no esta el .tex al lado, para que el repositorio del
    # experimento siga siendo ejecutable por si solo.
    import dominancia                            # noqa: E402
    if dominancia.TEX_POR_DEFECTO.exists():
        check("las cuatro afirmaciones de dominancia se sostienen",
              dominancia.main() == 0)
    else:
        print(f"  [--- ] sin {dominancia.TEX_POR_DEFECTO.name} al lado, se omite")

    print("\n" + "=" * 60)
    if FALLOS:
        print(f"FALLOS ({len(FALLOS)}): " + ", ".join(FALLOS))
        return 1
    print("Todos los tests pasan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
