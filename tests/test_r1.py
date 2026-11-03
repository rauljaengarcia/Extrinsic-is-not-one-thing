#!/usr/bin/env python3
"""
Tests de la R1.  `python tests/test_r1.py` desde la raiz del repositorio.

    1. src/ publicado intacto: el hash de los cuatro ficheros que produjeron
       los resultados publicados sigue siendo 2f4ecfd3aa636719.
    2. Reglas graduadas, extremos bit a bit (ecologia_graduada.py):
       E w=0 == E/p, E w=1 == E/x, I w=0 == I/p, tau=0 == x y tau=1 == p en I
       y en E; y, sin graduar, EcologiaGraduada == Ecologia.  I w=1 NO es I/x
       bit a bit, y se comprueba que no lo es (sortea en dos tandas).
    3. Regresion de r1_sensibilidad_coste.py: en (k_c, c0) = (0.25, 0.18), la
       semilla 0 de los brazos B y R coincide fila a fila con las filas E/x de
       results/ortogonalidad.json; y, si ya existe, el JSON de la rejilla dice
       que coincidieron las 100 filas.
    4. BCa frente a un caso de referencia: la media de una muestra normal, donde
       BCa ~ percentil ~ intervalo t.
    5. kappa y s de r1_topologia.py reproducidos desde ortogonalidad.json, con
       los valores esperados, y kappa igual al cociente L1/L0 de la Tabla 4.

Escala de los runs de 2: T = 8 000, como tests/tests.py, con el protocolo de
reetiquetado escalado (inyeccion en T/2, ventana de 600 pasos), semillas 0..2.
"""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "experiments"))

import runner                                   # noqa: E402  (fija el threading)
import numpy as np                              # noqa: E402
import r1_comun as rc                           # noqa: E402
import estadistica as st                        # noqa: E402
import estadistica_r1 as er                     # noqa: E402
from ecologia_graduada import (EcologiaGraduada, run_one_graduada,  # noqa: E402
                               spec_graduada)
from runner import spec, run_one, run_grid      # noqa: E402

T = 8_000
SEMILLAS = (0, 1, 2)
INY = dict(inject_at=T // 2, n_inject=150, access_from=T // 2, access_until=T // 2 + 600)
NO_COMPARAR = ("label", "arm", "cond", "w", "tau", "periodo", "ajenos_vistos",
               "ajenos_admitidos", "pasos_parciales", "pasos_abiertos")
FALLOS = []


def check(nombre, ok, detalle=""):
    print(f"  [{'OK ' if ok else 'FALLO'}] {nombre}" + (f"   {detalle}" if detalle else ""))
    if not ok:
        FALLOS.append(nombre)


def igual(a, b):
    return not rc.misma_fila(a, b, NO_COMPARAR)


def main():
    print("\n1. src/ publicado intacto")
    h = rc.hash_src_publicado()
    check("hash de ecologia, estadistica, estilo y runner == 2f4ecfd3aa636719",
          h == rc.HASH_SRC_PUBLICADO, h)
    nuevos = sorted(p.name for p in rc.SRC.glob("*.py") if p.name not in rc.SRC_PUBLICADO)
    check("en src/ solo hay modulos nuevos, ninguno sustituido",
          set(nuevos) <= {"ecologia_graduada.py", "estadistica_r1.py"}, ", ".join(nuevos))

    print("\n2. Reglas graduadas: extremos bit a bit")
    for s in SEMILLAS:
        base = {c: run_one(spec(cond=c, seed=s, T=T, **INY)) for c in ("Ix", "Ip", "Ex", "Ep")}
        casos = [("E w=0 == E/p", dict(cond="Ep", w=0.0), "Ep"),
                 ("E w=1 == E/x", dict(cond="Ep", w=1.0), "Ex"),
                 ("I w=0 == I/p", dict(cond="Ip", w=0.0), "Ip"),
                 ("E tau=0 == E/x", dict(cond="Ep", tau=0.0), "Ex"),
                 ("E tau=1 == E/p", dict(cond="Ep", tau=1.0), "Ep"),
                 ("I tau=0 == I/x", dict(cond="Ip", tau=0.0), "Ix"),
                 ("I tau=1 == I/p", dict(cond="Ip", tau=1.0), "Ip"),
                 ("sin graduar == Ecologia", dict(cond="Ep"), "Ep")]
        for nombre, kw, ref in casos:
            g = run_one_graduada(spec_graduada(seed=s, T=T, **kw, **INY))
            check(f"seed {s}: {nombre}", igual(g, base[ref]),
                  f"births {g['births']:,} vs {base[ref]['births']:,}")
        g = run_one_graduada(spec_graduada(cond="Ip", seed=s, T=T, w=1.0, **INY))
        check(f"seed {s}: I w=1 NO es I/x bit a bit (dos tandas)", not igual(g, base["Ix"]),
              f"births {g['births']:,} vs {base['Ix']['births']:,}")
    # los intermedios no son ninguno de los extremos, y admiten la fraccion pedida
    e = EcologiaGraduada(cond="Ep", seed=0, w=0.5, log_every=0)
    e.run(T, **INY)
    frac = e.ajenos_admitidos / e.ajenos_vistos
    check("E w=0.5 admite ~la mitad de los ajenos", abs(frac - 0.5) < 0.02, f"{frac:.4f}")
    e = EcologiaGraduada(cond="Ip", seed=0, tau=0.25, log_every=0)
    e.run(T, **INY)
    frac = e.pasos_parciales / (e.pasos_parciales + e.pasos_abiertos)
    check("I tau=0.25: un cuarto de los pasos con transferencia son parciales",
          abs(frac - 0.25) < 0.01, f"{frac:.4f}")
    try:
        EcologiaGraduada(cond="Ix", w=0.5)
        check("graduar una regla ya abierta es un error", False)
    except ValueError:
        check("graduar una regla ya abierta es un error", True)

    print("\n3. Regresion de r1_sensibilidad_coste.py (k_c, c0) = (0.25, 0.18)")
    orto = rc.cargar("ortogonalidad")["filas"]
    pub = {r["label"]: r for r in orto if r["seed"] == 0 and r["label"] in ("Ex_B", "Ex_R")}
    filas = run_grid([spec(cond="Ex", seed=0, T=20_000, cfg={"kc": 0.25, "c0": 0.18},
                           remove_operator=rm) for rm in (False, True)], workers=2, quiet=True)
    for r, lab in zip(filas, ("Ex_B", "Ex_R")):
        check(f"semilla 0, brazo {lab[-1]}: fila identica a {lab} publicada",
              not rc.misma_fila(r, pub[lab]), f"phi_focal {r['phi_focal']:,}")
    p = rc.RESULTS / "r1_sensibilidad_coste.json"
    if p.is_file():
        reg = json.loads(p.read_text(encoding="utf-8"))["manifiesto"]["regresion"]
        check("r1_sensibilidad_coste.json: las 100 filas de la regresion coincidieron",
              bool(reg.get("ok")) and reg.get("filas_identicas") == 100,
              f"{reg.get('filas_identicas')}/{reg.get('filas')}")
    else:
        check("r1_sensibilidad_coste.json existe", False, "falta: correr el experimento")

    print("\n4. BCa frente a un caso de referencia (media de una normal)")
    x = np.random.default_rng(12345).normal(10.0, 2.0, 100)
    perc, bca = er.ambos(lambda m: m, x)
    sem = x.std(ddof=1) / np.sqrt(x.size)
    t = er.t_ppf(0.975, x.size - 1)
    tiv = (x.mean() - t * sem, x.mean() + t * sem)
    ancho = tiv[1] - tiv[0]
    dmax = max(abs(bca["lo"] - tiv[0]), abs(bca["hi"] - tiv[1]),
               abs(perc[1] - tiv[0]), abs(perc[2] - tiv[1]))
    check("t_ppf(0.975, 99) == 1.98422 (tablas)", abs(t - 1.9842169515) < 1e-6, f"{t:.7f}")
    check("BCa ~ percentil ~ t: extremos a menos del 5 % del ancho",
          dmax < 0.05 * ancho,
          f"t [{tiv[0]:.3f}, {tiv[1]:.3f}]  perc [{perc[1]:.3f}, {perc[2]:.3f}]  "
          f"BCa [{bca['lo']:.3f}, {bca['hi']:.3f}]  z0={bca['z0']:+.3f} a={bca['a']:+.4f}")
    y = np.random.default_rng(7).exponential(1.0, 60)
    perc, bca = er.ambos(lambda m: m, y)
    check("con una exponencial BCa se desplaza a la derecha del percentil",
          bca["lo"] > perc[1] and bca["hi"] > perc[2],
          f"perc [{perc[1]:.3f}, {perc[2]:.3f}]  BCa [{bca['lo']:.3f}, {bca['hi']:.3f}]")
    r = np.random.default_rng(3).normal(5.0, 1.0, 50)
    pc, _ = er.ambos(lambda a, b: b / a, r, r)
    check("un cociente identico en todas las semillas da un intervalo degenerado",
          pc[0] == pc[1] == pc[2] == 1.0, f"{pc}")
    import boot_emparejado as be
    b = np.random.default_rng(1).normal(100, 10, 40)
    rr = b + np.random.default_rng(2).normal(5, 3, 40)
    check("percentil_pareado == boot_emparejado.boot_pareado",
          er.percentil_pareado(st.delta, b, rr, seed=st.BOOT_SEED + 3)
          == be.boot_pareado(st.delta, b, rr, seed=st.BOOT_SEED + 3))

    print("\n5. kappa y s de r1_topologia.py")
    import r1_topologia as rt
    datos = rc.cargar("ortogonalidad")
    m = {c: rt.medir([r for r in datos["filas"] if r["label"] == f"{c}_iny"])
         for c in ("Ix", "Ip", "Ex", "Ep")}
    tabla4 = datos["manifiesto"]["resumen"]["eje_A"]
    check("kappa == cociente L1/L0 publicado (Tabla 4), punto",
          all(abs(m[c]["kappa"][0] - tabla4[c]["ratio"][0]) < 1e-12 for c in m))
    check("x: kappa ~ 0.99 y s ~ 1.00",
          all(abs(m[c]["kappa"][0] - 0.99) <= 0.01 and abs(m[c]["s"][0] - 1) <= 0.01
              for c in ("Ix", "Ex")),
          f"kappa {m['Ix']['kappa'][0]:.4f}  s {m['Ix']['s'][0]:.4f}")
    check("I/p (particion): kappa = 0 y s ~ 1.00",
          m["Ip"]["kappa"][0] == 0.0 and abs(m["Ip"]["s"][0] - 1) <= 0.01,
          f"s {m['Ip']['s'][0]:.4f}")
    check("E/p (exclusion): kappa = 0 y s = 0",
          m["Ep"]["kappa"][0] == 0.0 and m["Ep"]["s"][0] == 0.0)
    p = rc.RESULTS / "r1_topologia.json"
    if p.is_file():
        guardado = {f["cond"]: f for f in json.loads(p.read_text(encoding="utf-8"))["filas"]}
        check("r1_topologia.json reproducido (kappa, s, nacimientos, flujos)",
              all(guardado[c][k] == m[c][k] for c in m
                  for k in ("kappa", "s", "nacimientos") + rt.FLUJOS))
    else:
        check("r1_topologia.json existe", False, "falta: correr el experimento")

    print("\n" + "=" * 60)
    if FALLOS:
        print(f"FALLOS ({len(FALLOS)}): " + ", ".join(FALLOS))
        return 1
    print("Todos los tests de la R1 pasan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
