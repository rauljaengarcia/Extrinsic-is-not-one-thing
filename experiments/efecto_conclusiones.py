#!/usr/bin/env python3
"""
¿Cambia alguna conclusion del texto al emparejar el bootstrap?

    python efecto_conclusiones.py

Lee `results/boot_emparejado.json` (lo produce `boot_emparejado.py --json`) y
evalua con codigo las tres afirmaciones del manuscrito que dependen de la
anchura de los intervalos, no de los puntos.  Los puntos no cambian al
emparejar; solo cambian los intervalos, asi que solo estas tres pueden moverse.

Sale con codigo 1 si alguna conclusion cambia de signo entre los dos metodos.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "results" / "boot_emparejado.json"

MANIPS = ("manip0_transferencia", "manip1_proveedor", "manip2_supresion")


def solapan(a, b):
    return a[1] <= b[2] and b[1] <= a[2]


def cubre(iv, x):
    return iv[1] <= x <= iv[2]


def main():
    d = json.loads(DATOS.read_text(encoding="utf-8"))
    cambios = []

    # --- INVARIANTE, no resultado: los puntos son medias y no pueden depender
    #     del remuestreo.  Si esto falla, los dos metodos no estan mirando las
    #     mismas filas y todo lo demas sobra.
    print("\n" + "=" * 78)
    print("  INVARIANTE: las estimaciones puntuales coinciden entre metodos")
    print("=" * 78)
    malos = 0
    for manip in MANIPS:
        for dosis, fila in sorted(d[manip].items(), key=lambda kv: int(kv[0])):
            for est in ("delta", "rho"):
                sep = fila[est]["separado"][0]
                par = fila[est]["emparejado"][0]
                if sep != sep and par != par:          # ambos NaN
                    continue
                if abs(sep - par) > 1e-12:
                    malos += 1
                    print(f"  [DIFIERE] {manip} L1={dosis} {est}: "
                          f"{sep!r} vs {par!r}")
    if malos:
        print(f"  {malos} puntos NO coinciden: el emparejado no es comparable")
        return 1
    print("  OK: los puntos coinciden en los 22 casos, al ultimo digito")

    print("\n" + "=" * 78)
    print("  (a) A n_L1=200, ¿se solapan los tres intervalos de rho y cubren el 1?")
    print("=" * 78)
    for metodo in ("separado", "emparejado"):
        ivs = {m: d[m]["200"]["rho"][metodo] for m in MANIPS if "200" in d[m]}
        todos_cubren = all(cubre(v, 1.0) for v in ivs.values())
        pares = {(a, b): solapan(ivs[a], ivs[b]) for a, b in combinations(ivs, 2)}
        print(f"\n  -- {metodo}")
        for m, v in ivs.items():
            print(f"     {m:<24} {v[0]:>6.3f} [{v[1]:>6.3f},{v[2]:>6.3f}]"
                  f"   cubre 1: {'si' if cubre(v, 1.0) else 'NO'}")
        for (a, b), ok in pares.items():
            print(f"     solapan {a[5:12]:<8} y {b[5:12]:<8}: {'si' if ok else 'NO'}")
        print(f"     -> los tres cubren 1 y todos los pares se solapan: "
              f"{'SI' if todos_cubren and all(pares.values()) else 'NO'}")
        cambios.append(("a", metodo, todos_cubren and all(pares.values())))

    print("\n" + "=" * 78)
    print("  (b) ¿El rho alto de la transferencia sigue excluyendo el 1 por abajo?")
    print("=" * 78)
    for metodo in ("separado", "emparejado"):
        iv = d["manip0_transferencia"]["100"]["rho"][metodo]
        excl = iv[1] > 1.0
        print(f"  {metodo:<12} rho = {iv[0]:.3f} [{iv[1]:.3f}, {iv[2]:.3f}]"
              f"   extremo inferior {'>' if excl else '<='} 1: "
              f"{'EXCLUYE el 1' if excl else 'NO lo excluye'}")
        cambios.append(("b", metodo, excl))

    print("\n" + "=" * 78)
    print("  (c) ¿Siguen disjuntos los intervalos de n_L1=0 y n_L1=200?")
    print("      (manipulacion del recurso: es donde el texto establece el rango)")
    print("=" * 78)
    for metodo in ("separado", "emparejado"):
        lo = d["manip1_proveedor"]["0"]["rho"][metodo]
        hi = d["manip1_proveedor"]["200"]["rho"][metodo]
        disj = not solapan(lo, hi)
        print(f"  {metodo:<12} n_L1=0   {lo[0]:>6.3f} [{lo[1]:>6.3f},{lo[2]:>6.3f}]")
        print(f"  {'':12} n_L1=200 {hi[0]:>6.3f} [{hi[1]:>6.3f},{hi[2]:>6.3f}]"
              f"   -> {'DISJUNTOS' if disj else 'SE SOLAPAN'}")
        cambios.append(("c", metodo, disj))

    print("\n" + "=" * 78)
    por_pregunta = {}
    for preg, metodo, val in cambios:
        por_pregunta.setdefault(preg, {})[metodo] = val
    n_cambia = 0
    for preg, v in sorted(por_pregunta.items()):
        igual = v["separado"] == v["emparejado"]
        if not igual:
            n_cambia += 1
        print(f"  ({preg})  separado={v['separado']}  emparejado={v['emparejado']}"
              f"   -> {'SIN CAMBIO' if igual else '*** CAMBIA ***'}")
    print("=" * 78)
    if n_cambia:
        print(f"  {n_cambia} conclusion(es) cambian: hay que revisar el texto.")
        return 1
    print("  Ninguna conclusion del texto cambia al emparejar el bootstrap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
