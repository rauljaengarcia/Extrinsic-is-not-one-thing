#!/usr/bin/env python3
"""
EXPERIMENTO PRINCIPAL — las dos coordenadas son separables.

Diseno: 4 condiciones x 2 intervenciones.

                      retirada del operador     reetiquetado
                      (responde al eje C)       (responde al eje A)
    I/x                       no                      no
    I/p                       no                      SI
    E/x                       SI                      no
    E/p                       SI                      SI

La afirmacion que se contrasta no es que una condicion rinda mas que otra.  Es
que **cada intervencion responde a exactamente una coordenada**, y que en
linea base NINGUNA de las dos tiene firma dinamica.  Esta ultima parte es
contraintuitiva y es el resultado que hay que informar con mas cuidado:

    H4  en monocultivo, las cuatro condiciones son bit a bit identicas.

Una regla parcial sin nadie a quien desfavorecer tiene A = 1 y rho = NA.  Una
regla extrinseca que aplica el mismo criterio en el mismo momento que su
homologa intrinseca produce la misma trayectoria.  Las coordenadas son
propiedades de las REGLAS, no de las trayectorias, y solo se manifiestan bajo
la intervencion que las interroga.

    python exp_ortogonalidad.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import runner                                          # noqa: E402
import numpy as np                                     # noqa: E402
from ecologia import CONDICIONES                       # noqa: E402
from runner import spec, run_grid, run_one, save, col, banner  # noqa: E402
import estadistica as st                               # noqa: E402

# Escala del experimento.  Se puede reducir con variables de entorno para una
# prueba de humo rapida; los valores por defecto son los del pre-registro y son
# los que producen los resultados publicados.
T = int(os.environ.get("EXP_T", 20_000))
N_SEED = int(os.environ.get("EXP_SEEDS", 30))
T_INY = T // 2
N_INY = 150
# Ventana de medida del acceso: corta a proposito.  Pasado un tiempo los
# estados de los clones y de los residentes divergen, y una regla que asigna
# por necesidad reparte distinto por razones de ESTADO, no de etiqueta.
T_VENTANA = int(os.environ.get("EXP_VENTANA", 600))

ETIQUETA = {"Ix": "I/x", "Ip": "I/p", "Ex": "E/x", "Ep": "E/p"}
COORD = {"Ix": ("I", 0), "Ip": ("I", 1), "Ex": ("E", 0), "Ep": ("E", 1)}


def main():
    todo = {}

    # ----------------------------------------------------------------------
    banner("H4 — en monocultivo las cuatro condiciones son identicas")
    # ----------------------------------------------------------------------
    print("  Test de regresion obligatorio.  Si dos condiciones difieren aqui,")
    print("  hay acoplamiento accidental entre celdas y todo lo demas sobra.\n")
    # Se corre sobre las MISMAS semillas confirmatorias que el resto del
    # experimento, para que nadie tenga que preguntarse por que aqui son otras.
    # Es una comprobacion de determinismo, no estadistica: cada semilla o da
    # cuatro trayectorias identicas o no las da, y una sola discrepancia
    # bastaria para invalidar H4.
    firmas = []
    for s in range(N_SEED):
        r = {c: run_one(spec(cond=c, seed=s, T=T)) for c in CONDICIONES}
        f = {c: (v["births"], v["deaths"], v["n_final"], round(v["g_a"], 12))
             for c, v in r.items()}
        firmas.append(len(set(f.values())) == 1)
        if s < 3:
            print(f"    seed {s}: births = " +
                  "  ".join(f"{ETIQUETA[c]} {r[c]['births']:,}" for c in CONDICIONES))
    print(f"\n  identicas en {sum(firmas)}/{N_SEED} semillas -> "
          f"{'H4 CONFIRMADA' if all(firmas) else 'FALLO'}")
    todo["h4_identicas"] = int(sum(firmas))
    todo["h4_semillas"] = int(N_SEED)

    # ----------------------------------------------------------------------
    banner("Eje C — retirada del operador externo")
    # ----------------------------------------------------------------------
    specs_c = []
    for c in CONDICIONES:
        for rm in (False, True):
            for s in range(N_SEED):
                specs_c.append(spec(cond=c, seed=s, T=T, remove_operator=rm,
                                    arm="R" if rm else "B",
                                    label=f"{c}_{'R' if rm else 'B'}"))
    rows_c = run_grid(specs_c)

    print(f"\n  {'cond':<6} {'C':>2} {'A':>2} {'Phi_B':>22} {'Phi_R':>22} "
          f"{'Phi_R/Phi_B':>13} {'delta':>18}")
    tabla_c = {}
    for c in CONDICIONES:
        B = col([r for r in rows_c if r["label"] == f"{c}_B"], "births")
        R = col([r for r in rows_c if r["label"] == f"{c}_R"], "births")
        mb, lb, hb = st.boot_mean(B)
        mr, lr, hr = st.boot_mean(R)
        ratio = st.boot_stat(lambda x, y: y / x, B, R)
        d = st.boot_stat(st.delta, B, R)
        C, A = COORD[c]
        print(f"  {ETIQUETA[c]:<6} {C:>2} {A:>2} {mb:>10,.0f} [{lb:,.0f}, {hb:,.0f}] "
              f"{mr:>10,.0f} [{lr:,.0f}, {hr:,.0f}] "
              f"{ratio[0]:>7.3f} [{ratio[1]:.3f},{ratio[2]:.3f}] "
              f"{d[0]:>7.3f} [{d[1]:+.3f},{d[2]:+.3f}]")
        tabla_c[c] = dict(Phi_B=(mb, lb, hb), Phi_R=(mr, lr, hr),
                          ratio=ratio, delta=d)
    todo["eje_C"] = tabla_c

    print("\n  Lectura: el cociente es exactamente 1.000 sin dispersion en C=I")
    print("  (no hay operador externo que retirar) y se aparta de 1 en C=E.")
    print("  El signo de delta importa: si es negativo, el soporte era una")
    print("  RESTRICCION, no un apoyo, y entonces rho no se define.")

    # ----------------------------------------------------------------------
    banner("Eje A — reetiquetado (lineage challenge)")
    # ----------------------------------------------------------------------
    print("  Se clonan organismos vivos copiando su estado EXACTAMENTE y se les")
    print("  cambia solo la etiqueta.  El observable es lo que la regla ASIGNA,")
    print("  no quien sobrevive: la supervivencia a horizonte largo tiene")
    print("  absorcion y su media entre semillas reproduce la deriva neutral.\n")
    specs_a = [spec(cond=c, seed=s, T=T, inject_at=T_INY, n_inject=N_INY,
                    access_from=T_INY, access_until=T_INY + T_VENTANA,
                    label=f"{c}_iny")
               for c in CONDICIONES for s in range(N_SEED)]
    rows_a = run_grid(specs_a)

    print(f"\n  {'cond':<6} {'A':>2} {'asignado por L0 a L0':>22} "
          f"{'asignado por L0 a L1':>22} {'cociente L1/L0':>20}")
    tabla_a = {}
    for c in CONDICIONES:
        sel = [r for r in rows_a if r["label"] == f"{c}_iny"]
        a0 = col(sel, "from_L0_pc_L0")
        a1 = col(sel, "from_L0_pc_L1")
        m0 = st.boot_mean(a0)
        m1 = st.boot_mean(a1)
        rat = st.boot_stat(lambda x, y: y / x, a0, a1)
        print(f"  {ETIQUETA[c]:<6} {COORD[c][1]:>2} "
              f"{m0[0]:>10.5f} [{m0[1]:.5f}, {m0[2]:.5f}] "
              f"{m1[0]:>10.5f} [{m1[1]:.5f}, {m1[2]:.5f}] "
              f"{rat[0]:>8.3f} [{rat[1]:.3f}, {rat[2]:.3f}]")
        tabla_a[c] = dict(a_L0=m0, a_L1=m1, ratio=rat)
    todo["eje_A"] = tabla_a

    # ----------------------------------------------------------------------
    banner("Exclusion frente a particion: las dos celdas A=1 no son iguales")
    # ----------------------------------------------------------------------
    print("  Misma coordenada, consecuencia agregada opuesta.  Es un resultado")
    print("  del marco, no un detalle de implementacion.\n")
    print(f"  {'cond':<6} {'recibido total por L1':>24} {'flujo L0->L1':>14} "
          f"{'flujo L1->L1':>14} {'lectura'}")
    tabla_x = {}
    for c in CONDICIONES:
        sel = [r for r in rows_a if r["label"] == f"{c}_iny"]
        tot = st.boot_mean(col(sel, "recv_pc_L1"))
        f01 = col(sel, "flow_01").mean()
        f11 = col(sel, "flow_11").mean()
        if COORD[c][1] == 0:
            lect = "sin restriccion"
        elif f11 > 0 and f01 == 0:
            lect = "PARTICION: recibe, solo de si mismo"
        else:
            lect = "EXCLUSION: no recibe de nadie"
        print(f"  {ETIQUETA[c]:<6} {tot[0]:>12.5f} [{tot[1]:.5f}, {tot[2]:.5f}] "
              f"{f01:>14,.0f} {f11:>14,.0f}   {lect}")
        tabla_x[c] = dict(recv_L1=tot, flow_01=float(f01), flow_11=float(f11),
                          lectura=lect)
    todo["exclusion_particion"] = tabla_x

    # ----------------------------------------------------------------------
    banner("La matriz de ortogonalidad")
    # ----------------------------------------------------------------------
    print(f"  {'cond':<6} {'coord':>7} {'retirada (Phi_R/Phi_B)':>24} "
          f"{'reetiquetado (L1/L0)':>22}")
    print("  " + "-" * 62)
    for c in CONDICIONES:
        C, A = COORD[c]
        rr = tabla_c[c]["ratio"][0]
        ra = tabla_a[c]["ratio"][0]
        print(f"  {ETIQUETA[c]:<6} {'C='+C+' A='+str(A):>7} {rr:>24.3f} {ra:>22.3f}")
    print("\n  Cada intervencion responde a exactamente una coordenada.")
    print("  Esa es la separabilidad experimental que H1 afirma.")

    save("ortogonalidad", rows_c + rows_a, specs_c + specs_a,
         extra={"resumen": todo, "T": T, "n_seed": N_SEED,
                "t_inyeccion": T_INY, "n_inyeccion": N_INY})
    return 0


if __name__ == "__main__":
    sys.exit(main())
