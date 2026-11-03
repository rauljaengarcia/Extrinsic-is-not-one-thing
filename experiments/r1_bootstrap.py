#!/usr/bin/env python3
r"""
R1.Q3 (Revisor 1): el bootstrap.  Tabla 4 emparejada y BCa como segundo estimador.

Solo LEE resultados publicados (ortogonalidad.json, brazos.json,
boot_emparejado.json) y el manuscrito; no simula nada.

1. TABLA 4 EMPAREJADA, desde ortogonalidad.json: retirada del operador
   (Phi_R/Phi_B, sobre `births`) y reetiquetado (L1/L0, sobre `from_L0_pc_*`) en
   las cuatro condiciones, con el intervalo publicado (sin emparejar) al lado.
   Las condiciones de una fila comparten semillas 0..99, asi que el estimador
   que el diseno pide es el emparejado, como en Metodos.  En las filas I se
   cuenta en cuantas semillas Phi_R y Phi_B son identicos.

2. BCa EMPAREJADO (estadistica_r1.bca_pareado: aceleracion por jackknife sobre
   semillas, sobre las mismas replicas que el percentil) para los 20 intervalos
   de delta y rho de la §6.5 y la Tabla 5 que estan en boot_emparejado.json, y
   para los de la Tabla 4.  Los datos por semilla se seleccionan con
   `boot_emparejado.phis_por_semilla`, la funcion que produjo lo publicado, y
   antes de nada se comprueba que el percentil emparejado de aqui reproduce
   boot_emparejado.json exactamente.  El guardian: donde lo publicado es NA,
   BCa tambien; donde una replica da NaN (el guardian muerde dentro de la
   replica), se descarta, como en el percentil publicado, para z0 y para los
   percentiles, y los theta_(i) NaN se descartan del jackknife; se informan
   los dos recuentos.

3. N_BOOT: se confirma en el codigo (valor por defecto de las funciones de
   bootstrap, y que los scripts publicados no lo cambian) y por reproduccion:
   los intervalos publicados, recalculados con 10 000 replicas y la semilla
   fija, salen identicos; con otro numero de replicas no saldrian.  Y se busca
   en el manuscrito si la cifra consta.

4. TABLA DE AFIRMACIONES: cada afirmacion del texto que depende de un
   intervalo, con el percentil emparejado y con BCa.

5. BCa frente a un caso de referencia: la media de una normal, donde BCa ~
   percentil ~ intervalo t.

    python experiments/r1_bootstrap.py [--tex RUTA/main.tex]

Por defecto el manuscrito es manuscript/main.tex del repositorio.
"""
import argparse
import inspect
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "experiments"))

import runner                                          # noqa: E402
import numpy as np                                     # noqa: E402
import r1_comun as rc                                  # noqa: E402
import estadistica as st                               # noqa: E402
import estadistica_r1 as er                            # noqa: E402
import boot_emparejado as be                           # noqa: E402
from runner import col, banner                         # noqa: E402

CONDS = ("Ix", "Ip", "Ex", "Ep")
ETIQ = {"Ix": "I/x", "Ip": "I/p", "Ex": "E/x", "Ep": "E/p"}
MANIP = {"manip0_transferencia": "transfer", "manip1_proveedor": "resource",
         "manip2_supresion": "suppression"}


def alineado(filas, etiqueta):
    sel = sorted((r for r in filas if r["label"] == etiqueta), key=lambda r: r["seed"])
    return sel, [r["seed"] for r in sel]


def resumen_bca(b):
    return {k: b[k] for k in ("punto", "lo", "hi", "z0", "a", "alfa_lo", "alfa_hi",
                              "replicas_nan", "jackknife_nan", "nota")}


def iv_bca(b):
    return [b["punto"], b["lo"], b["hi"]]


def igual_iv(a, b, tol=1e-12):
    for x, y in zip(a[:3], b[:3]):
        if (x != x) and (y != y):
            continue
        if not abs(x - y) <= tol:
            return False
    return True


# --------------------------------------------------------------------------
def tabla4(orto, filas_out):
    filas = orto["filas"]
    pub = orto["manifiesto"]["resumen"]
    t4 = {}
    reproducido = True
    for c in CONDS:
        fb, sb = alineado(filas, f"{c}_B")
        fr, sr = alineado(filas, f"{c}_R")
        fi, _ = alineado(filas, f"{c}_iny")
        assert sb == sr == list(range(len(fb)))
        B, R = col(fb, "births"), col(fr, "births")
        a0, a1 = col(fi, "from_L0_pc_L0"), col(fi, "from_L0_pc_L1")
        ret_p, ret_b = er.ambos(er.cociente, B, R)
        del_p, del_b = er.ambos(st.delta, B, R)
        ree_p, ree_b = er.ambos(er.cociente, a0, a1)
        # lo publicado, recalculado: el bootstrap separado de exp_ortogonalidad.py
        ret_sep = st.boot_stat(lambda x, y: y / x, B, R)
        ree_sep = st.boot_stat(lambda x, y: y / x, a0, a1)
        rep = (igual_iv(ret_sep, pub["eje_C"][c]["ratio"])
               and igual_iv(ree_sep, pub["eje_A"][c]["ratio"]))
        reproducido &= rep
        t4[c] = {
            "retirada": {"publicado": pub["eje_C"][c]["ratio"], "emparejado": rc.iv(ret_p),
                         "bca": resumen_bca(ret_b)},
            "delta_retirada": {"publicado": pub["eje_C"][c]["delta"],
                               "emparejado": rc.iv(del_p), "bca": resumen_bca(del_b)},
            "reetiquetado": {"publicado": pub["eje_A"][c]["ratio"], "emparejado": rc.iv(ree_p),
                             "bca": resumen_bca(ree_b)},
            "semillas_identicas_B_R": int(np.sum(B == R)), "n_semillas": int(B.size),
            "publicado_reproducido": bool(rep),
        }
        for est, d in (("retirada Phi_R/Phi_B", t4[c]["retirada"]),
                       ("reetiquetado L1/L0", t4[c]["reetiquetado"])):
            filas_out.append({"fuente": "Tabla 4", "cond": ETIQ[c], "estadistico": est,
                              "publicado_sin_emparejar": d["publicado"],
                              "percentil_emparejado": d["emparejado"], "bca": d["bca"]})
    return t4, reproducido


def los_20(brazos, bpub, filas_out):
    filas, man = brazos["filas"], brazos["manifiesto"]
    out, n_iv, todos_iguales = {}, 0, True
    for clave in be.MANIPULACIONES:
        out[clave] = {}
        for d in man["dosis"]:
            enc = be.phis_por_semilla(filas, clave, d)
            if not enc:
                continue
            arr, semillas, _ = enc
            b, r, s = arr["B"], arr["R"], arr["S"]
            eps = 0.05 * b.mean()
            guard = man["resumen"][clave][str(d)].get("guardian")
            fila = {}
            for est, fn, args_, semilla in (
                    ("delta", st.delta, (b, r), st.BOOT_SEED + 3),
                    ("rho", lambda x, y, z: st.rho(x, y, z, eps), (b, r, s), st.BOOT_SEED + 4)):
                perc, bca = er.ambos(fn, *args_, seed=semilla)
                ref = bpub[clave][str(d)][est]["emparejado"]
                igual = igual_iv(perc, ref)
                todos_iguales &= igual
                na = est == "rho" and bool(guard)
                fila[est] = {"emparejado": rc.iv(perc), "emparejado_publicado": ref,
                             "reproduce_publicado": igual, "NA_publicado": na,
                             "bca": None if na else resumen_bca(bca),
                             "n_semillas": len(semillas)}
                if not na:
                    n_iv += 1
                    filas_out.append({"fuente": "boot_emparejado", "manipulacion": MANIP[clave],
                                      "n_L1": d, "estadistico": est,
                                      "percentil_emparejado": rc.iv(perc),
                                      "bca": resumen_bca(bca)})
            out[clave][str(d)] = fila
    return out, n_iv, todos_iguales


def n_boot_confirmado(tex):
    """Codigo + reproduccion + manuscrito."""
    defectos = {
        "estadistica.N_BOOT": st.N_BOOT,
        "estadistica.boot_mean(n_boot=)": inspect.signature(st.boot_mean).parameters["n_boot"].default,
        "estadistica.boot_stat(n_boot=)": inspect.signature(st.boot_stat).parameters["n_boot"].default,
        "boot_emparejado.boot_pareado(n_boot=)":
            inspect.signature(be.boot_pareado).parameters["n_boot"].default,
        "boot_emparejado.boot_separado(n_boot=)":
            inspect.signature(be.boot_separado).parameters["n_boot"].default,
    }
    scripts = {}
    for nombre in ("exp_ortogonalidad.py", "exp_brazos.py", "boot_emparejado.py"):
        txt = (RAIZ / "experiments" / nombre).read_text(encoding="utf-8")
        # una llamada que fije n_boot a mano: "n_boot=<numero>" fuera de las definiciones
        fijados = re.findall(r"n_boot\s*=\s*(\d[\d_]*)", txt)
        scripts[nombre] = {"n_boot_fijado_a_mano": fijados}
    manuscrito = {"ruta": str(tex), "existe": tex.is_file()}
    if tex.is_file():
        t = tex.read_text(encoding="utf-8")
        cifras = []
        for m in re.finditer(r"(?<![\d.])10(?:\\,|,|\s|~)?000\b|10\^\{?4\}?", t):
            linea = t.count("\n", 0, m.start()) + 1
            ctx = t[max(0, m.start() - 60): m.end() + 40].replace("\n", " ")
            cifras.append({"linea": linea, "contexto": ctx})
        boot = [t.count("\n", 0, m.start()) + 1 for m in re.finditer(r"bootstrap", t)]
        frases = [m.group(0) for m in re.finditer(
            r"[^.]*\b(?:resamples?|replicates?|replications?)\b[^.]*\.", t)
            if "bootstrap" in m.group(0) or "seed" in m.group(0)]
        manuscrito.update(cifras_10000=cifras, lineas_con_bootstrap=boot,
                          frases_sobre_replicas=frases)
    return {"valores_por_defecto": defectos, "scripts": scripts, "manuscrito": manuscrito}


def afirmaciones(t4, b20):
    """Cada afirmacion del texto que depende de un intervalo."""
    def tres(clave, d, est):
        f = b20[clave][str(d)][est]
        return f["emparejado"], (iv_bca(f["bca"]) if f["bca"] else None)

    out = []

    def add(donde, texto, prueba, per, bca, pub=None):
        out.append({"donde": donde, "afirmacion": texto, "prueba": prueba,
                    "publicado_sin_emparejar": pub,
                    "con_percentil_emparejado": bool(per),
                    "con_BCa": None if bca is None else bool(bca)})

    p, b = tres("manip0_transferencia", 0, "delta")
    add("sec. 6.5 'A support that was a constraint'",
        "delta = -0.125 [-0.132, -0.118] a n_L1 = 0 (transferencia), estrictamente < 0",
        "extremo superior < 0", p[2] < 0, b[2] < 0)
    p, b = tres("manip0_transferencia", 100, "rho")
    add("sec. 6.5 'rho above one'",
        "rho = 1.66 [1.07, 3.04] a n_L1 = 100 (transferencia): el intervalo excluye 1 por debajo",
        "extremo inferior > 1", p[1] > 1, b[1] > 1)
    p0, b0 = tres("manip1_proveedor", 0, "rho")
    p2, b2 = tres("manip1_proveedor", 200, "rho")
    add("sec. 6.5 'The range of rho'",
        "recurso: rho a n_L1 = 0 [-0.10, 0.05] y a n_L1 = 200 [0.86, 1.11] son disjuntos",
        "los dos intervalos no se solapan", not er.solapan(p0, p2), not er.solapan(b0, b2))
    ps, bs = zip(*(tres(k, 200, "rho") for k in be.MANIPULACIONES))
    add("sec. 6.5 'what the data do not separate'",
        "a n_L1 = 200 los tres intervalos de rho cubren 1",
        "1 dentro de los tres", all(er.cubre(x, 1) for x in ps), all(er.cubre(x, 1) for x in bs))
    pares = [(0, 1), (0, 2), (1, 2)]
    add("sec. 6.5 'what the data do not separate'",
        "a n_L1 = 200 todos los pares de intervalos de rho se solapan",
        "los tres pares se solapan", all(er.solapan(ps[i], ps[j]) for i, j in pares),
        all(er.solapan(bs[i], bs[j]) for i, j in pares))
    for c in ("Ix", "Ip"):
        d = t4[c]["retirada"]
        add(f"Tabla 4 y sec. 6.3, fila {ETIQ[c]}",
            "retirar el operador deja C = I en exactamente 1.000, con dispersion cero entre "
            "semillas",
            f"intervalo [1, 1] ({t4[c]['semillas_identicas_B_R']}/{t4[c]['n_semillas']} "
            "semillas con Phi_R == Phi_B)",
            d["emparejado"][1] == d["emparejado"][2] == 1.0,
            d["bca"]["lo"] == d["bca"]["hi"] == 1.0,
            pub=d["publicado"][1] == d["publicado"][2] == 1.0)
    for c in ("Ex", "Ep"):
        d = t4[c]["retirada"]
        add(f"Tabla 4 y sec. 6.3, fila {ETIQ[c]}",
            "retirar el operador mueve las condiciones extrinsecas (cociente distinto de 1)",
            "1 fuera del intervalo", not er.cubre(d["emparejado"], 1),
            not er.cubre(iv_bca(d["bca"]), 1), pub=not er.cubre(d["publicado"], 1))
    for c in ("Ix", "Ex"):
        d = t4[c]["reetiquetado"]
        add(f"Tabla 4, sec. 6.3 y pie de Fig. 1c, fila {ETIQ[c]}",
            "reetiquetar deja sin cambio las condiciones exchangeable (cociente compatible "
            "con 1)",
            "1 dentro del intervalo", er.cubre(d["emparejado"], 1),
            er.cubre(iv_bca(d["bca"]), 1), pub=er.cubre(d["publicado"], 1))
    for c in ("Ip", "Ep"):
        d = t4[c]["reetiquetado"]
        add(f"Tabla 4 y sec. 6.3, fila {ETIQ[c]}",
            "reetiquetar lleva a cero exacto la asignacion al linaje reetiquetado",
            "intervalo [0, 0]", d["emparejado"][1] == d["emparejado"][2] == 0.0,
            d["bca"]["lo"] == d["bca"]["hi"] == 0.0,
            pub=d["publicado"][1] == d["publicado"][2] == 0.0)
    return out


def referencia():
    x = np.random.default_rng(12345).normal(10.0, 2.0, 100)
    perc, bca = er.ambos(lambda m: m, x)
    sem = x.std(ddof=1) / np.sqrt(x.size)
    t = er.t_ppf(0.975, x.size - 1)
    return {"muestra": "N(10, 2), n = 100, semilla 12345", "estadistico": "media",
            "t": [float(x.mean()), float(x.mean() - t * sem), float(x.mean() + t * sem)],
            "t_0975_99": t, "percentil": rc.iv(perc), "bca": resumen_bca(bca)}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=str(RAIZ / "manuscript" / "main.tex"))
    args = ap.parse_args()
    rc.preparar()
    filas_out = []

    banner("1. Tabla 4 emparejada (ortogonalidad.json)")
    orto = rc.cargar("ortogonalidad")
    t4, t4_rep = tabla4(orto, filas_out)
    print(f"  {'cond':<5} {'retirada: publicado':>24} {'emparejado':>24} {'BCa':>24}  identicas")
    for c in CONDS:
        d = t4[c]["retirada"]
        print(f"  {ETIQ[c]:<5} {rc.fmt(d['publicado']):>24} {rc.fmt(d['emparejado']):>24} "
              f"{rc.fmt(iv_bca(d['bca'])):>24}  {t4[c]['semillas_identicas_B_R']}/"
              f"{t4[c]['n_semillas']}")
    print(f"\n  {'cond':<5} {'reetiq.: publicado':>24} {'emparejado':>24} {'BCa':>24}")
    for c in CONDS:
        d = t4[c]["reetiquetado"]
        print(f"  {ETIQ[c]:<5} {rc.fmt(d['publicado']):>24} {rc.fmt(d['emparejado']):>24} "
              f"{rc.fmt(iv_bca(d['bca'])):>24}")
    print(f"\n  delta de la retirada:  " + "   ".join(
        f"{ETIQ[c]} {rc.fmt(t4[c]['delta_retirada']['emparejado'])}" for c in CONDS))
    print(f"  lo publicado, recalculado sin emparejar con 10 000 replicas: "
          f"{'identico' if t4_rep else 'DIFIERE'}")

    banner("2. BCa emparejado de los intervalos de delta y rho (boot_emparejado.json)")
    brazos = rc.cargar("brazos")
    bpub = rc.cargar("boot_emparejado")
    b20, n_iv, rep20 = los_20(brazos, bpub, filas_out)
    print(f"  percentil emparejado de aqui == boot_emparejado.json: "
          f"{'identico en todos' if rep20 else 'DIFIERE'};  intervalos no NA: {n_iv}")
    print(f"\n  {'manip.':<12} {'n_L1':>4} {'est.':<5} {'percentil emparejado':>26} "
          f"{'BCa emparejado':>26} {'z0':>7} {'a':>8} {'NaN rep/jk':>10}")
    for clave, por in b20.items():
        for d, fila in por.items():
            for est in ("delta", "rho"):
                f = fila[est]
                if f["NA_publicado"]:
                    print(f"  {MANIP[clave]:<12} {d:>4} {est:<5} {'NA (guardian)':>26}")
                    continue
                b = f["bca"]
                print(f"  {MANIP[clave]:<12} {d:>4} {est:<5} {rc.fmt(f['emparejado'], 4):>26} "
                      f"{rc.fmt(iv_bca(b), 4):>26} {b['z0']:>+7.3f} {b['a']:>+8.4f} "
                      f"{b['replicas_nan']:>5}/{b['jackknife_nan']:<4}")

    banner("3. N_BOOT")
    nb = n_boot_confirmado(Path(args.tex))
    for k, v in nb["valores_por_defecto"].items():
        print(f"  {k:<42} {v:,}")
    for k, v in nb["scripts"].items():
        print(f"  {k:<42} n_boot fijado a mano: {v['n_boot_fijado_a_mano'] or 'nunca'}")
    print(f"  reproduccion exacta con 10 000 replicas: Tabla 4 {t4_rep}; "
          f"los 20 emparejados {rep20}")
    m = nb["manuscrito"]
    print(f"  manuscrito {m['ruta']}:")
    if m["existe"]:
        print(f"    cifras 10 000 en el texto: {len(m['cifras_10000'])}")
        for x in m["cifras_10000"]:
            print(f"      linea {x['linea']}: ...{x['contexto']}...")
        print(f"    lineas que dicen 'bootstrap': {m['lineas_con_bootstrap']}")
        print(f"    frases sobre replicas/remuestras: {len(m['frases_sobre_replicas'])}")
        for f in m["frases_sobre_replicas"]:
            print(f"      '{' '.join(f.split())[:150]}'")
    else:
        print("    no existe")

    banner("4. Afirmaciones que dependen de un intervalo")
    af = afirmaciones(t4, b20)
    print(f"  {'se sostiene con':<44} {'pub.':>5} {'emp.':>5} {'BCa':>5}")
    for a in af:
        def s(v):
            return "-" if v is None else ("SI" if v else "NO")
        print(f"  {a['donde'][:44]:<44} {s(a['publicado_sin_emparejar']):>5} "
              f"{s(a['con_percentil_emparejado']):>5} {s(a['con_BCa']):>5}   {a['afirmacion']}")

    banner("5. BCa frente a un caso de referencia")
    ref = referencia()
    print(f"  {ref['muestra']}, {ref['estadistico']}:")
    print(f"    t          {rc.fmt(ref['t'], 4)}   (t_0.975,99 = {ref['t_0975_99']:.6f})")
    print(f"    percentil  {rc.fmt(ref['percentil'], 4)}")
    print(f"    BCa        {rc.fmt(iv_bca(ref['bca']), 4)}   z0 = {ref['bca']['z0']:+.4f}  "
          f"a = {ref['bca']['a']:+.5f}")

    runner.save("r1_bootstrap", filas_out, [], extra={
        "n_simulaciones": 0,
        "tabla4": t4, "tabla4_publicado_reproducido": t4_rep,
        "boot_emparejado_bca": b20, "intervalos_no_NA": n_iv,
        "boot_emparejado_reproducido": rep20,
        "n_boot": nb, "afirmaciones": af, "referencia_bca": ref,
        "bootstrap": {"n_boot": st.N_BOOT, "alpha": st.ALPHA,
                      "semillas": {"Tabla 4": st.BOOT_SEED, "delta (brazos)": st.BOOT_SEED + 3,
                                   "rho (brazos)": st.BOOT_SEED + 4},
                      "bca": "Efron 1987; z0 con empates a la mitad; a por jackknife "
                             "emparejado sobre semillas; NaN descartados"},
        "procedencia": rc.procedencia(__file__, ["ortogonalidad.json", "brazos.json",
                                                 "boot_emparejado.json"]),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
