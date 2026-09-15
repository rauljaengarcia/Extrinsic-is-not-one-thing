#!/usr/bin/env python3
"""
C2 — recuperar la quinta validacion en vez de declararla inutil.

Dos movimientos, y arreglan dos cosas a la vez.

1. MEDIR V_k, la varianza en numero de descendientes, y contrastar

       N_e ~ 4N / (V_k + 2)                          [Wright 1938, Crow & Kimura]

   Si se cumple, queda EXPLICADO por que `N_e/N` cae al crecer `N`, en vez de
   haber relajado un criterio.  Una desviacion se convierte en un resultado.

2. CORRER LOS DOS ESTIMADORES DE N_e AL MISMO N.  Hasta ahora el de deriva se
   media a N = 369-1475 y el de mutacion-deriva a otro tamano, asi que no eran
   comparables.  Igualados, si concuerdan hay dos estimadores independientes que
   convergen, que es validacion mucho mas fuerte que la prueba original.

NO se modifica `ecologia.py`.  `V_k` se obtiene conduciendo la ecologia desde
fuera y contando descendientes por progenitor con `last_parents`, que es un
atributo publico.

    python analisis_c2.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import runner                                          # noqa: E402
import numpy as np                                     # noqa: E402
from ecologia import Ecologia, Config                  # noqa: E402
from runner import banner, save                        # noqa: E402
import estadistica as st                               # noqa: E402

CFG_NEUTRAL = dict(transfer_on=False, mu_base=0.0, mutate_genes=(3,),
                   g_init_sd=(0.0, 0.0, 0.0, 0.30))
T0 = 5_000                       # transitorio demografico
N_SEED = 16
ESCALA = {250.0: 30_000, 500.0: 60_000, 1000.0: 120_000}   # horizonte ~ N


# ==========================================================================
def una_corrida(R, T1, seed):
    """Devuelve V_k, N censal, generaciones, y el decaimiento de varianza.

    `V_k` se mide sobre COHORTES CERRADAS: se cuentan los descendientes de los
    individuos vivos en un instante de corte, hasta que todos han muerto.  Con
    generaciones solapadas no hay otra forma de que el conteo sea comparable
    con la teoria, que esta escrita para cohortes.
    """
    e = Ecologia(cond="Ix", seed=seed, cfg=Config(R=R, **CFG_NEUTRAL),
                 log_every=0)
    e.run(T0)
    idx0 = np.flatnonzero(e.alive)
    if idx0.size < 20:
        return None
    v0 = e.G[idx0, 3].var()
    b0, N0 = e.births, idx0.size

    # Cohorte cerrada: los vivos en el corte y sus descendientes directos.
    #
    # Hay que identificar a cada miembro por (ranura, instante de nacimiento),
    # no solo por ranura.  Las ranuras se REUTILIZAN cuando un organismo muere,
    # asi que contar por ranura sigue sumando descendientes de quien la ocupa
    # despues.  Con ese fallo salian medias de 45 y 188 hijos por individuo, y
    # la prediccion de Wright quedaba sin sentido.
    hijos = np.zeros(idx0.size, dtype=np.int64)
    ranura = idx0.copy()
    nacido = e.birth_t[idx0].copy()
    pos = {int(r): j for j, r in enumerate(ranura)}
    vivos = np.ones(idx0.size, dtype=bool)

    # La simulacion corre SIEMPRE hasta T1.  El conteo de la cohorte se detiene
    # cuando esta se extingue, pero cortar la corrida ahi seria fatal: el N_e
    # por deriva quedaria medido sobre 7-8 generaciones mientras el de
    # mutacion-deriva se mide sobre cientos, y compararlos no significaria
    # nada.  Ese error hacia parecer que los dos estimadores no convergen.
    # Tiempo generacional: edad media de los progenitores al reproducirse.
    # Se acumula sobre TODOS los eventos, no solo los de la cohorte, porque la
    # edad de un progenitor vivo es correcta se llame como se llame su ranura.
    edad_sum = 0.0
    edad_n = 0
    t_ini = e.t

    cohorte_viva = True
    for _ in range(T1):
        if not e.step():
            break
        p = e.last_parents
        if p.size:
            edad_sum += float((e.t - 1 - e.birth_t[p]).sum())
            edad_n += int(p.size)
        if cohorte_viva:
            for pi in e.last_parents.tolist():
                j = pos.get(pi)
                # solo cuenta si la ranura sigue ocupada por el MISMO organismo
                if j is not None and vivos[j] and e.birth_t[pi] == nacido[j]:
                    hijos[j] += 1
            vivos &= e.alive[ranura] & (e.birth_t[ranura] == nacido)
            if not vivos.any():
                cohorte_viva = False

    idx1 = np.flatnonzero(e.alive)
    if idx1.size < 20 or v0 <= 0:
        return None
    k = hijos.astype(np.float64)
    N = 0.5 * (N0 + idx1.size)
    gens = (e.births - b0) / N
    v1 = e.G[idx1, 3].var()
    pasos = max(e.t - t_ini, 1)
    Tgen = edad_sum / edad_n if edad_n else np.nan     # intervalo generacional
    nacim = (e.births - b0) / pasos                    # nacimientos por paso
    return dict(N=N, Vk=float(k.var()), kbar=float(k.mean()),
                gens=gens, ratio=float(v1 / v0), t_final=e.t,
                Tgen=Tgen, nacim=nacim, vida=N / nacim if nacim else np.nan)


def main():
    banner("C2(1) — V_k y la prediccion N_e ~ 4N/(V_k+2)")
    print("  `V_k` se mide sobre cohortes cerradas: los vivos en un corte, y")
    print("  sus descendientes hasta que todos han muerto.  Con generaciones")
    print("  solapadas es la unica forma de que el conteo sea comparable con")
    print("  una teoria escrita para cohortes.\n")
    print(f"  {'R':>7} {'N':>7} {'k medio':>9} {'V_k':>9} {'T gen':>7} "
          f"{'vida':>7} {'T/vida':>7} {'N_e deriva':>11} {'4N/(Vk+2)':>10} "
          f"{'Hill':>8} {'med/Hill':>9}")
    filas = []
    for R, T1 in ESCALA.items():
        rs = [una_corrida(R, T1, s) for s in range(N_SEED)]
        rs = [r for r in rs if r]
        if not rs:
            continue
        N = np.mean([r["N"] for r in rs])
        Vk = np.mean([r["Vk"] for r in rs])
        kb = np.mean([r["kbar"] for r in rs])
        g = np.mean([r["gens"] for r in rs])
        Tg = np.mean([r["Tgen"] for r in rs])
        nb = np.mean([r["nacim"] for r in rs])
        vida = np.mean([r["vida"] for r in rs])
        rat = np.array([r["ratio"] for r in rs])
        lam = -np.log(np.maximum(rat, 1e-12)) / np.array([r["gens"] for r in rs])
        Ne_der = float(np.mean(1.0 / (2.0 * lam[lam > 0])))
        Ne_teo = 4.0 * N / (Vk + 2.0)
        # Hill (1972): con generaciones solapadas el censo se sustituye por el
        # numero que entra por unidad de tiempo multiplicado por el intervalo
        # generacional.  N_e = 4 N_b T / (V_k + 2).
        Ne_hill = 4.0 * nb * Tg / (Vk + 2.0)
        filas.append(dict(R=R, N=N, Vk=Vk, kbar=kb, gens=g, Tgen=Tg,
                          nacim=nb, vida=vida, Tsobrevida=Tg / vida,
                          Ne_deriva=Ne_der, Ne_teorico=Ne_teo, Ne_hill=Ne_hill,
                          cociente=Ne_der / Ne_teo,
                          cociente_hill=Ne_der / Ne_hill if Ne_hill else np.nan))
        print(f"  {R:>7.0f} {N:>7.0f} {kb:>9.3f} {Vk:>9.3f} {Tg:>7.1f} "
              f"{vida:>7.1f} {Tg/vida:>7.3f} {Ne_der:>11.0f} {Ne_teo:>10.0f} "
              f"{Ne_hill:>8.0f} {Ne_der/Ne_hill:>9.3f}")

    if filas:
        c = np.array([f["cociente"] for f in filas])
        ch = np.array([f["cociente_hill"] for f in filas])
        Tr = np.array([f["Tsobrevida"] for f in filas])
        Vks = np.array([f["Vk"] for f in filas])
        print(f"\n  V_k {'CRECE' if Vks.max() > Vks.min()*1.15 else 'es aprox. constante'}"
              f" ({Vks.mean():.2f}): "
              f"{'explica' if Vks.max() > Vks.min()*1.15 else 'NO explica'} la caida.")
        print(f"  Con censo (generaciones discretas): medido/predicho = "
              f"{c.min():.2f}-{c.max():.2f}, y CAE con N.")
        print(f"  Con Hill (generaciones solapadas): medido/predicho = "
              f"{ch.min():.2f}-{ch.max():.2f}"
              f", dispersion {(ch.max()-ch.min())/ch.mean():.1%}.")
        print(f"  T/vida = {Tr.min():.3f}-{Tr.max():.3f}"
              f" ({'decreciente' if Tr[-1] < Tr[0] else 'creciente'} con N),"
              f" que es el factor que el censo ignora.")

    banner("C2(2) — los dos estimadores de N_e al mismo N")
    print("  Hasta ahora se median a tamanos distintos y no eran comparables.\n")
    print(f"  {'R':>7} {'N':>7} {'N_e por deriva':>16} "
          f"{'N_e por mutacion-deriva':>25} {'cociente':>10}")
    comp = []
    for R in (250.0, 500.0, 1000.0):
        # deriva: sin mutacion, varianza inicial en el marcador
        der = [una_corrida(R, ESCALA[R], s) for s in range(N_SEED)]
        der = [r for r in der if r]
        if not der:
            continue
        N = np.mean([r["N"] for r in der])
        lam = np.array([-np.log(max(r["ratio"], 1e-12)) / r["gens"] for r in der])
        Ne_der = float(np.mean(1.0 / (2.0 * lam[lam > 0])))
        # mutacion-deriva: sin varianza inicial, con entrada mutacional
        vs = []
        for s in range(N_SEED):
            e = Ecologia(cond="Ix", seed=s, log_every=0,
                         cfg=Config(R=R, transfer_on=False, mu_base=0.010,
                                    mutate_genes=(3,), g_init_sd=0.0))
            e.run(T0 + ESCALA[R])
            i = np.flatnonzero(e.alive)
            if i.size > 20:
                vs.append(e.G[i, 3].var())
        Ne_mut = float(np.mean(vs) / (2.0 * 0.010 ** 2))
        comp.append(dict(R=R, N=N, Ne_deriva=Ne_der, Ne_mutacion=Ne_mut,
                         cociente=Ne_der / Ne_mut))
        print(f"  {R:>7.0f} {N:>7.0f} {Ne_der:>16.0f} {Ne_mut:>25.0f} "
              f"{Ne_der/Ne_mut:>10.3f}")

    if comp:
        c = np.array([x["cociente"] for x in comp])
        print(f"\n  Concuerdan dentro de un factor {c.mean():.2f} "
              f"(rango {c.min():.2f}-{c.max():.2f}).")
        if 0.5 < c.mean() < 2.0:
            print("  Dos estimadores independientes que convergen al mismo N.")
        else:
            print("  NO convergen.  Se informa como discrepancia, no se ajusta.")

    save("c2_ne", filas + comp, [],
         extra={"nota": "C2: V_k, prediccion de Wright y reconciliacion de N_e",
                "n_seed": N_SEED, "T0": T0, "escala": {str(k): v for k, v in ESCALA.items()}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
