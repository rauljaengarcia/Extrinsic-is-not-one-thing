#!/usr/bin/env python3
"""
Comprueba las afirmaciones que el manuscrito hace sobre la Tabla 2.

    python dominancia.py [ruta/a/main.tex]

Existe porque dos veces se coló en el texto una propiedad de la tabla que
nadie habia ejecutado: una frase decia que ninguna fila dominaba a la del
replicador y a la de los soberanos a la vez, cuando la poblacion microbiana
de vida libre domina a las dos; y una contrafactica de la Conclusion se
afirmaba sin condicion cuando solo vale sobre las funciones asentadas.  Las
propiedades de un orden parcial sobre ocho filas y seis funciones no se leen
de un vistazo.  Se ejecutan.

La tabla se PARSEA de `main.tex`, no se transcribe aqui: si la tabla cambia,
esta comprobacion cambia con ella y el desajuste sale como fallo en vez de
pasar inadvertido.

Convenio de lectura de la §2.5: una funcion servida por dos reglas se compara
por la mas debil de las dos en cada eje, por separado.

    eje de control        I > E   (mas intrinseca es mas fuerte)
    eje de intercambio    x > p   (exchangeable es mas fuerte)

La entrada de habitat de los agentes soberanos esta declarada indeterminada
(`$^{*}$` en la tabla).  Toda afirmacion se comprueba bajo AMBOS valores que
puede tomar; si una solo vale bajo uno de ellos, es que el texto afirma de
mas.

Sale con codigo 1 si alguna afirmacion deja de cumplirse.
"""
import re
import sys
from itertools import combinations
from pathlib import Path

TEX_POR_DEFECTO = (Path(__file__).resolve().parent.parent
                   / "manuscript" / "main.tex")
FUNCS = ["G", "E", "R", "M", "Q", "H"]
CTRL = {"I": 1, "E": 0}          # intrinsic > extrinsic
EXCH = {"x": 1, "p": 0}          # exchangeable > recipient-partial

FALLOS = []


def check(nombre, ok, detalle=""):
    print(f"  [{'OK ' if ok else 'FALLO'}] {nombre}" + (f"   {detalle}" if detalle else ""))
    if not ok:
        FALLOS.append(nombre)


# --------------------------------------------------------------------------
# parseo
# --------------------------------------------------------------------------

def parse_tabla(texto):
    """Extrae las filas de `tab:audit` como {nombre: {funcion: (ctrl, exch)}}.

    Devuelve tambien que celdas venian marcadas como indeterminadas.
    """
    i = texto.index(r"\label{tab:audit}")
    j = texto.index(r"\end{tabular}", i)
    filas, indeterminadas = {}, {}
    for linea in texto[i:j].splitlines():
        if "&" not in linea or r"\textbf{System}" in linea:
            continue
        celdas = [c.strip() for c in linea.replace(r"\\", "").split("&")]
        if len(celdas) != len(FUNCS) + 1:
            continue
        nombre = celdas[0]
        if not re.match(r"^[A-Za-z]", nombre):
            continue
        perfil, indet = {}, set()
        for f, celda in zip(FUNCS, celdas[1:]):
            if "$^{*}$" in celda:
                indet.add(f)
                celda = celda.replace("$^{*}$", "")
            pares = [(CTRL[r[0]], EXCH[r[2]]) for r in
                     (x.strip() for x in celda.split(","))
                     if re.fullmatch(r"[IE]/[xp]", r)]
            if not pares:
                raise ValueError(f"celda no parseada: {celda!r} en {nombre!r}")
            # la mas debil en cada eje, POR SEPARADO: es una envolvente inferior
            perfil[f] = (min(p[0] for p in pares), min(p[1] for p in pares))
        filas[nombre] = perfil
        if indet:
            indeterminadas[nombre] = indet
    if len(filas) < 2:
        raise ValueError(f"solo {len(filas)} filas parseadas; el tabular cambio de forma")
    return filas, indeterminadas


# --------------------------------------------------------------------------
# el orden parcial
# --------------------------------------------------------------------------

def compara(a, b, filas, saltar=(), override=None):
    """'A>B', 'B>A', 'A=B' o 'inc' sobre las funciones consideradas.

    `override` es una lista de (fila, funcion, valor) aplicada antes de
    comparar; las entradas de filas ajenas a este par se ignoran.
    """
    pa, pb = dict(filas[a]), dict(filas[b])
    for fila, f, val in (override or ()):
        if fila == a:
            pa[f] = val
        elif fila == b:
            pb[f] = val
    mejor_a = mejor_b = False
    for f in FUNCS:
        if f in saltar:
            continue
        for eje in (0, 1):
            if pa[f][eje] > pb[f][eje]:
                mejor_a = True
            elif pa[f][eje] < pb[f][eje]:
                mejor_b = True
    if mejor_a and mejor_b:
        return "inc"
    if mejor_a:
        return "A>B"
    if mejor_b:
        return "B>A"
    return "A=B"


def domina(a, b, filas, saltar=(), override=None):
    """a >= b en todas las funciones consideradas (dominancia debil)."""
    return compara(a, b, filas, saltar, override) in ("A>B", "A=B")


def localizar(filas, fragmento):
    coincidencias = [n for n in filas if fragmento.lower() in n.lower()]
    if len(coincidencias) != 1:
        raise ValueError(f"{fragmento!r} identifica {len(coincidencias)} filas")
    return coincidencias[0]


# --------------------------------------------------------------------------
# las cuatro afirmaciones que el texto hace hoy
# --------------------------------------------------------------------------

def main(ruta=None):
    tex = Path(ruta or TEX_POR_DEFECTO)
    filas, indet = parse_tabla(tex.read_text(encoding="utf-8"))

    repl = localizar(filas, "replicator")
    sov = localizar(filas, "Sovereign")
    micro = localizar(filas, "microbial")
    bio = [localizar(filas, "microbial"), localizar(filas, "symbiont")]
    art = [n for n in filas if n not in bio]

    print(f"\n  {tex}")
    print(f"  {'-' * 66}")
    print(f"  filas: {len(filas)} ({len(art)} artificiales, {len(bio)} biologicas)")
    for n, p in filas.items():
        marca = f"  [indeterminada en {','.join(sorted(indet[n]))}]" if n in indet else ""
        print("   ", n.ljust(36), " ".join(
            f"{f}={'I' if p[f][0] else 'E'}/{'x' if p[f][1] else 'p'}"
            for f in FUNCS) + marca)

    # la entrada de habitat de los soberanos esta declarada sin resolver: toda
    # afirmacion se comprueba bajo los dos valores que puede tomar
    if sov not in indet or indet[sov] != {"H"}:
        check("la indeterminada sigue siendo H en la fila de soberanos", False,
              f"marcadas: {indet.get(sov)}")
    VALORES_H = (("H = E/x", [(sov, "H", (0, 1))]),
                 ("H = E/p", [(sov, "H", (0, 0))]))

    # (a) §6.7: ninguna fila artificial domina a la vez al replicador y a los
    #     soberanos, bajo cada uno de los dos valores de H
    print("\n(a) ninguna fila ARTIFICIAL domina al replicador y a los soberanos a la vez")
    for etiq, ov in VALORES_H:
        culpables = [n for n in art if n not in (repl, sov)
                     and domina(n, repl, filas, (), ov)
                     and domina(n, sov, filas, (), ov)]
        check(f"con {etiq}", not culpables, f"dominan ambas: {culpables}" if culpables else "")

    # (b) §6.7: la fila de vida libre domina a cada una de las dos
    print("\n(b) la poblacion de vida libre domina a cada una de las dos")
    for etiq, ov in VALORES_H:
        check(f"vs replicador, {etiq}", domina(micro, repl, filas, (), ov),
              compara(micro, repl, filas, (), ov))
        check(f"vs soberanos,  {etiq}", domina(micro, sov, filas, (), ov),
              compara(micro, sov, filas, (), ov))

    # (c) §6.7: GA y LLM son identicas celda a celda, y son el UNICO par identico
    print("\n(c) GA y LLM identicas celda a celda, y unico par identico")
    ga, llm = localizar(filas, "genetic algorithm"), localizar(filas, "LLM")
    check("GA == LLM", filas[ga] == filas[llm])
    otros = [(a, b) for a, b in combinations(filas, 2)
             if filas[a] == filas[b] and {a, b} != {ga, llm}
             and a not in indet and b not in indet]
    check("no hay otro par identico", not otros, f"tambien identicos: {otros}" if otros else "")

    # (d) Conclusion: sin la intervencion en M, los soberanos dominan al
    #     replicador en las CINCO funciones asentadas, y no en las seis
    print("\n(d) sin la intervencion en M, los soberanos dominan al replicador")
    sin_M = [(sov, "M", (0, 1))]                      # M pasa de E/p a E/x
    check("en las cinco funciones asentadas (sin H)",
          domina(sov, repl, filas, ("H",), sin_M),
          compara(sov, repl, filas, ("H",), sin_M))
    with_p = sin_M + [(sov, "H", (0, 0))]
    check("y NO en las seis si H fuese E/p, que es por lo que el texto se acota",
          not domina(sov, repl, filas, (), with_p),
          compara(sov, repl, filas, (), with_p))

    # contexto: los tres pares de §4.7, que P3 describe explicitamente
    print("\n  (contexto) los tres pares de §4.7")
    for etiq, ov in VALORES_H:
        print(f"    -- {etiq}")
        for a, b in combinations([localizar(filas, "Avida"), repl, sov], 2):
            print(f"       {a[:30]:<32} vs {b[:30]:<32} "
                  f"{compara(a, b, filas, (), ov)}")

    print("\n" + "=" * 60)
    if FALLOS:
        print(f"FALLOS ({len(FALLOS)}): " + ", ".join(FALLOS))
        print("El texto afirma sobre la Tabla 2 algo que la tabla ya no sostiene.")
        return 1
    print("Las cuatro afirmaciones sobre la Tabla 2 se sostienen.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
