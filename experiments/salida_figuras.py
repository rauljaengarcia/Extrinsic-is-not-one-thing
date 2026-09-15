#!/usr/bin/env python3
"""
Guardado de figuras en PDF vectorial ademas de PNG.

Vive FUERA de `src/` a proposito.  `runner.code_hash()` resume todo
`src/*.py`, y ese hash esta escrito en el manifiesto de cada fichero de
`resultados/`.  Meter esto en `src/estilo.py` cambiaria el hash y dejaria
todos los resultados publicados sin manifiesto valido, por un cambio que no
toca ni una linea de la simulacion.  Aqui no cuenta para el hash.

Por que PDF:

  - el manuscrito carga `\\includegraphics{fig1_ortogonalidad}` sin extension
    y sin subcarpeta, y graphicx elige el PDF; leer los cinco PNG costaba
    1,78 s por pase de compilacion, con tres pases;
  - Elsevier pide vectorial (EPS o PDF) para dibujos vectoriales, y un PDF no
    tiene resolucion que pueda quedarse por debajo de especificacion.

El PNG se mantiene: sirve para previsualizar y es lo que va al repositorio.
"""
from pathlib import Path


def guardar(fig, destino, dpi=300):
    """Escribe `destino.png` y `destino.pdf`. Devuelve los dos caminos."""
    destino = Path(destino)
    png = destino.with_suffix(".png")
    pdf = destino.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)                 # vectorial: el dpi no interviene
    return png, pdf
