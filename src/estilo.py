"""
Estilo comun de las figuras.

TODO EL TEXTO QUE APAREZCA EN UNA FIGURA VA EN INGLES: las figuras se insertan
en un manuscrito en ingles.  Los comentarios y los nombres del codigo siguen en
castellano; lo que se rotula, no.

Paleta Okabe-Ito, validada para dicromacia: todos los pares superan el suelo de
vision normal (dE >= 15 en OKLab x100) y el de deuteranopia/protanopia
(dE >= 8).  El script de validacion esta en `validar_paleta.py`.

La identidad NUNCA se codifica solo por color: cada condicion lleva ademas
trama y marcador propios, que es lo que hace legible la figura impresa en gris
y en las pruebas de imprenta de la revista.
"""
import matplotlib as mpl

# --- paleta categorica, en orden fijo. No se cicla ni se genera un quinto tono.
COLOR = {"Ix": "#0072B2", "Ip": "#D55E00", "Ex": "#009E73", "Ep": "#CC79A7"}
TRAMA = {"Ix": "", "Ip": "///", "Ex": "...", "Ep": "xxx"}
MARCA = {"Ix": "o", "Ip": "s", "Ex": "^", "Ep": "D"}
ETIQ = {"Ix": "I/x", "Ip": "I/p", "Ex": "E/x", "Ep": "E/p"}

# --- brazos
COLOR_ARM = {"B": "#0072B2", "R": "#999999", "S": "#D55E00"}
TRAMA_ARM = {"B": "", "R": "///", "S": "..."}
ETIQ_ARM = {"B": "B  baseline", "R": "R  removal", "S": "S  substitution"}

TINTA = "#222222"
TINTA_2 = "#666666"
REJILLA = "#DDDDDD"


def aplicar():
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": TINTA_2,
        "axes.linewidth": 0.8,
        "axes.labelcolor": TINTA,
        "axes.grid": False,
        "grid.color": REJILLA,
        "grid.linewidth": 0.6,
        "xtick.color": TINTA_2,
        "ytick.color": TINTA_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "text.color": TINTA,
        "hatch.linewidth": 0.7,
    })


def limpiar(ax, rejilla="y"):
    """Ejes recesivos: sin marco superior/derecho, rejilla tenue en un solo eje."""
    ax.spines[["top", "right"]].set_visible(False)
    if rejilla:
        ax.grid(axis=rejilla, zorder=0)
        ax.set_axisbelow(True)
