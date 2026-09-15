#!/usr/bin/env python3
"""
Hashes por fichero del codigo fuente.

El manifiesto de cada resultado guarda un unico hash de todo `src/`, lo cual
tiene un inconveniente: cambiar un rotulo de figura lo altera igual que
cambiar el sustrato, y quien lea los manifiestos no puede distinguir un cambio
inocuo de uno que invalida los resultados.

Este script imprime el desglose para que esa distincion sea comprobable.
El fichero que importa es `ecologia.py`: si su hash cambia entre una validacion
y un experimento, los resultados de ese experimento no estan validados.

    python hashes.py
"""
import hashlib
import json
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
SRC = AQUI / "src"
RES = AQUI / "resultados"

#: Ficheros cuyo contenido puede alterar un resultado numerico.
COMPUTACIONALES = {"ecologia.py", "runner.py", "estadistica.py"}


def main():
    print(f"  {'fichero':<18} {'modificado':<17} {'sha256[:16]':<18} efecto")
    print("  " + "-" * 70)
    for p in sorted(SRC.glob("*.py")):
        h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        t = time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
        efecto = ("SUSTRATO: invalida resultados" if p.name == "ecologia.py"
                  else "numerico" if p.name in COMPUTACIONALES
                  else "solo presentacion")
        print(f"  {p.name:<18} {t:<17} {h:<18} {efecto}")

    print(f"\n  {'resultado':<28} {'fecha (UTC)':<22} hash de src/")
    print("  " + "-" * 70)
    for f in sorted(RES.glob("*.json")):
        try:
            m = json.loads(f.read_text(encoding="utf-8"))["manifiesto"]
        except (KeyError, json.JSONDecodeError):
            continue
        print(f"  {f.stem:<28} {m['fecha_utc']:<22} {m['code_sha256_16']}")

    eco = SRC / "ecologia.py"
    print(f"\n  El sustrato se modifico por ultima vez el "
          f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(eco.stat().st_mtime))}.")
    print("  Cualquier resultado con fecha posterior se produjo con ese sustrato.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
