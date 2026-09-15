#!/usr/bin/env python3
"""Comprueba que la biseccion del igualado converge antes de gastar 30 min."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import runner                                    # noqa: E402
from exp_brazos import calibrar_supresion, DOSIS  # noqa: E402


def main():
    for d in DOSIS:
        c = calibrar_supresion(d, iters=9)
        print(f"\n  dosis L1 = {d}")
        print(f"    nacimientos sin regla : {c['libre']:>10,.0f}")
        print(f"    objetivo (brazo B)    : {c['objetivo']:>10,.0f}")
        if c.get("vacua"):
            print("    regla vacua en monocultivo: igualado trivial")
            continue
        print(f"    {'frac':>8} {'nacimientos':>13} {'error':>8}")
        for frac, got, err in c["traza"]:
            print(f"    {frac:>8.4f} {got:>13,.0f} {err:>7.2%}")
        print(f"    -> elegido frac = {c['frac']:.4f}, error {c['err']:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
