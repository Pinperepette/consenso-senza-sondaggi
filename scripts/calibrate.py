#!/usr/bin/env python3
"""Auto-calibra i parametri di proiezione (smorzamento trend, costo del governare)
minimizzando il CRPS out-of-sample, e li salva in model_config. Letti da nowcast
e backtest. Onesto: i numeri li decide la validazione, non la mano."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from consenso.model.calibration import calibrate  # noqa: E402


def main() -> int:
    r = calibrate(num_warmup=400, num_samples=400)
    print(f"calibrato: trend_damping={r['trend_damping']} gov_cost={r['gov_cost']} "
          f"(CRPS {r['crps'] * 100:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
