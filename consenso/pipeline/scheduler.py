"""Scheduler (APScheduler) per polling fonti e ri-stima periodica.

In produzione: un job notturno che ricontrolla le fonti e, se sono comparse
nuove elezioni caricate, rilancia il modello (event-driven sul flag). Qui è
fornito lo scheletro avviabile; gli intervalli sono configurabili via env.
"""
from __future__ import annotations

import os

from apscheduler.schedulers.background import BackgroundScheduler

from consenso.pipeline.autosync import sync


def nightly_job() -> None:
    """Scarica le elezioni nuove dall'archivio del Ministero e, se ne trova,
    rilancia il modello (gestito dentro autosync.sync)."""
    res = sync(rerun=True)
    print(f"[autosync] nuove: {res['n_new']}, errori: {len(res['errors'])}")


def build_scheduler() -> BackgroundScheduler:
    sched = BackgroundScheduler(timezone="Europe/Rome")
    hour = int(os.environ.get("CONSENSO_NIGHTLY_HOUR", "3"))
    sched.add_job(nightly_job, "cron", hour=hour, minute=0, id="nightly_remodel")
    return sched


if __name__ == "__main__":  # pragma: no cover
    import time

    s = build_scheduler()
    s.start()
    print("Scheduler avviato. Ctrl-C per uscire.")
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        s.shutdown()
