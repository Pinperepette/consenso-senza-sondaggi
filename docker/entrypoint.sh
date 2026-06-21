#!/bin/sh
set -e

echo "Attendo MongoDB ($CONSENSO_MONGO_URI)..."
until python -c "import os;from pymongo import MongoClient;MongoClient(os.environ['CONSENSO_MONGO_URI'],serverSelectionTimeoutMS=2000).admin.command('ping')" 2>/dev/null; do
  sleep 2
done
echo "MongoDB connesso (db=$CONSENSO_DB)."

# se il DB e' vuoto, popolalo da solo IN BACKGROUND (non blocca l'avvio del web).
# AUTO_BOOTSTRAP=0 per disattivarlo (popolamento manuale).
RUNS=$(python -c "from consenso.db.client import get_db; print(get_db()['model_runs'].count_documents({}))" 2>/dev/null || echo 0)
if [ "$RUNS" = "0" ] && [ "${AUTO_BOOTSTRAP:-1}" != "0" ]; then
  echo "------------------------------------------------------------"
  echo " DB vuoto: avvio bootstrap automatico in background."
  echo " Scarica i dati pubblici e addestra il modello (~15-30 min)."
  echo " Progresso:  docker compose exec web cat /tmp/bootstrap.log"
  echo " Per disattivarlo: AUTO_BOOTSTRAP=0"
  echo "------------------------------------------------------------"
  # le catene restano quelle dell'ambiente (default 4): calibrazione robusta.
  nohup python scripts/bootstrap.py > /tmp/bootstrap.log 2>&1 &
elif [ "$RUNS" = "0" ]; then
  echo "------------------------------------------------------------"
  echo " DB vuoto (AUTO_BOOTSTRAP=0). Popolalo con:"
  echo "   docker compose exec web python scripts/bootstrap.py"
  echo "------------------------------------------------------------"
fi

exec "$@"
