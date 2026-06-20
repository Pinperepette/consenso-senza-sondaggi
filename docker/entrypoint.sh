#!/bin/sh
set -e

echo "Attendo MongoDB ($CONSENSO_MONGO_URI)..."
until python -c "import os;from pymongo import MongoClient;MongoClient(os.environ['CONSENSO_MONGO_URI'],serverSelectionTimeoutMS=2000).admin.command('ping')" 2>/dev/null; do
  sleep 2
done
echo "MongoDB connesso (db=$CONSENSO_DB)."

# se il DB e' vuoto, avvisa come popolarlo (non blocca l'avvio)
RUNS=$(python -c "from consenso.db.client import get_db; print(get_db()['model_runs'].count_documents({}))" 2>/dev/null || echo 0)
if [ "$RUNS" = "0" ]; then
  echo "------------------------------------------------------------"
  echo " DB vuoto: nessuna stima. Popolalo una volta con:"
  echo "   docker compose exec web python scripts/bootstrap.py"
  echo "------------------------------------------------------------"
fi

exec "$@"
