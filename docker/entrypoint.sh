#!/bin/bash
set -e

# Wait for database
echo "Waiting for database..."
until pg_isready -h "${DATABASE_HOST:-postgresql}" -p "${DATABASE_PORT:-5432}" -U "${DATABASE_USER:-anomaly_user}"; do
  echo "Database is unavailable - sleeping"
  sleep 1
done

echo "Database is ready!"

# Run migrations
echo "Running database migrations..."
alembic upgrade head

# Start application
echo "Starting application..."
exec "$@"

