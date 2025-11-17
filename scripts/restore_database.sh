#!/bin/bash
# Database restore script

set -e

# Configuration
BACKUP_BUCKET="s3://blockchain-anomaly-backups"
PGHOST="${PGHOST:-postgresql}"
PGPORT="${PGPORT:-5432}"
PGDATABASE="${PGDATABASE:-blockchain_anomaly}"
PGUSER="${PGUSER:-anomaly_user}"

# Parse arguments
BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
  echo "Usage: $0 <backup_file>"
  echo "Available backups:"
  aws s3 ls $BACKUP_BUCKET/
  exit 1
fi

# Confirmation
read -p "This will restore database from $BACKUP_FILE. Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
  echo "Aborted."
  exit 0
fi

# Download backup
echo "Downloading backup..."
aws s3 cp $BACKUP_BUCKET/$BACKUP_FILE /tmp/$BACKUP_FILE

# Stop API pods (to prevent writes during restore)
echo "Scaling down API pods..."
kubectl scale deployment api --replicas=0 -n blockchain-anomaly-prod

# Drop and recreate database
echo "Dropping existing database..."
psql -h $PGHOST -p $PGPORT -U $PGUSER -d postgres -c "DROP DATABASE IF EXISTS $PGDATABASE;"
psql -h $PGHOST -p $PGPORT -U $PGUSER -d postgres -c "CREATE DATABASE $PGDATABASE;"

# Restore from backup
echo "Restoring database..."
gunzip -c /tmp/$BACKUP_FILE | psql -h $PGHOST -p $PGPORT -U $PGUSER -d $PGDATABASE

# Scale up API pods
echo "Scaling up API pods..."
kubectl scale deployment api --replicas=3 -n blockchain-anomaly-prod

# Cleanup
rm /tmp/$BACKUP_FILE

echo "Restore completed successfully!"

