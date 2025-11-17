#!/bin/bash
# Test backup script - creates a test backup and verifies it

set -e

# Configuration
PGHOST="${PGHOST:-postgresql}"
PGPORT="${PGPORT:-5432}"
PGDATABASE="${PGDATABASE:-blockchain_anomaly}"
PGUSER="${PGUSER:-anomaly_user}"
BACKUP_FILE="/tmp/test_backup_$(date +%Y%m%d-%H%M%S).sql.gz"

echo "Creating test backup..."
pg_dump -h $PGHOST -p $PGPORT -U $PGUSER -d $PGDATABASE | gzip > $BACKUP_FILE

echo "Verifying backup file..."
if [ -f "$BACKUP_FILE" ] && [ -s "$BACKUP_FILE" ]; then
  echo "Backup file created successfully: $BACKUP_FILE"
  echo "File size: $(du -h $BACKUP_FILE | cut -f1)"
  
  # Test restore to a temporary database
  echo "Testing restore to temporary database..."
  TEMP_DB="test_restore_$(date +%s)"
  
  psql -h $PGHOST -p $PGPORT -U $PGUSER -d postgres -c "CREATE DATABASE $TEMP_DB;"
  gunzip -c $BACKUP_FILE | psql -h $PGHOST -p $PGPORT -U $PGUSER -d $TEMP_DB
  psql -h $PGHOST -p $PGPORT -U $PGUSER -d postgres -c "DROP DATABASE $TEMP_DB;"
  
  echo "Backup verification successful!"
  echo "Test backup file: $BACKUP_FILE"
else
  echo "ERROR: Backup file creation failed!"
  exit 1
fi

