# Database Migration Guide

**Project:** Blockchain Anomaly Detection System
**Last Updated:** 2025-11-17
**Phase:** Phase 2 - Data Persistence Layer

---

## Overview

This guide covers database setup, migration, and maintenance procedures for the PostgreSQL database used in the Blockchain Anomaly Detection system.

---

## Table of Contents

1. [Initial Setup](#initial-setup)
2. [Creating Migrations](#creating-migrations)
3. [Running Migrations](#running-migrations)
4. [Rollback Procedures](#rollback-procedures)
5. [Backup and Restore](#backup-and-restore)
6. [Troubleshooting](#troubleshooting)

---

## Initial Setup

### Prerequisites

- Python 3.9+ with dependencies installed
- PostgreSQL 15+ server running
- Database credentials configured in environment

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This includes:
- SQLAlchemy==2.0.23
- psycopg2-binary==2.9.9
- alembic==1.12.1

### Step 2: Configure Database Connection

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Configure the following variables:

```bash
# Database Configuration
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=blockchain_anomaly
DATABASE_USER=anomaly_user
DATABASE_PASSWORD=your_secure_password

# Optional: Override auto-constructed URL
# DATABASE_URL=postgresql://user:pass@host:port/dbname

# Connection Pool Settings
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=3600
```

### Step 3: Create Database

Connect to PostgreSQL and create the database:

```bash
psql -U postgres
```

```sql
CREATE DATABASE blockchain_anomaly;
CREATE USER anomaly_user WITH ENCRYPTED PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE blockchain_anomaly TO anomaly_user;
```

### Step 4: Verify Connection

Test the database connection:

```python
python -c "from src.database.connection import check_db_connection; print('Connected' if check_db_connection() else 'Failed')"
```

---

## Creating Migrations

### Generate Initial Migration

Create the initial migration from the database models:

```bash
# Generate migration automatically from models
alembic revision --autogenerate -m "Initial schema"
```

This will:
1. Inspect `src/database/models.py`
2. Compare with the current database state
3. Generate a migration file in `alembic/versions/`

### Manual Migration

For custom migrations (data migrations, complex schema changes):

```bash
# Create empty migration file
alembic revision -m "Description of changes"
```

Edit the generated file in `alembic/versions/` to add custom logic.

### Migration File Structure

Example migration file:

```python
"""Initial schema

Revision ID: abc123def456
Revises:
Create Date: 2025-11-17 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'abc123def456'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create tables, add columns, etc.
    op.create_table('users',
        sa.Column('id', sa.String(36), primary_key=True),
        sa.Column('email', sa.String(255), nullable=False),
        # ... more columns
    )


def downgrade():
    # Reverse the changes
    op.drop_table('users')
```

---

## Running Migrations

### Apply All Pending Migrations

```bash
# Upgrade to the latest version
alembic upgrade head
```

### Apply Specific Migration

```bash
# Upgrade to a specific revision
alembic upgrade abc123def456

# Upgrade by relative offset
alembic upgrade +1  # Upgrade by 1 version
```

### Check Migration Status

```bash
# Show current version
alembic current

# Show migration history
alembic history

# Show pending migrations
alembic history --verbose
```

### Dry Run (SQL Output)

Generate SQL without executing:

```bash
# Show SQL for all migrations
alembic upgrade head --sql

# Show SQL for next migration
alembic upgrade +1 --sql
```

---

## Rollback Procedures

### Downgrade One Version

```bash
alembic downgrade -1
```

### Downgrade to Specific Version

```bash
alembic downgrade abc123def456
```

### Rollback to Base (Empty Database)

```bash
# WARNING: This will drop all tables!
alembic downgrade base
```

### Safe Rollback Process

1. **Create Backup First:**
   ```bash
   ./scripts/test_backup.sh
   ```

2. **Test Downgrade:**
   ```bash
   # Generate SQL to review changes
   alembic downgrade -1 --sql
   ```

3. **Execute Downgrade:**
   ```bash
   alembic downgrade -1
   ```

4. **Verify State:**
   ```bash
   alembic current
   ```

---

## Backup and Restore

### Automated Backups (Production)

In Kubernetes, backups run automatically via CronJob:

```yaml
# k8s/postgresql-backup-cronjob.yaml
schedule: "0 2 * * *"  # Daily at 2 AM
```

Backups are stored in S3 with 30-day retention.

### Manual Backup

```bash
./scripts/test_backup.sh
```

This creates a compressed backup in `/tmp/`:

```bash
/tmp/test_backup_20251117-100000.sql.gz
```

### Restore from Backup

```bash
./scripts/restore_database.sh backup-20251117-020000.sql.gz
```

This will:
1. Download backup from S3
2. Scale down API pods
3. Drop and recreate database
4. Restore from backup
5. Scale up API pods

---

## Kubernetes Deployment

### Deploy Database

```bash
# Apply PostgreSQL manifests
kubectl apply -f k8s/postgresql-configmap.yaml
kubectl apply -f k8s/postgresql-secret.yaml
kubectl apply -f k8s/postgresql-pvc.yaml
kubectl apply -f k8s/postgresql-statefulset.yaml
kubectl apply -f k8s/postgresql-service.yaml
kubectl apply -f k8s/postgresql-backup-cronjob.yaml
```

### Run Migrations in Kubernetes

Option 1: Using kubectl exec

```bash
# Get the API pod name
POD=$(kubectl get pods -n blockchain-anomaly-prod -l app=api -o jsonpath='{.items[0].metadata.name}')

# Run migrations
kubectl exec -it $POD -n blockchain-anomaly-prod -- alembic upgrade head
```

Option 2: Migration Job

Create a one-time Kubernetes Job:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: database-migration
  namespace: blockchain-anomaly-prod
spec:
  template:
    spec:
      containers:
      - name: migration
        image: your-image:latest
        command: ["alembic", "upgrade", "head"]
        envFrom:
        - configMapRef:
            name: postgresql-config
        - secretRef:
            name: postgresql-secret
      restartPolicy: OnFailure
```

Apply and monitor:

```bash
kubectl apply -f migration-job.yaml
kubectl logs -f job/database-migration -n blockchain-anomaly-prod
```

---

## Troubleshooting

### Connection Refused

**Symptom:** `psycopg2.OperationalError: could not connect to server`

**Solutions:**
1. Check PostgreSQL is running:
   ```bash
   kubectl get pods -n blockchain-anomaly-prod -l app=postgresql
   ```

2. Verify service:
   ```bash
   kubectl get svc postgresql -n blockchain-anomaly-prod
   ```

3. Check credentials in secret:
   ```bash
   kubectl get secret postgresql-secret -n blockchain-anomaly-prod -o yaml
   ```

### Migration Conflicts

**Symptom:** `Multiple head revisions are present`

**Solution:**
```bash
# Merge multiple heads
alembic merge heads -m "Merge migration branches"

# Apply the merge
alembic upgrade head
```

### Database Locked

**Symptom:** `database is locked` or `deadlock detected`

**Solution:**
1. Stop API pods:
   ```bash
   kubectl scale deployment api --replicas=0 -n blockchain-anomaly-prod
   ```

2. Run migration:
   ```bash
   alembic upgrade head
   ```

3. Restart API pods:
   ```bash
   kubectl scale deployment api --replicas=3 -n blockchain-anomaly-prod
   ```

### Failed Migration

**Symptom:** Migration fails partway through

**Solution:**
1. Check the error message and fix the issue
2. Mark migration as applied if partially completed:
   ```bash
   alembic stamp head
   ```

3. Or rollback and retry:
   ```bash
   alembic downgrade -1
   # Fix the issue
   alembic upgrade head
   ```

### Out of Sync

**Symptom:** Database schema doesn't match models

**Solution:**
```bash
# Generate a new migration to sync
alembic revision --autogenerate -m "Sync database with models"

# Review the generated migration carefully
# Edit alembic/versions/xxxxx_sync_database_with_models.py if needed

# Apply the migration
alembic upgrade head
```

---

## Database Schema

### Tables

1. **users** - User authentication and profiles
2. **api_keys** - API key management
3. **transactions** - Blockchain transaction history
4. **anomalies** - Detected anomalies
5. **models** - ML model metadata
6. **model_versions** - Model versioning (supports A/B testing)
7. **predictions** - Prediction history
8. **audit_logs** - Security audit trail
9. **system_metrics** - Performance metrics

### Indexes

Optimized indexes for common queries:
- `idx_user_email_active` - User lookup
- `idx_transaction_hash` - Transaction lookup
- `idx_transaction_timestamp_value` - Time-based queries
- `idx_anomaly_severity_detected` - Anomaly filtering
- `idx_audit_user_timestamp` - Audit log queries

### Relationships

```
users
  ├── api_keys (1:N)
  ├── predictions (1:N)
  └── audit_logs (1:N)

transactions
  ├── anomalies (1:N)
  └── predictions (1:N)

models
  └── model_versions (1:N)

model_versions
  ├── anomalies (1:N)
  └── predictions (1:N)
```

---

## Best Practices

### Before Migration

1. ✅ **Backup the database**
2. ✅ **Review migration SQL** (using `--sql` flag)
3. ✅ **Test in staging environment first**
4. ✅ **Plan downtime window** (if needed)
5. ✅ **Notify team members**

### During Migration

1. ✅ **Monitor logs** for errors
2. ✅ **Keep backup handy** for quick rollback
3. ✅ **Watch database metrics** (CPU, memory, disk)
4. ✅ **Be ready to rollback** if issues arise

### After Migration

1. ✅ **Verify application functionality**
2. ✅ **Run smoke tests**
3. ✅ **Check data integrity**
4. ✅ **Monitor for errors** in production
5. ✅ **Document any issues** encountered

---

## Migration Checklist

### Development

- [ ] Generate migration: `alembic revision --autogenerate -m "Description"`
- [ ] Review generated migration file
- [ ] Test upgrade: `alembic upgrade head`
- [ ] Test downgrade: `alembic downgrade -1`
- [ ] Commit migration file to git

### Staging

- [ ] Deploy code with migration file
- [ ] Create backup
- [ ] Run migration: `alembic upgrade head`
- [ ] Verify application works
- [ ] Run integration tests
- [ ] Document any issues

### Production

- [ ] Schedule maintenance window
- [ ] Notify stakeholders
- [ ] Create backup
- [ ] Scale down application pods (optional)
- [ ] Run migration: `alembic upgrade head`
- [ ] Verify migration completed
- [ ] Scale up application pods
- [ ] Monitor for errors
- [ ] Confirm application healthy

---

## Additional Resources

- **Alembic Documentation:** https://alembic.sqlalchemy.org/
- **SQLAlchemy Documentation:** https://docs.sqlalchemy.org/
- **PostgreSQL Documentation:** https://www.postgresql.org/docs/

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review application logs
3. Check database logs
4. Consult the team documentation
5. Contact the database administrator

---

**Last Updated:** 2025-11-17
**Maintained By:** DevOps Team
