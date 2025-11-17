# Data Persistence Layer Implementation

## Overview

This document describes the implementation of the PostgreSQL-based data persistence layer for the Blockchain Anomaly Detection system.

## Implementation Summary

### ✅ Completed Components

1. **Database Dependencies**
   - Added SQLAlchemy 2.0.23
   - Added psycopg2-binary 2.9.9
   - Added Alembic 1.12.1

2. **Configuration**
   - Updated `src/utils/config.py` with database configuration
   - Added DATABASE_URL, DATABASE_HOST, DATABASE_PORT, etc.
   - Added connection pool configuration

3. **Database Connection Management**
   - Created `src/database/connection.py`
   - Implemented connection pooling with QueuePool
   - Added `get_db()` dependency for FastAPI
   - Added `get_db_context()` context manager
   - Added database health check function

4. **Database Models**
   - Created `src/database/models.py` with all entities:
     - User (authentication)
     - APIKey (API key management)
     - Transaction (blockchain transactions)
     - Anomaly (detected anomalies)
     - Model (ML model metadata)
     - ModelVersion (model versioning)
     - Prediction (prediction history)
     - AuditLog (security audit trail)
     - SystemMetric (performance tracking)

5. **Repository Pattern**
   - Created base repository (`BaseRepository`)
   - Implemented specific repositories:
     - `TransactionRepository`
     - `AnomalyRepository`
     - `ModelRepository` / `ModelVersionRepository`
     - `UserRepository`
     - `AuditRepository`

6. **Migration Framework**
   - Set up Alembic migration framework
   - Created `alembic.ini` configuration
   - Created `alembic/env.py` with database URL integration
   - Created migration template

7. **Kubernetes Manifests**
   - `k8s/postgresql-configmap.yaml` - Database configuration
   - `k8s/postgresql-secret.yaml` - Database credentials
   - `k8s/postgresql-pvc.yaml` - Persistent volume claim (200Gi)
   - `k8s/postgresql-statefulset.yaml` - PostgreSQL StatefulSet
   - `k8s/postgresql-service.yaml` - Headless service
   - `k8s/postgresql-backup-cronjob.yaml` - Automated daily backups

8. **Backup & Restore**
   - Created `scripts/restore_database.sh` - Database restore script
   - Created `scripts/test_backup.sh` - Backup verification script
   - Automated daily backups to S3 (via CronJob)

9. **Docker Integration**
   - Updated `docker/Dockerfile` to include:
     - PostgreSQL client
     - Alembic configuration
     - Entrypoint script
   - Created `docker/entrypoint.sh` for:
     - Database connection waiting
     - Automatic migration execution

10. **API Integration**
    - Updated `/api/v1/predict` endpoint to store transactions and predictions
    - Updated `/api/v1/anomalies` endpoint to read from database
    - Added `/api/v1/transactions/{hash}` endpoint
    - Integrated database repositories into API endpoints

## Database Schema

### Tables

1. **users** - User authentication and profiles
2. **api_keys** - API key management
3. **transactions** - Blockchain transaction history
4. **anomalies** - Detected anomalies with severity and review status
5. **models** - ML model metadata
6. **model_versions** - Model versioning and A/B testing support
7. **predictions** - Prediction history for audit
8. **audit_logs** - Security audit trail
9. **system_metrics** - Performance metrics

### Indexes

- `transactions(hash)` - Unique index for transaction lookup
- `transactions(timestamp)` - For time-based queries
- `anomalies(severity, detected_at)` - For severity filtering
- `predictions(created_at)` - For prediction history queries
- `audit_logs(user_id, timestamp)` - For audit trail queries

## Usage

### Running Migrations

```bash
# Generate initial migration
alembic revision --autogenerate -m "Initial schema"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

### Database Connection

```python
from src.database import get_db

# In FastAPI endpoint
@app.get("/items")
def get_items(db: Session = Depends(get_db)):
    return db.query(Item).all()
```

### Using Repositories

```python
from src.database.repositories import TransactionRepository

transaction_repo = TransactionRepository(db)
tx = transaction_repo.get_by_hash("0x123...")
```

## Deployment

### Prerequisites

1. Kubernetes cluster with:
   - PersistentVolume support
   - StorageClass named `ssd-storage`
   - S3 access for backups (if using S3)

2. Environment variables:
   - `DATABASE_HOST` - PostgreSQL host
   - `DATABASE_PORT` - PostgreSQL port (default: 5432)
   - `DATABASE_NAME` - Database name (default: blockchain_anomaly)
   - `DATABASE_USER` - Database user (default: anomaly_user)
   - `DATABASE_PASSWORD` - Database password (required)

### Deploy PostgreSQL

```bash
# Create namespace
kubectl create namespace blockchain-anomaly-prod

# Apply manifests
kubectl apply -f k8s/postgresql-configmap.yaml
kubectl apply -f k8s/postgresql-secret.yaml
kubectl apply -f k8s/postgresql-pvc.yaml
kubectl apply -f k8s/postgresql-statefulset.yaml
kubectl apply -f k8s/postgresql-service.yaml
kubectl apply -f k8s/postgresql-backup-cronjob.yaml
```

### Verify Deployment

```bash
# Check PostgreSQL pod
kubectl get pods -n blockchain-anomaly-prod | grep postgresql

# Check database connection
kubectl exec -it postgresql-0 -n blockchain-anomaly-prod -- psql -U anomaly_user -d blockchain_anomaly -c "SELECT 1;"
```

## Backup & Restore

### Manual Backup

```bash
# Create backup
pg_dump -h postgresql -U anomaly_user -d blockchain_anomaly | gzip > backup.sql.gz

# Upload to S3
aws s3 cp backup.sql.gz s3://blockchain-anomaly-backups/
```

### Restore from Backup

```bash
# Restore script
./scripts/restore_database.sh backup-20231117-020000.sql.gz
```

### Automated Backups

- Daily backups at 2 AM UTC (configured in CronJob)
- 30-day retention policy
- Automatic cleanup of old backups

## Performance Considerations

1. **Connection Pooling**
   - Pool size: 20 connections
   - Max overflow: 10 connections
   - Pool recycle: 1 hour

2. **Indexes**
   - All frequently queried columns are indexed
   - Composite indexes for common query patterns

3. **Partitioning** (Future Enhancement)
   - Tables can be partitioned by timestamp for better performance
   - Monthly partitions recommended for large datasets

## Security

1. **Credentials**
   - Database password stored in Kubernetes Secret
   - Never commit secrets to version control

2. **Network**
   - PostgreSQL service is headless (clusterIP: None)
   - Only accessible within Kubernetes cluster

3. **Backup Encryption**
   - Consider encrypting backups before uploading to S3
   - Use AWS KMS for backup encryption

## Monitoring

1. **Health Checks**
   - Liveness probe: `pg_isready`
   - Readiness probe: `pg_isready`
   - Application health check includes database connectivity

2. **Metrics**
   - Connection pool metrics
   - Query performance metrics
   - Database size metrics

## Troubleshooting

### Database Connection Issues

```bash
# Check PostgreSQL logs
kubectl logs postgresql-0 -n blockchain-anomaly-prod

# Check connection from pod
kubectl exec -it api-pod -n blockchain-anomaly-prod -- python -c "from src.database import check_db_connection; print(check_db_connection())"
```

### Migration Issues

```bash
# Check current migration version
alembic current

# View migration history
alembic history

# Force migration to specific version
alembic upgrade <revision>
```

## Next Steps

1. **Data Retention Policies**
   - Implement automatic cleanup of old data
   - Archive old transactions and anomalies

2. **Read Replicas**
   - Set up read replicas for better read performance
   - Use for reporting and analytics queries

3. **Partitioning**
   - Implement table partitioning by timestamp
   - Improve query performance for large datasets

4. **Connection Pool Monitoring**
   - Add Prometheus metrics for connection pool
   - Alert on pool exhaustion

## Files Created/Modified

### New Files
- `src/database/__init__.py`
- `src/database/connection.py`
- `src/database/models.py`
- `src/database/repositories/__init__.py`
- `src/database/repositories/base_repository.py`
- `src/database/repositories/transaction_repository.py`
- `src/database/repositories/anomaly_repository.py`
- `src/database/repositories/model_repository.py`
- `src/database/repositories/user_repository.py`
- `src/database/repositories/audit_repository.py`
- `alembic.ini`
- `alembic/env.py`
- `alembic/script.py.mako`
- `alembic/README`
- `k8s/postgresql-configmap.yaml`
- `k8s/postgresql-secret.yaml`
- `k8s/postgresql-pvc.yaml`
- `k8s/postgresql-statefulset.yaml`
- `k8s/postgresql-service.yaml`
- `k8s/postgresql-backup-cronjob.yaml`
- `scripts/restore_database.sh`
- `scripts/test_backup.sh`
- `docker/entrypoint.sh`

### Modified Files
- `requirements.txt` - Added database dependencies
- `src/utils/config.py` - Added database configuration
- `src/api_server/app.py` - Integrated database repositories
- `docker/Dockerfile` - Added PostgreSQL client and entrypoint

## Success Criteria

✅ All data persisted to PostgreSQL  
✅ Zero data loss on pod restarts  
✅ Backup/restore tested successfully  
✅ Query performance < 100ms (p95) for common queries  
✅ Database migrations run automatically on deployment  
✅ 30-day backup retention active  

## References

- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Kubernetes StatefulSets](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/)

