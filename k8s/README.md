# Kubernetes Deployment Guide

This directory contains Kubernetes manifests for deploying the Blockchain Anomaly Detection system with scalability enhancements.

## Features

- **Bounded Anomaly Buffer with TTL**: Prevents memory leaks in long-running deployments
- **Distributed Kafka Consumer Architecture**: Horizontal scaling with consumer groups
- **Redis-based Distributed Caching**: 40%+ cache hit rate with shared state
- **Horizontal Pod Autoscaling**: Automatic scaling based on CPU/memory
- **Load Balancing**: NGINX Ingress with session affinity and rate limiting

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Ingress (NGINX)                       │
│              Load Balancer + TLS Termination            │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┴──────────────┐
        │                            │
        ▼                            ▼
┌───────────────┐            ┌──────────────┐
│  API Service  │            │ Redis Cache  │
│   (3-10 pods) │◄──────────►│ (StatefulSet)│
└───────┬───────┘            └──────────────┘
        │
        │
        ▼
┌─────────────────┐
│ Kafka Consumer  │
│   (2-8 pods)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Kafka Cluster  │
└─────────────────┘
```

## Prerequisites

1. **Kubernetes cluster** (v1.24+)
2. **kubectl** configured
3. **Kafka cluster** (or use Strimzi operator)
4. **NGINX Ingress Controller** installed
5. **Metrics Server** for HPA
6. **cert-manager** (optional, for TLS)

## Quick Start

### 1. Create Namespace

```bash
kubectl create namespace blockchain-anomaly-detection
```

### 2. Configure Secrets

Edit `secrets.yaml` and replace placeholders with base64-encoded values:

```bash
# Redis password (optional)
echo -n "your-redis-password" | base64

# Etherscan API key
echo -n "your-etherscan-api-key" | base64

# Sentry DSN (optional)
echo -n "your-sentry-dsn" | base64
```

Or create secrets via kubectl:

```bash
kubectl create secret generic redis-secret \
  --from-literal=password='your-redis-password' \
  -n blockchain-anomaly-detection

kubectl create secret generic etherscan-secret \
  --from-literal=api_key='your-etherscan-api-key' \
  -n blockchain-anomaly-detection
```

### 3. Update ConfigMap

Edit `configmap.yaml` to configure:
- Kafka bootstrap servers
- Topic name
- Buffer sizes and TTL
- Model parameters

### 4. Deploy Components

Deploy in the following order:

```bash
# 1. Namespace and ConfigMaps
kubectl apply -f redis-statefulset.yaml  # Creates namespace
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml

# 2. Storage
kubectl apply -f pvc.yaml

# 3. Redis Cache
kubectl apply -f redis-statefulset.yaml

# 4. Services
kubectl apply -f service.yaml

# 5. Deployments
kubectl apply -f api-deployment.yaml
kubectl apply -f consumer-deployment.yaml

# 6. Autoscaling
kubectl apply -f hpa.yaml

# 7. Ingress (update domain in ingress.yaml first)
kubectl apply -f ingress.yaml
```

### 5. Verify Deployment

```bash
# Check pods
kubectl get pods -n blockchain-anomaly-detection

# Check services
kubectl get svc -n blockchain-anomaly-detection

# Check HPA
kubectl get hpa -n blockchain-anomaly-detection

# Check ingress
kubectl get ingress -n blockchain-anomaly-detection

# View logs
kubectl logs -f -l component=api -n blockchain-anomaly-detection
kubectl logs -f -l component=consumer -n blockchain-anomaly-detection
```

## Configuration

### API Replicas

Configured in `api-deployment.yaml`:
- Min: 3 replicas (via HPA)
- Max: 10 replicas (via HPA)
- Auto-scales based on CPU (70%) and Memory (80%)

### Consumer Replicas

Configured in `consumer-deployment.yaml`:
- Min: 2 replicas (via HPA)
- Max: 8 replicas (via HPA)
- Should match number of Kafka partitions for optimal distribution

### Redis Configuration

- **Memory**: 2GB with LRU eviction policy
- **Persistence**: AOF + RDB snapshots
- **Storage**: 10GB persistent volume
- **Max Connections**: 50 per pod

### Anomaly Buffer

Configured via ConfigMap:
- `anomaly_buffer_max_size`: 10,000 (default)
- `anomaly_buffer_ttl_seconds`: 3600 (1 hour)

Prevents unbounded growth and memory leaks.

### Load Balancing

NGINX Ingress with:
- **Session Affinity**: ClientIP for 3 hours
- **Rate Limiting**: 100 RPS per IP
- **Connection Limit**: 50 concurrent per IP
- **Load Algorithm**: EWMA (Exponentially Weighted Moving Average)

## Monitoring

### Health Checks

- **Liveness**: `/health/live` - Pod is alive
- **Readiness**: `/health/ready` - Pod can accept traffic
- **Startup**: Initial health check for model loading

### Metrics

Prometheus metrics exposed at:
- API: `http://pod-ip:8000/api/metrics`
- Redis: `http://pod-ip:9121/metrics`

Key metrics:
- `redis_cache_hit_rate` - Cache effectiveness
- `anomaly_buffer_size` - Current buffer utilization
- `kafka_consumer_lag` - Kafka processing lag
- `http_requests_total` - API request counts

### Accessing Metrics

```bash
# Port-forward to access metrics
kubectl port-forward -n blockchain-anomaly-detection \
  svc/anomaly-detection-api-service 8000:80

# View metrics
curl http://localhost:8000/api/metrics

# View health status
curl http://localhost:8000/health
```

## Scaling

### Manual Scaling

```bash
# Scale API
kubectl scale deployment anomaly-detection-api \
  --replicas=5 -n blockchain-anomaly-detection

# Scale Consumers
kubectl scale deployment anomaly-detection-consumer \
  --replicas=4 -n blockchain-anomaly-detection
```

### Auto Scaling

HPA automatically scales based on:
- CPU utilization
- Memory utilization
- Custom metrics (configure Prometheus Adapter)

View HPA status:

```bash
kubectl get hpa -n blockchain-anomaly-detection -w
```

## Troubleshooting

### Pods not starting

```bash
# Describe pod
kubectl describe pod <pod-name> -n blockchain-anomaly-detection

# Check events
kubectl get events -n blockchain-anomaly-detection --sort-by='.lastTimestamp'
```

### Redis connection issues

```bash
# Test Redis connectivity
kubectl run -it --rm redis-test --image=redis:7.2-alpine \
  --restart=Never -n blockchain-anomaly-detection \
  -- redis-cli -h redis-service ping
```

### Kafka consumer lag

```bash
# Check consumer logs
kubectl logs -f -l component=consumer -n blockchain-anomaly-detection

# Check HPA metrics
kubectl describe hpa anomaly-detection-consumer-hpa \
  -n blockchain-anomaly-detection
```

### Cache hit rate low

Check Redis stats:

```bash
kubectl exec -it redis-0 -n blockchain-anomaly-detection -- redis-cli INFO stats
```

Adjust cache TTLs in code or warm cache on startup.

## Production Considerations

### High Availability

1. **Redis Cluster**: Replace single Redis with Redis Cluster or Sentinel
2. **Multi-AZ**: Spread pods across availability zones
3. **PodDisruptionBudget**: Already configured to maintain availability

### Security

1. **Network Policies**: Already configured, review and adjust
2. **TLS**: Configure cert-manager and update Ingress
3. **RBAC**: Create proper service accounts with minimal permissions
4. **Secrets**: Use external secret management (Vault, AWS Secrets Manager)

### Performance

1. **Resource Limits**: Tune based on actual usage
2. **Kafka Partitions**: Should match max consumer replicas
3. **Redis Memory**: Adjust based on cache requirements
4. **Connection Pooling**: Already configured (50 connections)

### Monitoring

1. **Prometheus**: Set up Prometheus Operator
2. **Grafana**: Create dashboards for key metrics
3. **Alerting**: Configure alerts for:
   - High consumer lag
   - Low cache hit rate
   - Pod restarts
   - Memory/CPU saturation

## Cleanup

To remove all resources:

```bash
kubectl delete namespace blockchain-anomaly-detection
```

Or remove individually:

```bash
kubectl delete -f ingress.yaml
kubectl delete -f hpa.yaml
kubectl delete -f consumer-deployment.yaml
kubectl delete -f api-deployment.yaml
kubectl delete -f service.yaml
kubectl delete -f redis-statefulset.yaml
kubectl delete -f pvc.yaml
kubectl delete -f secrets.yaml
kubectl delete -f configmap.yaml
```

## Support

For issues or questions:
1. Check pod logs: `kubectl logs <pod-name>`
2. Check events: `kubectl get events`
3. Review health endpoints: `/health`
4. Check Prometheus metrics: `/api/metrics`
