# Phase 0: Foundation - Setup Guide

**Duration:** 1 week
**Status:** Implementation Complete
**Last Updated:** 2025-11-17

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Task 1: Production Kubernetes Cluster](#task-1-production-kubernetes-cluster)
4. [Task 2: Kubernetes Security](#task-2-kubernetes-security)
5. [Task 3: Enhanced CI/CD Pipeline](#task-3-enhanced-cicd-pipeline)
6. [Task 4: Monitoring Setup](#task-4-monitoring-setup)
7. [Task 5: Integration Test Environment](#task-5-integration-test-environment)
8. [Task 6: Load Testing](#task-6-load-testing)
9. [Verification](#verification)
10. [Troubleshooting](#troubleshooting)

---

## Overview

Phase 0 prepares the foundational infrastructure for production deployment:
- ✅ Production Kubernetes cluster with Terraform
- ✅ Network policies and security hardening
- ✅ Enhanced CI/CD with comprehensive security scanning
- ✅ Prometheus & Grafana monitoring
- ✅ Integration test environment
- ✅ Load testing infrastructure

---

## Prerequisites

### Required Tools

```bash
# Install required tools
# macOS
brew install terraform kubectl helm gcloud  # For GKE
brew install docker docker-compose
brew install python@3.10

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y terraform kubectl helm docker.io docker-compose python3.10

# Verify installations
terraform version  # Should be >= 1.5.0
kubectl version --client
helm version
docker --version
python3.10 --version
```

### Cloud Provider Setup

**For Google Cloud (GKE):**
```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize gcloud
gcloud init

# Set project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable container.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable servicenetworking.googleapis.com

# Create service account for Terraform
gcloud iam service-accounts create terraform-sa \
    --display-name="Terraform Service Account"

# Grant permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:terraform-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/container.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:terraform-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/compute.admin"

# Create and download key
gcloud iam service-accounts keys create terraform-key.json \
    --iam-account=terraform-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/terraform-key.json"
```

**For AWS (EKS) or Azure (AKS):**
See Terraform modules in `terraform/modules/` and adapt accordingly.

---

## Task 1: Production Kubernetes Cluster

### Step 1.1: Configure Terraform Variables

```bash
cd terraform/production

# Create terraform.tfvars
cat > terraform.tfvars <<EOF
project_id = "your-gcp-project-id"
region     = "us-central1"
environment = "production"

# Node pool configuration
api_node_machine_type      = "n2-standard-4"  # 4 vCPU, 16GB RAM
consumer_node_machine_type = "n2-standard-8"  # 8 vCPU, 32GB RAM
data_node_machine_type     = "n2-standard-4"  # 4 vCPU, 16GB RAM

# Security (IMPORTANT: Restrict in production)
master_authorized_networks = [
  {
    cidr_block   = "YOUR_OFFICE_IP/32"
    display_name = "Office Network"
  }
]
EOF
```

### Step 1.2: Initialize and Plan

```bash
# Initialize Terraform
terraform init

# Review the plan
terraform plan -out=tfplan

# Review the output carefully
# Expected resources:
# - 1 GKE cluster
# - 3 node pools (api, consumer, data)
# - 1 VPC network
# - 1 subnet
# - 3 Kubernetes namespaces
# - Firewall rules
# - Cloud NAT
```

### Step 1.3: Apply Configuration

```bash
# Apply the configuration
terraform apply tfplan

# This will take 10-15 minutes
# Wait for completion
```

### Step 1.4: Configure kubectl

```bash
# Get cluster credentials
gcloud container clusters get-credentials production-gke-cluster \
    --region us-central1 \
    --project YOUR_PROJECT_ID

# Verify connection
kubectl get nodes

# Expected output:
# NAME                          STATUS   ROLES    AGE   VERSION
# gke-prod-api-pool-...         Ready    <none>   1m    v1.28.x
# gke-prod-consumer-pool-...    Ready    <none>   1m    v1.28.x
# gke-prod-data-pool-...        Ready    <none>   1m    v1.28.x

# Verify namespaces
kubectl get namespaces

# Expected:
# blockchain-anomaly-prod
# monitoring
# logging
```

### Step 1.5: Verify Node Pools

```bash
# Check node pools
kubectl get nodes -L workload_type

# Verify labels
kubectl get nodes -L workload_type,environment

# Expected labels:
# - workload_type=api (3+ nodes)
# - workload_type=consumer (2+ nodes)
# - workload_type=data (2+ nodes)
```

---

## Task 2: Kubernetes Security

### Step 2.1: Apply Network Policies

```bash
# Apply network policies (zero-trust networking)
kubectl apply -f k8s/network-policies.yaml

# Verify policies
kubectl get networkpolicies -n blockchain-anomaly-prod

# Expected:
# default-deny-ingress
# default-deny-egress
# api-server-network-policy
# consumer-network-policy
# redis-network-policy
# postgresql-network-policy
```

### Step 2.2: Apply Resource Quotas

```bash
# Apply resource quotas
kubectl apply -f k8s/resource-quotas.yaml

# Verify quotas
kubectl get resourcequota -n blockchain-anomaly-prod

# Expected:
# compute-resources
# object-counts

# View quota details
kubectl describe resourcequota compute-resources -n blockchain-anomaly-prod
```

### Step 2.3: Apply Pod Security Standards

```bash
# Label namespace with Pod Security Standard
kubectl label namespace blockchain-anomaly-prod \
    pod-security.kubernetes.io/enforce=restricted \
    pod-security.kubernetes.io/audit=restricted \
    pod-security.kubernetes.io/warn=restricted

# Apply RBAC and service accounts
kubectl apply -f k8s/pod-security-standards.yaml

# Verify service accounts
kubectl get serviceaccounts -n blockchain-anomaly-prod

# Expected:
# api-service-account
# consumer-service-account

# Verify roles
kubectl get roles -n blockchain-anomaly-prod
kubectl get rolebindings -n blockchain-anomaly-prod
```

---

## Task 3: Enhanced CI/CD Pipeline

### Step 3.1: Configure GitHub Secrets

```bash
# Required secrets for enhanced security scanning:
# 1. SNYK_TOKEN (for Snyk vulnerability scanning)
# 2. GITLEAKS_LICENSE (optional, for Gitleaks)
# 3. CODECOV_TOKEN (for code coverage)

# Add secrets via GitHub UI:
# Settings → Secrets and variables → Actions → New repository secret

# Or via GitHub CLI:
gh secret set SNYK_TOKEN
# Paste your Snyk token when prompted

gh secret set CODECOV_TOKEN
# Paste your Codecov token when prompted
```

### Step 3.2: Verify CI/CD Workflows

```bash
# List workflows
ls -la .github/workflows/

# Expected files:
# - ci-cd.yml (existing, enhanced)
# - security-scan.yml (NEW - comprehensive security scanning)
# - performance-test.yml (NEW - load and performance tests)

# Verify workflow syntax
for file in .github/workflows/*.yml; do
    echo "Checking $file..."
    yamllint $file || echo "OK (yamllint not installed)"
done
```

### Step 3.3: Test Security Scanning Locally

```bash
# Install security tools
pip install safety bandit semgrep pip-audit

# Run Bandit
bandit -r src/ -f json -o bandit-results.json

# Run Safety
safety check --file requirements.txt

# Run Semgrep
semgrep --config=auto src/

# Install Trivy for container scanning
wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
sudo apt-get update
sudo apt-get install trivy

# Build and scan Docker image
docker build -f docker/Dockerfile -t blockchain-anomaly-detection:test .
trivy image blockchain-anomaly-detection:test
```

### Step 3.4: Trigger Test Workflow

```bash
# Push to trigger workflows
git add .
git commit -m "test: Trigger Phase 0 CI/CD workflows"
git push

# Monitor workflow runs
gh run list

# View specific run
gh run watch
```

---

## Task 4: Monitoring Setup

### Step 4.1: Install Prometheus & Grafana (Helm)

```bash
# Add Helm repositories
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Install kube-prometheus-stack
helm install prometheus prometheus-community/kube-prometheus-stack \
    --namespace monitoring \
    --create-namespace \
    --values monitoring/prometheus-values.yaml \
    --wait

# This installs:
# - Prometheus Operator
# - Prometheus
# - Alertmanager
# - Grafana
# - Node Exporter
# - Kube State Metrics
```

### Step 4.2: Verify Prometheus Installation

```bash
# Check pods
kubectl get pods -n monitoring

# Expected pods:
# prometheus-kube-prometheus-operator-...
# prometheus-prometheus-kube-prometheus-prometheus-0
# alertmanager-prometheus-kube-prometheus-alertmanager-0
# prometheus-grafana-...
# prometheus-kube-state-metrics-...
# prometheus-prometheus-node-exporter-...

# Port-forward Prometheus
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090 &

# Access Prometheus UI
open http://localhost:9090

# Verify targets
# Navigate to Status → Targets
# All targets should be "UP"
```

### Step 4.3: Configure Grafana

```bash
# Get Grafana admin password
kubectl get secret -n monitoring prometheus-grafana \
    -o jsonpath="{.data.admin-password}" | base64 --decode; echo

# Port-forward Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80 &

# Access Grafana
open http://localhost:3000

# Login: admin / <password from above>

# Verify datasource
# Configuration → Data Sources → Prometheus
# Should be pre-configured

# Import dashboards
# Dashboards are auto-provisioned from prometheus-values.yaml
```

### Step 4.4: Apply Alert Rules

```bash
# Apply custom alert rules
kubectl apply -f monitoring/alert-rules.yaml

# Verify PrometheusRule
kubectl get prometheusrules -n monitoring

# Expected:
# blockchain-anomaly-alert-rules

# View alert rules in Prometheus
# Navigate to Alerts in Prometheus UI
```

### Step 4.5: Configure Alertmanager (Optional)

```bash
# Edit prometheus-values.yaml to configure:
# - Email SMTP settings
# - Slack webhook URL
# - PagerDuty integration

# Update Helm release
helm upgrade prometheus prometheus-community/kube-prometheus-stack \
    --namespace monitoring \
    --values monitoring/prometheus-values.yaml
```

---

## Task 5: Integration Test Environment

### Step 5.1: Start Docker Compose Environment

```bash
# Start all services
docker-compose -f docker-compose.test.yml up -d

# Wait for services to be healthy
docker-compose -f docker-compose.test.yml ps

# Expected services (all "healthy"):
# - test-postgres
# - test-redis
# - test-zookeeper
# - test-kafka
# - test-api
# - test-prometheus
# - test-grafana

# View logs
docker-compose -f docker-compose.test.yml logs -f api
```

### Step 5.2: Verify Service Health

```bash
# Check PostgreSQL
docker-compose -f docker-compose.test.yml exec postgres \
    psql -U testuser -d testdb -c "SELECT version();"

# Check Redis
docker-compose -f docker-compose.test.yml exec redis \
    redis-cli ping
# Expected: PONG

# Check Kafka
docker-compose -f docker-compose.test.yml exec kafka \
    kafka-topics --bootstrap-server localhost:9092 --list

# Check API
curl http://localhost:8000/health
# Expected: {"status": "healthy", ...}

# Check Prometheus
curl http://localhost:9090/-/healthy
# Expected: Prometheus is Healthy.

# Check Grafana
curl http://localhost:3000/api/health
# Expected: {"database": "ok", ...}
```

### Step 5.3: Run Integration Tests

```bash
# Run integration tests against Docker Compose environment
pytest tests/test_integration.py -v

# Run with coverage
pytest tests/test_integration.py -v \
    --cov=src --cov-report=html --cov-report=term

# Expected: All tests passing
```

### Step 5.4: Test Kafka Streaming (Optional)

```bash
# Start consumer with profile
docker-compose -f docker-compose.test.yml --profile with-consumer up -d consumer

# Produce test messages
docker-compose -f docker-compose.test.yml exec kafka \
    kafka-console-producer --bootstrap-server localhost:9092 --topic blockchain-transactions <<EOF
{"hash": "0x123", "value": 100, "gas": 21000, "gasPrice": 20}
{"hash": "0x456", "value": 200, "gas": 21000, "gasPrice": 20}
EOF

# Check consumer logs
docker-compose -f docker-compose.test.yml logs consumer

# Verify anomalies endpoint
curl http://localhost:8000/api/v1/anomalies?limit=10
```

### Step 5.5: Cleanup

```bash
# Stop all services
docker-compose -f docker-compose.test.yml down -v

# Remove volumes (optional, for clean state)
docker volume prune -f
```

---

## Task 6: Load Testing

### Step 6.1: Install Locust

```bash
# Install Locust
pip install locust requests

# Verify installation
locust --version
```

### Step 6.2: Run Basic Load Test

```bash
# Start Docker Compose environment
docker-compose -f docker-compose.test.yml up -d

# Wait for API to be ready
curl http://localhost:8000/health

# Run Locust load test (headless mode)
locust -f tests/load/locustfile.py \
    --headless \
    --users 100 \
    --spawn-rate 10 \
    --run-time 2m \
    --host http://localhost:8000 \
    --html locust-report.html \
    --csv locust-stats

# Monitor results
tail -f locust-stats_stats.csv
```

### Step 6.3: View Load Test Report

```bash
# Open HTML report
open locust-report.html

# Review metrics:
# - Total requests
# - Failure rate (should be < 1%)
# - Average response time (should be < 200ms)
# - p95 response time (should be < 500ms)
# - RPS (requests per second)
```

### Step 6.4: Run Performance Benchmarks

```bash
# Install pytest-benchmark
pip install pytest-benchmark

# Run benchmarks
pytest tests/benchmarks/ --benchmark-only

# Generate histogram
pytest tests/benchmarks/ --benchmark-only \
    --benchmark-histogram=benchmark-histogram

# View results
open benchmark-histogram.svg
```

### Step 6.5: Run Memory Profiling

```bash
# Install memory-profiler
pip install memory-profiler matplotlib

# Run memory profiling
python -m memory_profiler tests/profiling/memory_profile_test.py

# Or with visualization
mprof run tests/profiling/memory_profile_test.py
mprof plot
```

---

## Verification

### Checklist

Run through this checklist to verify Phase 0 completion:

- [ ] **Kubernetes Cluster**
  - [ ] GKE cluster created and accessible
  - [ ] 3 node pools (api, consumer, data) running
  - [ ] Namespaces created (blockchain-anomaly-prod, monitoring, logging)
  - [ ] kubectl configured and working

- [ ] **Security**
  - [ ] Network policies applied (default deny)
  - [ ] Resource quotas active
  - [ ] Pod Security Standards enforced (restricted)
  - [ ] RBAC roles and service accounts configured

- [ ] **CI/CD**
  - [ ] Security scanning workflows added
  - [ ] Performance testing workflow added
  - [ ] GitHub secrets configured
  - [ ] Workflows trigger successfully

- [ ] **Monitoring**
  - [ ] Prometheus installed and scraping metrics
  - [ ] Grafana accessible with dashboards
  - [ ] Alert rules configured
  - [ ] Alertmanager configured (if applicable)

- [ ] **Integration Testing**
  - [ ] Docker Compose environment starts successfully
  - [ ] All services healthy
  - [ ] Integration tests pass
  - [ ] Kafka streaming works (if enabled)

- [ ] **Load Testing**
  - [ ] Locust installed
  - [ ] Load tests run successfully
  - [ ] Performance benchmarks complete
  - [ ] Memory profiling works

### Verification Scripts

```bash
# Run comprehensive verification
./scripts/verify-phase0.sh

# Or manually:

echo "=== Checking Kubernetes Cluster ==="
kubectl get nodes
kubectl get namespaces

echo "=== Checking Security Policies ==="
kubectl get networkpolicies -n blockchain-anomaly-prod
kubectl get resourcequota -n blockchain-anomaly-prod

echo "=== Checking Monitoring ==="
kubectl get pods -n monitoring
kubectl get prometheusrules -n monitoring

echo "=== Checking Docker Compose ==="
docker-compose -f docker-compose.test.yml ps

echo "=== Running Quick Test ==="
curl -f http://localhost:8000/health || echo "API not running"
curl -f http://localhost:9090/-/healthy || echo "Prometheus not running"

echo "=== Phase 0 Verification Complete ==="
```

---

## Troubleshooting

### Issue: Terraform Apply Fails

**Problem:** Terraform fails with permission errors.

**Solution:**
```bash
# Verify service account has required roles
gcloud projects get-iam-policy YOUR_PROJECT_ID \
    --flatten="bindings[].members" \
    --filter="bindings.members:terraform-sa@"

# Re-apply IAM bindings if needed
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:terraform-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/container.admin"
```

### Issue: kubectl Cannot Connect to Cluster

**Problem:** `kubectl` commands fail with connection errors.

**Solution:**
```bash
# Re-fetch credentials
gcloud container clusters get-credentials production-gke-cluster \
    --region us-central1 \
    --project YOUR_PROJECT_ID

# Verify context
kubectl config current-context

# Test connection
kubectl get nodes
```

### Issue: Helm Install Fails

**Problem:** Prometheus Helm chart installation fails.

**Solution:**
```bash
# Check if namespace exists
kubectl get namespace monitoring || kubectl create namespace monitoring

# Check Helm repository
helm repo update

# Try with verbose output
helm install prometheus prometheus-community/kube-prometheus-stack \
    --namespace monitoring \
    --values monitoring/prometheus-values.yaml \
    --debug --dry-run

# If still fails, check values file syntax
yamllint monitoring/prometheus-values.yaml
```

### Issue: Docker Compose Services Not Healthy

**Problem:** Services stuck in "starting" state.

**Solution:**
```bash
# Check logs
docker-compose -f docker-compose.test.yml logs

# Restart problematic service
docker-compose -f docker-compose.test.yml restart postgres

# Check resource usage
docker stats

# If still issues, rebuild
docker-compose -f docker-compose.test.yml down -v
docker-compose -f docker-compose.test.yml build --no-cache
docker-compose -f docker-compose.test.yml up -d
```

### Issue: Load Tests Failing

**Problem:** High failure rate in Locust tests.

**Solution:**
```bash
# Check API logs
docker-compose -f docker-compose.test.yml logs api

# Reduce load
locust -f tests/load/locustfile.py \
    --headless \
    --users 10 \  # Reduced from 100
    --spawn-rate 1 \  # Reduced from 10
    --run-time 1m \
    --host http://localhost:8000

# Check system resources
docker stats

# Scale API if needed
docker-compose -f docker-compose.test.yml up -d --scale api=3
```

---

## Next Steps

After completing Phase 0, proceed to:

- **Phase 1: Security & Authentication** (2 weeks)
  - Implement OAuth2/JWT authentication
  - Add API key management
  - Implement rate limiting
  - Add comprehensive audit logging

See `docs/PRODUCTION_READINESS_PLAN.md` for detailed Phase 1 instructions.

---

## Additional Resources

- **Terraform Documentation**: https://www.terraform.io/docs
- **Kubernetes Documentation**: https://kubernetes.io/docs
- **Helm Documentation**: https://helm.sh/docs
- **Prometheus Documentation**: https://prometheus.io/docs
- **Grafana Documentation**: https://grafana.com/docs
- **Locust Documentation**: https://docs.locust.io

---

**Phase 0 Complete!** ✅

You now have a production-ready Kubernetes cluster with:
- ✅ Secure networking and resource quotas
- ✅ Comprehensive CI/CD with security scanning
- ✅ Full monitoring stack (Prometheus & Grafana)
- ✅ Integration test environment
- ✅ Load testing infrastructure

Ready to proceed to Phase 1: Security & Authentication!
