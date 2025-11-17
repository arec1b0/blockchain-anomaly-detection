# Production Readiness Implementation Plan

**Project:** Blockchain Anomaly Detection System
**Version:** 1.0
**Date:** 2025-11-17
**Owner:** Daniil Krizhanonovskyi
**Status:** DRAFT

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Assessment](#current-state-assessment)
3. [Production Readiness Criteria](#production-readiness-criteria)
4. [Implementation Phases](#implementation-phases)
5. [Phase 0: Foundation & Setup](#phase-0-foundation--setup)
6. [Phase 1: Security & Authentication](#phase-1-security--authentication)
7. [Phase 2: Data Persistence Layer](#phase-2-data-persistence-layer)
8. [Phase 3: ML Model Lifecycle](#phase-3-ml-model-lifecycle)
9. [Phase 4: Performance & Scalability](#phase-4-performance--scalability)
10. [Phase 5: Production Hardening](#phase-5-production-hardening)
11. [Testing Strategy](#testing-strategy)
12. [Deployment & Rollout Plan](#deployment--rollout-plan)
13. [Monitoring & Observability](#monitoring--observability)
14. [Risk Management](#risk-management)
15. [Success Metrics](#success-metrics)
16. [Timeline & Resources](#timeline--resources)
17. [Appendix](#appendix)

---

## Executive Summary

### Overview

The Blockchain Anomaly Detection system has a **solid MVP foundation** with excellent scalability and observability patterns. However, it requires **critical enhancements** in security, data persistence, and ML model lifecycle management before production deployment.

### Current Maturity: 62% Production Ready

```
┌──────────────────────────────────────────────────────────┐
│ Component              │ Current │ Target │ Gap         │
├────────────────────────┼─────────┼────────┼─────────────┤
│ API Layer              │   80%   │  95%   │ +15% (Auth) │
│ Security & Auth        │    0%   │  95%   │ +95% 🔴     │
│ Data Persistence       │   20%   │  90%   │ +70% 🔴     │
│ ML Model Lifecycle     │   40%   │  85%   │ +45% 🔴     │
│ Streaming              │   70%   │  90%   │ +20%        │
│ Observability          │   90%   │  95%   │  +5%        │
│ Error Handling         │   90%   │  95%   │  +5%        │
│ Testing & QA           │   60%   │  85%   │ +25%        │
├────────────────────────┼─────────┼────────┼─────────────┤
│ OVERALL                │   62%   │  91%   │ +29%        │
└──────────────────────────────────────────────────────────┘

🔴 = Critical Blocker
```

### Key Deliverables

1. **Security**: OAuth2 authentication, API key management, rate limiting
2. **Persistence**: PostgreSQL database with audit logging and historical data
3. **ML Lifecycle**: Full model training pipeline with versioning and A/B testing
4. **Performance**: Cache optimization, distributed consumer integration
5. **Hardening**: Chaos testing, backup/recovery, incident response

### Timeline: 8-10 Weeks

- **Phase 0**: Foundation (1 week)
- **Phase 1**: Security (2 weeks)
- **Phase 2**: Persistence (2 weeks)
- **Phase 3**: ML Lifecycle (2 weeks)
- **Phase 4**: Performance (1.5 weeks)
- **Phase 5**: Hardening (1.5 weeks)

### Resource Requirements

- **2 Backend Engineers** (Python/FastAPI)
- **1 DevOps Engineer** (Kubernetes/PostgreSQL)
- **1 ML Engineer** (Model training/deployment)
- **1 QA Engineer** (Testing/automation)

---

## Current State Assessment

### Architecture Snapshot

**Codebase:**
- 5,279 LOC across 27 Python modules
- 15 test files with comprehensive coverage
- 10 Kubernetes manifests
- 88 documented error handling patterns
- 30+ Prometheus metrics

**Technology Stack:**
- FastAPI 0.100+ (REST API)
- Kafka (streaming)
- Redis (caching)
- scikit-learn (ML)
- Kubernetes (orchestration)

### Strengths

✅ **Modular Architecture**: Clear separation of concerns
✅ **Scalability Ready**: HPA, distributed caching, stateless design
✅ **Observability**: Comprehensive metrics and health checks
✅ **Error Resilience**: 88+ error handling patterns
✅ **Documentation**: Excellent CLAUDE.md guide

### Critical Gaps

🔴 **Security**: No authentication/authorization
🔴 **Persistence**: Memory-only storage with TTL
🔴 **Model Training**: Simulated endpoint, no actual training
⚠️ **Testing**: Limited integration tests
⚠️ **Distributed Consumer**: Code exists but not wired

### Risk Assessment

| Risk | Probability | Impact | Mitigation Priority |
|------|-------------|--------|---------------------|
| Security breach (no auth) | HIGH | CRITICAL | P0 |
| Data loss (no persistence) | MEDIUM | HIGH | P0 |
| Model degradation | MEDIUM | MEDIUM | P1 |
| Performance bottleneck | LOW | MEDIUM | P1 |
| Service downtime | LOW | HIGH | P2 |

---

## Production Readiness Criteria

### Definition of Production Ready

A system is considered **production ready** when it meets the following criteria:

#### 1. Security & Compliance
- [ ] Authentication and authorization implemented
- [ ] API rate limiting active
- [ ] Secrets management via Kubernetes secrets
- [ ] Audit logging for all mutations
- [ ] OWASP Top 10 vulnerabilities addressed
- [ ] Penetration testing completed

#### 2. Data Management
- [ ] Persistent storage for all critical data
- [ ] Automated backup/restore procedures
- [ ] Data retention policies implemented
- [ ] GDPR/compliance requirements met
- [ ] Database migrations automated

#### 3. Reliability & Resilience
- [ ] 99.9% uptime SLA achievable
- [ ] Graceful degradation under load
- [ ] Circuit breakers for external dependencies
- [ ] Automatic retry logic with backoff
- [ ] Chaos testing completed

#### 4. Performance & Scalability
- [ ] Load testing completed (10K RPS target)
- [ ] Response time < 200ms (p95)
- [ ] Horizontal auto-scaling verified
- [ ] Cache hit rate > 70%
- [ ] Resource limits optimized

#### 5. Observability
- [ ] Prometheus metrics comprehensive
- [ ] Grafana dashboards deployed
- [ ] Alerting rules configured
- [ ] Distributed tracing (Jaeger/Zipkin)
- [ ] Log aggregation (ELK/Loki)

#### 6. Testing & Quality
- [ ] Unit test coverage > 85%
- [ ] Integration tests comprehensive
- [ ] E2E tests automated
- [ ] Load testing automated
- [ ] Security scanning in CI/CD

#### 7. Operations & Documentation
- [ ] Runbooks for common incidents
- [ ] On-call rotation established
- [ ] API documentation complete (OpenAPI)
- [ ] Deployment automation tested
- [ ] Rollback procedures verified

---

## Implementation Phases

### Phase Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                   │
│  Phase 0: Foundation (1w)                                        │
│    ├─ Infrastructure setup                                       │
│    ├─ CI/CD enhancements                                         │
│    └─ Testing infrastructure                                     │
│                                                                   │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Phase 1: Security & Authentication (2w) 🔴 CRITICAL            │
│    ├─ OAuth2/JWT implementation                                  │
│    ├─ API key management                                         │
│    ├─ Rate limiting                                              │
│    └─ Audit logging                                              │
│                                                                   │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Phase 2: Data Persistence Layer (2w) 🔴 CRITICAL               │
│    ├─ PostgreSQL setup                                           │
│    ├─ Database schema design                                     │
│    ├─ Migration framework                                        │
│    └─ Backup/restore automation                                  │
│                                                                   │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Phase 3: ML Model Lifecycle (2w) 🔴 CRITICAL                   │
│    ├─ Model training pipeline                                    │
│    ├─ Model versioning & registry                                │
│    ├─ A/B testing framework                                      │
│    └─ Model monitoring & drift detection                         │
│                                                                   │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Phase 4: Performance & Scalability (1.5w)                       │
│    ├─ Wire distributed Kafka consumer                            │
│    ├─ Cache optimization                                         │
│    ├─ Query optimization                                         │
│    └─ Load testing & tuning                                      │
│                                                                   │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Phase 5: Production Hardening (1.5w)                            │
│    ├─ Chaos engineering tests                                    │
│    ├─ Disaster recovery drills                                   │
│    ├─ Security penetration testing                               │
│    └─ Documentation & runbooks                                   │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘

Total Duration: 8-10 weeks
```

---

## Phase 0: Foundation & Setup

**Duration:** 1 week
**Priority:** P0 (Prerequisite)
**Team:** DevOps + Backend Engineers

### Objectives

1. Prepare infrastructure for production deployment
2. Enhance CI/CD pipeline for production workflows
3. Set up testing infrastructure

### Tasks

#### 0.1 Infrastructure Setup (2 days)

**0.1.1 Create Production Kubernetes Cluster**

```bash
# Cluster specifications
- Kubernetes version: 1.28+
- Node pools:
  - api-pool: 3 nodes (4 vCPU, 16GB RAM)
  - consumer-pool: 2 nodes (8 vCPU, 32GB RAM)
  - data-pool: 2 nodes (4 vCPU, 16GB RAM, SSD)
- Auto-scaling: Min 3, Max 15 nodes
- Multi-zone deployment for HA
```

**Files to Create:**
- `terraform/production/cluster.tf`
- `terraform/production/node_pools.tf`
- `terraform/production/networking.tf`

**0.1.2 Set Up Namespaces**

```bash
kubectl create namespace blockchain-anomaly-prod
kubectl create namespace blockchain-anomaly-staging
kubectl create namespace monitoring
kubectl create namespace logging
```

**0.1.3 Configure Network Policies**

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-network-policy
spec:
  podSelector:
    matchLabels:
      app: api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: redis
  - to:
    - podSelector:
        matchLabels:
          app: postgresql
```

**Files to Create:**
- `k8s/network-policy.yaml`
- `k8s/resource-quotas.yaml`
- `k8s/pod-security-policy.yaml`

#### 0.2 CI/CD Enhancements (2 days)

**0.2.1 Add Security Scanning**

Update `.github/workflows/ci-cd.yml`:

```yaml
security-scan:
  runs-on: ubuntu-latest
  steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Run Bandit security scan
      run: |
        pip install bandit
        bandit -r src/ -f json -o bandit-report.json

    - name: Run Safety vulnerability check
      run: |
        pip install safety
        safety check --json

    - name: Trivy container scan
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'blockchain-anomaly-detection:${{ github.sha }}'
        format: 'sarif'
        output: 'trivy-results.sarif'

    - name: Upload scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
```

**0.2.2 Add Performance Testing Stage**

```yaml
performance-test:
  runs-on: ubuntu-latest
  steps:
    - name: Run Locust load tests
      run: |
        pip install locust
        locust -f tests/load/locustfile.py --headless \
          --users 1000 --spawn-rate 100 --run-time 5m \
          --host http://localhost:8000
```

**Files to Modify:**
- `.github/workflows/ci-cd.yml` (add security + performance stages)

**Files to Create:**
- `tests/load/locustfile.py`
- `.github/workflows/security-scan.yml`

#### 0.3 Testing Infrastructure (1 day)

**0.3.1 Set Up Integration Test Environment**

```yaml
# docker-compose.test.yml
version: '3.8'
services:
  api:
    build: .
    environment:
      - KAFKA_ENABLED=true
      - REDIS_ENABLED=true
      - DATABASE_URL=postgresql://test:test@postgres:5432/testdb

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: testdb
      POSTGRES_USER: test
      POSTGRES_PASSWORD: test

  redis:
    image: redis:7-alpine

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181

  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
```

**0.3.2 Create Test Data Fixtures**

```python
# tests/fixtures/transactions.py
import pytest
import pandas as pd

@pytest.fixture
def sample_transactions():
    """Generate sample transaction data for testing."""
    return pd.DataFrame({
        'hash': [f'0x{i:064x}' for i in range(1000)],
        'value': np.random.lognormal(10, 2, 1000),
        'gas': np.random.randint(21000, 100000, 1000),
        'gasPrice': np.random.lognormal(3, 1, 1000),
        'timeStamp': pd.date_range('2024-01-01', periods=1000, freq='1min')
    })

@pytest.fixture
def anomalous_transactions():
    """Generate transactions with known anomalies."""
    normal = sample_transactions()
    # Add 10 anomalies with extreme values
    anomalies = pd.DataFrame({
        'hash': [f'0xANOM{i:060x}' for i in range(10)],
        'value': [1e10] * 10,  # Extremely high values
        'gas': [500000] * 10,
        'gasPrice': [1000] * 10,
        'timeStamp': pd.date_range('2024-01-01', periods=10, freq='1h')
    })
    return pd.concat([normal, anomalies]).reset_index(drop=True)
```

**Files to Create:**
- `docker-compose.test.yml`
- `tests/fixtures/__init__.py`
- `tests/fixtures/transactions.py`
- `tests/fixtures/models.py`
- `tests/conftest.py` (shared fixtures)

#### 0.4 Monitoring Setup (1 day)

**0.4.1 Deploy Prometheus & Grafana**

```bash
# Using Helm charts
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --set grafana.adminPassword=admin \
  --set prometheus.prometheusSpec.retention=30d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=100Gi
```

**0.4.2 Import Grafana Dashboards**

```bash
# Import pre-built dashboard
kubectl create configmap grafana-dashboard-anomaly \
  --from-file=monitoring/grafana-dashboard.json \
  -n monitoring
```

**Files to Create:**
- `monitoring/prometheus-values.yaml`
- `monitoring/grafana-values.yaml`
- `monitoring/alertmanager-config.yaml`

### Deliverables

- [ ] Production Kubernetes cluster provisioned
- [ ] CI/CD pipeline enhanced with security scanning
- [ ] Integration test environment operational
- [ ] Prometheus & Grafana deployed
- [ ] Network policies configured
- [ ] Test fixtures created

### Success Criteria

- All infrastructure provisioning automated via Terraform
- CI/CD pipeline passes with all new stages
- Integration tests run successfully in Docker Compose
- Grafana dashboards display metrics from test environment

---

## Phase 1: Security & Authentication

**Duration:** 2 weeks
**Priority:** P0 (Critical Blocker)
**Team:** Backend Engineers + Security Specialist

### Objectives

1. Implement OAuth2/JWT authentication
2. Add API key management for service-to-service calls
3. Implement rate limiting and DDoS protection
4. Add comprehensive audit logging

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Client Request                           │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  NGINX Ingress                              │
│  ├─ TLS termination                                         │
│  ├─ IP whitelisting                                         │
│  └─ Basic rate limiting (burst)                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               FastAPI Middleware Stack                      │
│                                                             │
│  1. CORS Middleware                                         │
│  2. Request Tracking Middleware                             │
│  3. Rate Limiting Middleware (application level) ← NEW     │
│  4. Authentication Middleware ← NEW                         │
│  5. Authorization Middleware ← NEW                          │
│  6. Audit Logging Middleware ← NEW                          │
│                                                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   API Endpoints                             │
│  ├─ Public: /health, /docs                                  │
│  ├─ Authenticated: /api/v1/predict                          │
│  └─ Admin: /api/v1/models/train                             │
└─────────────────────────────────────────────────────────────┘
```

### Tasks

#### 1.1 Authentication System (5 days)

**1.1.1 Implement JWT Authentication**

Create `src/auth/jwt_handler.py`:

```python
"""
JWT token generation and validation.
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

security = HTTPBearer()


class JWTHandler:
    """Handles JWT token creation and validation."""

    def __init__(self):
        self.secret_key = config.JWT_SECRET_KEY
        if not self.secret_key:
            raise ValueError("JWT_SECRET_KEY must be set in environment")

    def create_access_token(
        self,
        user_id: str,
        roles: list[str],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT access token.

        Args:
            user_id: User identifier
            roles: List of user roles (e.g., ["user", "admin"])
            expires_delta: Token expiration time

        Returns:
            Encoded JWT token
        """
        to_encode = {
            "sub": user_id,
            "roles": roles,
            "type": "access"
        }

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({"exp": expire, "iat": datetime.utcnow()})

        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=ALGORITHM)
        logger.info(f"Created access token for user {user_id}")
        return encoded_jwt

    def create_refresh_token(self, user_id: str) -> str:
        """Create a refresh token for token renewal."""
        to_encode = {
            "sub": user_id,
            "type": "refresh",
            "exp": datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
            "iat": datetime.utcnow()
        }
        return jwt.encode(to_encode, self.secret_key, algorithm=ALGORITHM)

    def decode_token(self, token: str) -> Dict[str, Any]:
        """
        Decode and validate JWT token.

        Args:
            token: JWT token string

        Returns:
            Decoded token payload

        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.JWTError as e:
            logger.error(f"JWT validation error: {e}")
            raise HTTPException(status_code=401, detail="Could not validate credentials")

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)

    def hash_password(self, password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)


# Global handler instance
jwt_handler = JWTHandler()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Dict[str, Any]:
    """
    Dependency to extract and validate current user from JWT token.

    Usage:
        @app.get("/protected")
        async def protected_route(user: dict = Depends(get_current_user)):
            return {"user_id": user["sub"]}
    """
    token = credentials.credentials
    payload = jwt_handler.decode_token(token)

    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid token type")

    return payload


async def require_role(required_roles: list[str]):
    """
    Dependency to check if user has required role.

    Usage:
        @app.post("/admin/action")
        async def admin_action(
            user: dict = Depends(get_current_user),
            _: None = Depends(require_role(["admin"]))
        ):
            return {"status": "success"}
    """
    async def role_checker(user: Dict[str, Any] = Security(get_current_user)):
        user_roles = user.get("roles", [])
        if not any(role in user_roles for role in required_roles):
            logger.warning(f"User {user['sub']} lacks required roles: {required_roles}")
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required roles: {required_roles}"
            )
        return user

    return role_checker
```

**1.1.2 Create User Management Module**

Create `src/auth/user_manager.py`:

```python
"""
User management and authentication.
"""
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

from sqlalchemy.orm import Session
from fastapi import HTTPException

from src.database.models import User, APIKey
from src.auth.jwt_handler import jwt_handler
from src.utils.logger import get_logger

logger = get_logger(__name__)


class UserManager:
    """Manages user authentication and authorization."""

    def __init__(self, db: Session):
        self.db = db

    async def create_user(
        self,
        email: str,
        password: str,
        roles: list[str] = None
    ) -> User:
        """
        Create a new user.

        Args:
            email: User email (unique)
            password: Plain text password (will be hashed)
            roles: List of roles (default: ["user"])

        Returns:
            Created user object

        Raises:
            HTTPException: If user already exists
        """
        # Check if user exists
        existing_user = self.db.query(User).filter(User.email == email).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")

        # Hash password
        hashed_password = jwt_handler.hash_password(password)

        # Create user
        user = User(
            id=str(uuid.uuid4()),
            email=email,
            hashed_password=hashed_password,
            roles=roles or ["user"],
            is_active=True,
            created_at=datetime.utcnow()
        )

        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)

        logger.info(f"Created user: {email}")
        return user

    async def authenticate_user(
        self,
        email: str,
        password: str
    ) -> Optional[User]:
        """
        Authenticate user with email and password.

        Args:
            email: User email
            password: Plain text password

        Returns:
            User object if authentication successful, None otherwise
        """
        user = self.db.query(User).filter(User.email == email).first()

        if not user:
            logger.warning(f"Authentication failed: User not found ({email})")
            return None

        if not user.is_active:
            logger.warning(f"Authentication failed: User inactive ({email})")
            return None

        if not jwt_handler.verify_password(password, user.hashed_password):
            logger.warning(f"Authentication failed: Invalid password ({email})")
            return None

        # Update last login
        user.last_login = datetime.utcnow()
        self.db.commit()

        logger.info(f"User authenticated: {email}")
        return user

    async def create_api_key(
        self,
        user_id: str,
        name: str,
        expires_days: Optional[int] = None
    ) -> APIKey:
        """
        Create API key for service-to-service authentication.

        Args:
            user_id: User ID
            name: API key name/description
            expires_days: Days until expiration (None = no expiration)

        Returns:
            Created API key object
        """
        # Generate secure random key
        key = f"sk_{uuid.uuid4().hex}"
        hashed_key = jwt_handler.hash_password(key)

        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)

        api_key = APIKey(
            id=str(uuid.uuid4()),
            user_id=user_id,
            name=name,
            hashed_key=hashed_key,
            prefix=key[:8],  # Store prefix for identification
            is_active=True,
            expires_at=expires_at,
            created_at=datetime.utcnow()
        )

        self.db.add(api_key)
        self.db.commit()
        self.db.refresh(api_key)

        logger.info(f"Created API key: {name} for user {user_id}")

        # Return key only once (it's hashed in DB)
        api_key.plain_key = key
        return api_key

    async def validate_api_key(self, key: str) -> Optional[User]:
        """
        Validate API key and return associated user.

        Args:
            key: API key string

        Returns:
            User object if valid, None otherwise
        """
        prefix = key[:8]
        api_keys = self.db.query(APIKey).filter(
            APIKey.prefix == prefix,
            APIKey.is_active == True
        ).all()

        for api_key in api_keys:
            if jwt_handler.verify_password(key, api_key.hashed_key):
                # Check expiration
                if api_key.expires_at and api_key.expires_at < datetime.utcnow():
                    logger.warning(f"API key expired: {api_key.name}")
                    return None

                # Update last used
                api_key.last_used = datetime.utcnow()
                self.db.commit()

                # Get user
                user = self.db.query(User).filter(User.id == api_key.user_id).first()
                logger.info(f"API key validated: {api_key.name}")
                return user

        logger.warning(f"Invalid API key: {prefix}...")
        return None
```

**1.1.3 Add Authentication Endpoints**

Update `src/api_server/app.py`:

```python
from src.auth.jwt_handler import jwt_handler, get_current_user, require_role
from src.auth.user_manager import UserManager
from src.api_server.models import LoginRequest, LoginResponse, RegisterRequest

@app.post("/api/v1/auth/register", response_model=SuccessResponse, status_code=201)
async def register_user(request: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user."""
    user_manager = UserManager(db)
    user = await user_manager.create_user(
        email=request.email,
        password=request.password,
        roles=["user"]
    )
    return SuccessResponse(message=f"User {user.email} registered successfully")


@app.post("/api/v1/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate user and return tokens."""
    user_manager = UserManager(db)
    user = await user_manager.authenticate_user(request.email, request.password)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = jwt_handler.create_access_token(
        user_id=user.id,
        roles=user.roles
    )
    refresh_token = jwt_handler.create_refresh_token(user_id=user.id)

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=1800  # 30 minutes
    )


@app.post("/api/v1/auth/refresh", response_model=LoginResponse)
async def refresh_token(refresh_token: str):
    """Refresh access token using refresh token."""
    payload = jwt_handler.decode_token(refresh_token)

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type")

    # Issue new access token
    access_token = jwt_handler.create_access_token(
        user_id=payload["sub"],
        roles=payload.get("roles", ["user"])
    )

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=1800
    )


# Protect existing endpoints
@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(
    transaction: TransactionData,
    user: dict = Depends(get_current_user)  # ← Add authentication
):
    """Predict if a transaction is anomalous (requires authentication)."""
    # Existing logic...
    pass


@app.post("/api/v1/models/train", response_model=ModelTrainingResponse)
async def train_model(
    request: ModelTrainingRequest,
    user: dict = Depends(get_current_user),
    _: None = Depends(require_role(["admin"]))  # ← Admin only
):
    """Train a new model (admin only)."""
    # Training logic...
    pass
```

**Files to Create:**
- `src/auth/__init__.py`
- `src/auth/jwt_handler.py`
- `src/auth/user_manager.py`
- `src/auth/api_key_manager.py`

**Files to Modify:**
- `src/api_server/app.py` (add auth endpoints, protect existing)
- `src/api_server/models.py` (add auth request/response models)
- `src/utils/config.py` (add JWT_SECRET_KEY)

#### 1.2 Rate Limiting (2 days)

**1.2.1 Implement Rate Limiting Middleware**

Create `src/middleware/rate_limiter.py`:

```python
"""
Rate limiting middleware for API protection.
"""
import time
from typing import Dict, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
import asyncio

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter

from src.cache.redis_client import get_redis_client
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Prometheus metrics
rate_limit_hits = Counter(
    'rate_limit_hits_total',
    'Total number of rate limit hits.',
    ['endpoint', 'limit_type']
)

rate_limit_exceeded = Counter(
    'rate_limit_exceeded_total',
    'Total number of requests that exceeded rate limits.',
    ['endpoint', 'client_ip']
)


class RateLimiter:
    """
    Token bucket rate limiter with Redis backend.

    Supports multiple rate limit tiers:
    - Global: 10,000 RPS across all endpoints
    - Per-endpoint: Varies by endpoint
    - Per-user: Based on user tier
    - Per-IP: 100 RPS per IP address
    """

    def __init__(self):
        self.redis_client = get_redis_client()

        # Rate limit configurations (requests per window)
        self.limits = {
            "global": {"requests": 10000, "window": 60},  # 10K per minute
            "per_ip": {"requests": 100, "window": 60},     # 100 per minute per IP
            "predict": {"requests": 500, "window": 60},    # 500 predictions/min
            "batch_predict": {"requests": 50, "window": 60},  # 50 batch/min
            "train": {"requests": 5, "window": 3600},      # 5 training jobs/hour
        }

    async def check_rate_limit(
        self,
        key: str,
        limit_type: str,
        endpoint: str
    ) -> Tuple[bool, Dict]:
        """
        Check if request is within rate limit.

        Args:
            key: Unique identifier (IP, user_id, etc.)
            limit_type: Type of limit ("global", "per_ip", etc.)
            endpoint: API endpoint path

        Returns:
            Tuple of (is_allowed, info_dict)
            info_dict contains: remaining, reset_time, limit
        """
        config = self.limits.get(limit_type, self.limits["per_ip"])
        max_requests = config["requests"]
        window_seconds = config["window"]

        # Redis key: rate_limit:{type}:{key}:{window}
        now = int(time.time())
        window_key = f"rate_limit:{limit_type}:{key}:{now // window_seconds}"

        try:
            # Increment counter
            current = await self.redis_client.incr(window_key)

            # Set expiration on first request
            if current == 1:
                await self.redis_client.expire(window_key, window_seconds)

            # Calculate remaining and reset time
            remaining = max(0, max_requests - current)
            reset_time = ((now // window_seconds) + 1) * window_seconds

            # Check if limit exceeded
            is_allowed = current <= max_requests

            info = {
                "limit": max_requests,
                "remaining": remaining,
                "reset": reset_time,
                "retry_after": reset_time - now if not is_allowed else 0
            }

            # Update metrics
            rate_limit_hits.labels(endpoint=endpoint, limit_type=limit_type).inc()

            if not is_allowed:
                rate_limit_exceeded.labels(endpoint=endpoint, client_ip=key).inc()
                logger.warning(
                    f"Rate limit exceeded: {limit_type} for {key} on {endpoint}"
                )

            return is_allowed, info

        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            # Fail open on errors (allow request)
            return True, {
                "limit": max_requests,
                "remaining": max_requests,
                "reset": now + window_seconds,
                "retry_after": 0
            }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(self, app):
        super().__init__(app)
        self.rate_limiter = RateLimiter()

    async def dispatch(self, request: Request, call_next):
        """Process request and check rate limits."""

        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/health/ready", "/health/live"]:
            return await call_next(request)

        # Extract client identifier
        client_ip = request.client.host
        user_id = None

        # Get user_id from auth if available
        auth_header = request.headers.get("Authorization")
        if auth_header:
            try:
                # Extract user from token (simplified)
                from src.auth.jwt_handler import jwt_handler
                token = auth_header.replace("Bearer ", "")
                payload = jwt_handler.decode_token(token)
                user_id = payload.get("sub")
            except:
                pass

        # Determine endpoint type
        path = request.url.path
        if "/predict/batch" in path:
            endpoint_type = "batch_predict"
        elif "/predict" in path:
            endpoint_type = "predict"
        elif "/train" in path:
            endpoint_type = "train"
        else:
            endpoint_type = "default"

        # Check limits in order of specificity
        checks = [
            ("global", "global", path),
            ("per_ip", client_ip, path),
        ]

        if endpoint_type != "default":
            checks.append((endpoint_type, user_id or client_ip, path))

        for limit_type, key, endpoint in checks:
            is_allowed, info = await self.rate_limiter.check_rate_limit(
                key, limit_type, endpoint
            )

            if not is_allowed:
                logger.warning(
                    f"Rate limit exceeded: {limit_type} for "
                    f"{key} on {endpoint}. Retry after {info['retry_after']}s"
                )
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "limit": info["limit"],
                        "retry_after": info["retry_after"],
                        "reset": info["reset"]
                    },
                    headers={
                        "X-RateLimit-Limit": str(info["limit"]),
                        "X-RateLimit-Remaining": str(info["remaining"]),
                        "X-RateLimit-Reset": str(info["reset"]),
                        "Retry-After": str(info["retry_after"])
                    }
                )

        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(info["reset"])

        return response
```

**1.2.2 Add Rate Limiting to App**

Update `src/api_server/app.py`:

```python
from src.middleware.rate_limiter import RateLimitMiddleware

# Add middleware
app.add_middleware(RateLimitMiddleware)
```

**Files to Create:**
- `src/middleware/__init__.py`
- `src/middleware/rate_limiter.py`

**Files to Modify:**
- `src/api_server/app.py` (add middleware)
- `src/cache/redis_client.py` (ensure async methods)

#### 1.3 Audit Logging (2 days)

**1.3.1 Implement Audit Logger**

Create `src/audit/audit_logger.py`:

```python
"""
Audit logging for security and compliance.
"""
from datetime import datetime
from typing import Optional, Dict, Any
import json
import asyncio

from sqlalchemy.orm import Session
from prometheus_client import Counter

from src.database.models import AuditLog
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Prometheus metrics
audit_events = Counter(
    'audit_events_total',
    'Total number of audit events logged.',
    ['event_type', 'severity']
)


class AuditLogger:
    """
    Centralized audit logging for security events.

    Logs all:
    - Authentication events (login, logout, failed attempts)
    - Authorization events (permission denied)
    - Data mutations (create, update, delete)
    - Admin actions
    - API key operations
    """

    def __init__(self, db: Session):
        self.db = db

    async def log_event(
        self,
        event_type: str,
        user_id: Optional[str],
        resource: str,
        action: str,
        status: str,
        ip_address: str,
        user_agent: str,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "info"
    ):
        """
        Log an audit event.

        Args:
            event_type: Type of event (auth, data, admin, etc.)
            user_id: User who performed action (None for anonymous)
            resource: Resource affected (endpoint, model, etc.)
            action: Action performed (create, read, update, delete, login, etc.)
            status: Result status (success, failure, denied)
            ip_address: Client IP
            user_agent: Client user agent
            details: Additional context (dict)
            severity: Event severity (info, warning, error, critical)
        """
        try:
            audit_log = AuditLog(
                event_type=event_type,
                user_id=user_id,
                resource=resource,
                action=action,
                status=status,
                ip_address=ip_address,
                user_agent=user_agent,
                details=json.dumps(details) if details else None,
                severity=severity,
                timestamp=datetime.utcnow()
            )

            self.db.add(audit_log)
            self.db.commit()

            # Update metrics
            audit_events.labels(event_type=event_type, severity=severity).inc()

            # Also log to application logs for critical events
            if severity in ["error", "critical"]:
                logger.warning(
                    f"AUDIT [{severity.upper()}]: {event_type} - "
                    f"{action} on {resource} by {user_id or 'anonymous'} - {status}"
                )

        except Exception as e:
            logger.error(f"Failed to log audit event: {e}", exc_info=True)

    async def log_auth_event(
        self,
        action: str,
        user_id: Optional[str],
        email: str,
        status: str,
        ip_address: str,
        user_agent: str,
        failure_reason: Optional[str] = None
    ):
        """Log authentication event."""
        details = {"email": email}
        if failure_reason:
            details["failure_reason"] = failure_reason

        severity = "warning" if status == "failure" else "info"

        await self.log_event(
            event_type="auth",
            user_id=user_id,
            resource="authentication",
            action=action,
            status=status,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details,
            severity=severity
        )

    async def log_data_event(
        self,
        action: str,
        user_id: str,
        resource: str,
        resource_id: Optional[str],
        status: str,
        ip_address: str,
        user_agent: str,
        changes: Optional[Dict] = None
    ):
        """Log data mutation event."""
        details = {}
        if resource_id:
            details["resource_id"] = resource_id
        if changes:
            details["changes"] = changes

        await self.log_event(
            event_type="data",
            user_id=user_id,
            resource=resource,
            action=action,
            status=status,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details,
            severity="info"
        )

    async def log_admin_event(
        self,
        action: str,
        user_id: str,
        resource: str,
        status: str,
        ip_address: str,
        user_agent: str,
        details: Optional[Dict] = None
    ):
        """Log admin action (always warning level for visibility)."""
        await self.log_event(
            event_type="admin",
            user_id=user_id,
            resource=resource,
            action=action,
            status=status,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details,
            severity="warning"  # Admin actions always visible
        )


# Middleware for automatic audit logging
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

class AuditMiddleware(BaseHTTPMiddleware):
    """Middleware to automatically log all API requests."""

    async def dispatch(self, request: Request, call_next):
        """Log request and response."""
        start_time = datetime.utcnow()

        # Extract user info
        user_id = None
        try:
            auth_header = request.headers.get("Authorization")
            if auth_header:
                from src.auth.jwt_handler import jwt_handler
                token = auth_header.replace("Bearer ", "")
                payload = jwt_handler.decode_token(token)
                user_id = payload.get("sub")
        except:
            pass

        # Process request
        response = await call_next(request)

        # Log if mutation or admin action
        if request.method in ["POST", "PUT", "PATCH", "DELETE"]:
            from src.database import get_db
            db = next(get_db())
            audit_logger = AuditLogger(db)

            action = request.method.lower()
            resource = request.url.path
            status = "success" if response.status_code < 400 else "failure"
            severity = "error" if response.status_code >= 500 else "info"

            await audit_logger.log_event(
                event_type="api",
                user_id=user_id,
                resource=resource,
                action=action,
                status=status,
                ip_address=request.client.host,
                user_agent=request.headers.get("user-agent", "unknown"),
                details={
                    "method": request.method,
                    "status_code": response.status_code,
                    "duration_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
                },
                severity=severity
            )

        return response
```

**Files to Create:**
- `src/audit/__init__.py`
- `src/audit/audit_logger.py`

**Files to Modify:**
- `src/api_server/app.py` (add AuditMiddleware)
- `src/database/models.py` (add AuditLog model)

#### 1.4 Security Testing (2 days)

**1.4.1 Create Security Test Suite**

Create `tests/security/test_auth.py`:

```python
"""
Security tests for authentication and authorization.
"""
import pytest
from fastapi.testclient import TestClient
from src.api_server.app import app

client = TestClient(app)


class TestAuthentication:
    """Test authentication flows."""

    def test_register_user(self):
        """Test user registration."""
        response = client.post("/api/v1/auth/register", json={
            "email": "test@example.com",
            "password": "SecurePass123!"
        })
        assert response.status_code == 201
        assert "registered successfully" in response.json()["message"]

    def test_register_duplicate_user(self):
        """Test duplicate registration fails."""
        # Register first time
        client.post("/api/v1/auth/register", json={
            "email": "duplicate@example.com",
            "password": "SecurePass123!"
        })

        # Try again
        response = client.post("/api/v1/auth/register", json={
            "email": "duplicate@example.com",
            "password": "SecurePass123!"
        })
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]

    def test_login_success(self):
        """Test successful login."""
        # Register user
        client.post("/api/v1/auth/register", json={
            "email": "login@example.com",
            "password": "SecurePass123!"
        })

        # Login
        response = client.post("/api/v1/auth/login", json={
            "email": "login@example.com",
            "password": "SecurePass123!"
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_login_invalid_credentials(self):
        """Test login with wrong password."""
        client.post("/api/v1/auth/register", json={
            "email": "wrong@example.com",
            "password": "SecurePass123!"
        })

        response = client.post("/api/v1/auth/login", json={
            "email": "wrong@example.com",
            "password": "WrongPassword"
        })
        assert response.status_code == 401
        assert "Invalid credentials" in response.json()["detail"]

    def test_access_protected_endpoint_without_auth(self):
        """Test accessing protected endpoint without token."""
        response = client.post("/api/v1/predict", json={
            "hash": "0x123",
            "value": 100.0,
            "gas": 21000,
            "gasPrice": 20.0
        })
        assert response.status_code == 403  # Forbidden

    def test_access_protected_endpoint_with_auth(self):
        """Test accessing protected endpoint with valid token."""
        # Register and login
        client.post("/api/v1/auth/register", json={
            "email": "authed@example.com",
            "password": "SecurePass123!"
        })
        login_response = client.post("/api/v1/auth/login", json={
            "email": "authed@example.com",
            "password": "SecurePass123!"
        })
        token = login_response.json()["access_token"]

        # Access protected endpoint
        response = client.post(
            "/api/v1/predict",
            json={
                "hash": "0x123",
                "value": 100.0,
                "gas": 21000,
                "gasPrice": 20.0
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200

    def test_token_expiration(self):
        """Test that expired tokens are rejected."""
        # This would require time manipulation or creating expired token
        pass

    def test_refresh_token(self):
        """Test token refresh flow."""
        # Register and login
        client.post("/api/v1/auth/register", json={
            "email": "refresh@example.com",
            "password": "SecurePass123!"
        })
        login_response = client.post("/api/v1/auth/login", json={
            "email": "refresh@example.com",
            "password": "SecurePass123!"
        })
        refresh_token = login_response.json()["refresh_token"]

        # Refresh
        response = client.post("/api/v1/auth/refresh", json={
            "refresh_token": refresh_token
        })
        assert response.status_code == 200
        assert "access_token" in response.json()


class TestAuthorization:
    """Test role-based authorization."""

    def test_admin_endpoint_requires_admin_role(self):
        """Test that admin endpoints require admin role."""
        # Register regular user
        client.post("/api/v1/auth/register", json={
            "email": "user@example.com",
            "password": "SecurePass123!"
        })
        login_response = client.post("/api/v1/auth/login", json={
            "email": "user@example.com",
            "password": "SecurePass123!"
        })
        token = login_response.json()["access_token"]

        # Try admin endpoint
        response = client.post(
            "/api/v1/models/train",
            json={"contamination": 0.01},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403  # Forbidden
        assert "Insufficient permissions" in response.json()["detail"]


class TestAPIKeys:
    """Test API key authentication."""

    def test_create_api_key(self):
        """Test API key creation."""
        # Login
        client.post("/api/v1/auth/register", json={
            "email": "apikey@example.com",
            "password": "SecurePass123!"
        })
        login_response = client.post("/api/v1/auth/login", json={
            "email": "apikey@example.com",
            "password": "SecurePass123!"
        })
        token = login_response.json()["access_token"]

        # Create API key
        response = client.post(
            "/api/v1/auth/api-keys",
            json={"name": "My API Key", "expires_days": 30},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 201
        data = response.json()
        assert "key" in data
        assert data["key"].startswith("sk_")

    def test_use_api_key(self):
        """Test using API key for authentication."""
        # Create API key
        # (setup code omitted for brevity)
        api_key = "sk_..."

        # Use API key
        response = client.post(
            "/api/v1/predict",
            json={
                "hash": "0x123",
                "value": 100.0,
                "gas": 21000,
                "gasPrice": 20.0
            },
            headers={"X-API-Key": api_key}
        )
        assert response.status_code == 200


class TestRateLimiting:
    """Test rate limiting."""

    def test_rate_limit_enforcement(self):
        """Test that rate limits are enforced."""
        # Make 101 requests (limit is 100 per minute per IP)
        for i in range(101):
            response = client.get("/")
            if i < 100:
                assert response.status_code == 200
            else:
                assert response.status_code == 429  # Too Many Requests
                assert "Rate limit exceeded" in response.json()["detail"]["error"]

    def test_rate_limit_headers(self):
        """Test that rate limit headers are present."""
        response = client.get("/")
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers


class TestAuditLogging:
    """Test audit logging."""

    def test_login_events_logged(self):
        """Test that login attempts are logged."""
        # This would query the audit_logs table
        pass

    def test_admin_actions_logged(self):
        """Test that admin actions are logged."""
        pass

    def test_failed_auth_logged(self):
        """Test that failed authentication attempts are logged."""
        pass
```

**1.4.2 Run Security Scanners**

Add to CI/CD:
```bash
# OWASP ZAP scan
docker run -v $(pwd):/zap/wrk/:rw -t owasp/zap2docker-stable \
  zap-baseline.py -t http://localhost:8000 -r zap-report.html

# Dependency vulnerability scan
safety check --json

# Code security scan
bandit -r src/ -f json -o bandit-report.json
```

**Files to Create:**
- `tests/security/__init__.py`
- `tests/security/test_auth.py`
- `tests/security/test_rate_limiting.py`
- `tests/security/test_api_keys.py`
- `.github/workflows/security-scan.yml`

### Deliverables

- [ ] JWT authentication implemented
- [ ] User registration and login endpoints
- [ ] API key management for service accounts
- [ ] Rate limiting middleware active
- [ ] Audit logging for all mutations
- [ ] Security test suite (85%+ coverage)
- [ ] OWASP Top 10 compliance verified

### Success Criteria

- All endpoints protected with authentication
- Rate limits enforced (100 RPS per IP, configurable per endpoint)
- Audit logs captured for all security events
- Security tests passing
- No high/critical vulnerabilities in scans
- Documentation updated with auth examples

---

## Phase 2: Data Persistence Layer

**Duration:** 2 weeks
**Priority:** P0 (Critical Blocker)
**Team:** Backend + DevOps Engineers

### Objectives

1. Implement PostgreSQL database for persistent storage
2. Create database schema for all entities
3. Implement migration framework
4. Add backup and restore automation
5. Implement data retention policies

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Application Layer                      │
│  ├─ FastAPI API Server                                   │
│  └─ Kafka Stream Consumer                                │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│                  SQLAlchemy ORM                          │
│  ├─ Session management                                   │
│  ├─ Connection pooling                                   │
│  ├─ Query optimization                                   │
│  └─ Transaction management                               │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│              PostgreSQL Database                         │
│                                                           │
│  Tables:                                                  │
│  ├─ users (authentication)                               │
│  ├─ api_keys (API key management)                        │
│  ├─ transactions (historical blockchain data)            │
│  ├─ anomalies (detected anomalies)                       │
│  ├─ models (ML model metadata)                           │
│  ├─ model_versions (model versioning)                    │
│  ├─ predictions (prediction history)                     │
│  ├─ audit_logs (security audit trail)                    │
│  └─ system_metrics (performance tracking)                │
│                                                           │
│  Indexes:                                                 │
│  ├─ transactions(hash) [unique]                          │
│  ├─ transactions(timestamp)                              │
│  ├─ anomalies(severity, timestamp)                       │
│  ├─ predictions(transaction_hash, created_at)            │
│  └─ audit_logs(user_id, timestamp, event_type)           │
│                                                           │
│  Partitioning:                                            │
│  ├─ transactions: Range by timestamp (monthly)           │
│  ├─ anomalies: Range by timestamp (monthly)              │
│  └─ audit_logs: Range by timestamp (monthly)             │
└──────────────────────────────────────────────────────────┘
```

### Tasks

#### 2.1 Database Setup (3 days)

**2.1.1 PostgreSQL Deployment**

Create `k8s/postgresql-statefulset.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: postgresql-config
  namespace: blockchain-anomaly-prod
data:
  POSTGRES_DB: blockchain_anomaly
  POSTGRES_USER: anomaly_user
  PGDATA: /var/lib/postgresql/data/pgdata
---
apiVersion: v1
kind: Secret
metadata:
  name: postgresql-secret
  namespace: blockchain-anomaly-prod
type: Opaque
stringData:
  POSTGRES_PASSWORD: "CHANGE_ME_IN_PRODUCTION"
  # Generate with: openssl rand -base64 32
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgresql-pvc
  namespace: blockchain-anomaly-prod
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 200Gi
  storageClassName: ssd-storage
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgresql
  namespace: blockchain-anomaly-prod
spec:
  serviceName: postgresql
  replicas: 1
  selector:
    matchLabels:
      app: postgresql
  template:
    metadata:
      labels:
        app: postgresql
    spec:
      containers:
      - name: postgresql
        image: postgres:15-alpine
        ports:
        - containerPort: 5432
          name: postgres
        envFrom:
        - configMapRef:
            name: postgresql-config
        - secretRef:
            name: postgresql-secret
        volumeMounts:
        - name: postgresql-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - anomaly_user
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - anomaly_user
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: postgresql-storage
        persistentVolumeClaim:
          claimName: postgresql-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgresql
  namespace: blockchain-anomaly-prod
spec:
  selector:
    app: postgresql
  ports:
  - port: 5432
    targetPort: 5432
  clusterIP: None  # Headless service
```

**2.1.2 Connection Pooling Configuration**

Create `src/database/connection.py`:

```python
"""
Database connection management with connection pooling.
"""
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Generator
import logging

from src.utils.config import get_config

logger = logging.getLogger(__name__)
config = get_config()

# Database URL format: postgresql://user:password@host:port/database
DATABASE_URL = config.DATABASE_URL

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,              # Normal connection pool size
    max_overflow=10,           # Additional connections when needed
    pool_timeout=30,           # Timeout for getting connection from pool
    pool_recycle=3600,         # Recycle connections after 1 hour
    pool_pre_ping=True,        # Check connection health before using
    echo=False,                # Set to True for SQL query logging
    future=True                # Use SQLAlchemy 2.0 style
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)


@event.listens_for(engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    """Set up connection parameters on connect."""
    logger.debug("Database connection established")


@event.listens_for(engine, "close")
def receive_close(dbapi_conn, connection_record):
    """Log connection close."""
    logger.debug("Database connection closed")


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for getting database session.

    Usage in FastAPI:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager for database session.

    Usage:
        with get_db_context() as db:
            db.query(Model).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """Initialize database (create all tables)."""
    from src.database.models import Base
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized")


def check_db_connection() -> bool:
    """
    Check database connectivity.

    Returns:
        True if connection successful, False otherwise
    """
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False
```

**2.1.3 Database Models**

Create `src/database/models.py`:

```python
"""
SQLAlchemy database models.
"""
from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text,
    ForeignKey, Index, JSON, BigInteger, Enum as SQLEnum
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
import enum

Base = declarative_base()


class User(Base):
    """User authentication and profile."""
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    roles = Column(JSONB, default=["user"])  # ["user", "admin", "analyst"]
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Relationships
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="user")

    __table_args__ = (
        Index('idx_user_email_active', 'email', 'is_active'),
    )


class APIKey(Base):
    """API keys for service-to-service authentication."""
    __tablename__ = "api_keys"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    hashed_key = Column(String(255), nullable=False)
    prefix = Column(String(8), nullable=False, index=True)  # First 8 chars for lookup
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="api_keys")

    __table_args__ = (
        Index('idx_apikey_prefix_active', 'prefix', 'is_active'),
    )


class Transaction(Base):
    """Blockchain transactions (historical data)."""
    __tablename__ = "transactions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    hash = Column(String(66), unique=True, nullable=False, index=True)  # 0x + 64 hex chars
    block_number = Column(BigInteger, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    from_address = Column(String(42), nullable=False, index=True)
    to_address = Column(String(42), nullable=True, index=True)
    value = Column(Float, nullable=False)
    gas = Column(Float, nullable=False)
    gas_price = Column(Float, nullable=False)
    gas_used = Column(Float, nullable=True)
    nonce = Column(Integer, nullable=False)
    input_data = Column(Text, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    anomalies = relationship("Anomaly", back_populates="transaction")
    predictions = relationship("Prediction", back_populates="transaction")

    __table_args__ = (
        Index('idx_transaction_timestamp_value', 'timestamp', 'value'),
        Index('idx_transaction_from_to', 'from_address', 'to_address'),
        # Partition by timestamp (monthly)
        {
            'postgresql_partition_by': 'RANGE (timestamp)',
        }
    )


class SeverityEnum(enum.Enum):
    """Anomaly severity levels."""
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class Anomaly(Base):
    """Detected anomalies."""
    __tablename__ = "anomalies"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    transaction_id = Column(BigInteger, ForeignKey("transactions.id"), nullable=False)
    model_id = Column(String(36), ForeignKey("model_versions.id"), nullable=False)

    # Anomaly details
    anomaly_score = Column(Float, nullable=False)
    severity = Column(SQLEnum(SeverityEnum), nullable=False, index=True)
    confidence = Column(Float, nullable=False)

    # Explanation
    features_used = Column(JSONB, nullable=True)  # {"value": 1000, "gas": 50000}
    explanation = Column(Text, nullable=True)

    # Metadata
    detected_at = Column(DateTime, default=datetime.utcnow, index=True)
    reviewed = Column(Boolean, default=False)
    reviewed_by = Column(String(36), ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    false_positive = Column(Boolean, nullable=True)

    # Relationships
    transaction = relationship("Transaction", back_populates="anomalies")
    model_version = relationship("ModelVersion")

    __table_args__ = (
        Index('idx_anomaly_severity_detected', 'severity', 'detected_at'),
        Index('idx_anomaly_reviewed', 'reviewed', 'detected_at'),
        # Partition by detected_at (monthly)
        {
            'postgresql_partition_by': 'RANGE (detected_at)',
        }
    )


class Model(Base):
    """ML model metadata."""
    __tablename__ = "models"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    model_type = Column(String(50), nullable=False)  # "isolation_forest", "arima", etc.
    description = Column(Text, nullable=True)

    # Status
    is_active = Column(Boolean, default=True)
    created_by = Column(String(36), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    versions = relationship("ModelVersion", back_populates="model", cascade="all, delete-orphan")


class ModelVersion(Base):
    """ML model versions (supports A/B testing)."""
    __tablename__ = "model_versions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id = Column(String(36), ForeignKey("models.id"), nullable=False)
    version = Column(String(50), nullable=False)  # "1.0.0", "1.0.1", etc.

    # Model artifacts
    storage_path = Column(String(500), nullable=False)  # S3/GCS path
    checksum = Column(String(64), nullable=False)  # SHA256

    # Training metadata
    training_dataset_size = Column(Integer, nullable=True)
    training_duration_seconds = Column(Float, nullable=True)
    hyperparameters = Column(JSONB, nullable=True)

    # Performance metrics
    metrics = Column(JSONB, nullable=True)  # {"accuracy": 0.95, "f1": 0.92, ...}

    # Deployment
    is_deployed = Column(Boolean, default=False)
    deployed_at = Column(DateTime, nullable=True)
    traffic_percentage = Column(Float, default=0.0)  # For A/B testing (0-100)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(36), ForeignKey("users.id"), nullable=True)

    # Relationships
    model = relationship("Model", back_populates="versions")

    __table_args__ = (
        Index('idx_model_version_deployed', 'model_id', 'is_deployed'),
    )


class Prediction(Base):
    """Prediction history for audit and analysis."""
    __tablename__ = "predictions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    transaction_id = Column(BigInteger, ForeignKey("transactions.id"), nullable=True)
    model_version_id = Column(String(36), ForeignKey("model_versions.id"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)

    # Prediction
    is_anomaly = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=False)
    anomaly_score = Column(Float, nullable=True)

    # Request metadata
    request_id = Column(String(36), nullable=True, index=True)
    response_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    transaction = relationship("Transaction", back_populates="predictions")
    user = relationship("User", back_populates="predictions")

    __table_args__ = (
        Index('idx_prediction_created_at', 'created_at'),
        Index('idx_prediction_user_created', 'user_id', 'created_at'),
    )


class AuditLog(Base):
    """Security audit logs."""
    __tablename__ = "audit_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    # Event details
    event_type = Column(String(50), nullable=False, index=True)  # "auth", "data", "admin"
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    resource = Column(String(255), nullable=False)
    action = Column(String(50), nullable=False)  # "create", "read", "update", "delete"
    status = Column(String(20), nullable=False)  # "success", "failure", "denied"

    # Context
    ip_address = Column(String(45), nullable=False)  # IPv6 compatible
    user_agent = Column(String(500), nullable=True)
    details = Column(JSONB, nullable=True)

    # Severity
    severity = Column(String(20), nullable=False, default="info")  # "info", "warning", "error", "critical"

    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (
        Index('idx_audit_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_audit_event_timestamp', 'event_type', 'timestamp'),
        Index('idx_audit_severity_timestamp', 'severity', 'timestamp'),
        # Partition by timestamp (monthly)
        {
            'postgresql_partition_by': 'RANGE (timestamp)',
        }
    )


class SystemMetric(Base):
    """System performance metrics over time."""
    __tablename__ = "system_metrics"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    tags = Column(JSONB, nullable=True)  # {"pod": "api-1", "endpoint": "/predict"}
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (
        Index('idx_system_metric_name_timestamp', 'metric_name', 'timestamp'),
    )
```

**Files to Create:**
- `k8s/postgresql-statefulset.yaml`
- `k8s/postgresql-service.yaml`
- `k8s/postgresql-backup-cronjob.yaml`
- `src/database/__init__.py`
- `src/database/connection.py`
- `src/database/models.py`

**Files to Modify:**
- `src/utils/config.py` (add DATABASE_URL)
- `requirements.txt` (add psycopg2-binary, SQLAlchemy)

#### 2.2 Migration Framework (2 days)

**2.2.1 Set Up Alembic**

```bash
# Initialize Alembic
alembic init alembic

# Configure alembic.ini
```

Create `alembic/env.py`:

```python
"""Alembic migration environment."""
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

from src.database.models import Base
from src.utils.config import get_config

# Alembic Config object
config = context.config

# Setup logging
fileConfig(config.config_file_name)

# Get database URL from app config
app_config = get_config()
config.set_main_option("sqlalchemy.url", app_config.DATABASE_URL)

# Set target metadata for autogenerate
target_metadata = Base.metadata


def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

**2.2.2 Create Initial Migration**

```bash
# Generate migration from models
alembic revision --autogenerate -m "Initial schema"

# Review migration file in alembic/versions/

# Apply migration
alembic upgrade head
```

**2.2.3 Add Migration Script to Docker**

Update `docker/Dockerfile`:

```dockerfile
# Add migration step to entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
```

Create `docker/entrypoint.sh`:

```bash
#!/bin/bash
set -e

# Wait for database
echo "Waiting for database..."
while ! pg_isready -h $DATABASE_HOST -p $DATABASE_PORT -U $DATABASE_USER; do
  sleep 1
done
echo "Database is ready!"

# Run migrations
echo "Running database migrations..."
alembic upgrade head

# Start application
exec "$@"
```

**Files to Create:**
- `alembic/` (directory with migrations)
- `alembic/env.py`
- `alembic/versions/` (migration files)
- `docker/entrypoint.sh`

**Files to Modify:**
- `docker/Dockerfile` (add entrypoint)
- `requirements.txt` (add alembic)

#### 2.3 Repository Pattern Implementation (3 days)

**2.3.1 Base Repository**

Create `src/database/repositories/base_repository.py`:

```python
"""
Base repository with common CRUD operations.
"""
from typing import TypeVar, Generic, Type, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import desc

from src.database.models import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Base repository for common database operations."""

    def __init__(self, model: Type[ModelType], db: Session):
        self.model = model
        self.db = db

    def get(self, id: str) -> Optional[ModelType]:
        """Get single record by ID."""
        return self.db.query(self.model).filter(self.model.id == id).first()

    def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        order_by: str = "created_at"
    ) -> List[ModelType]:
        """Get all records with pagination."""
        query = self.db.query(self.model)
        if hasattr(self.model, order_by):
            query = query.order_by(desc(getattr(self.model, order_by)))
        return query.offset(skip).limit(limit).all()

    def create(self, obj: ModelType) -> ModelType:
        """Create new record."""
        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)
        return obj

    def update(self, id: str, updates: dict) -> Optional[ModelType]:
        """Update record by ID."""
        obj = self.get(id)
        if obj:
            for key, value in updates.items():
                if hasattr(obj, key):
                    setattr(obj, key, value)
            self.db.commit()
            self.db.refresh(obj)
        return obj

    def delete(self, id: str) -> bool:
        """Delete record by ID."""
        obj = self.get(id)
        if obj:
            self.db.delete(obj)
            self.db.commit()
            return True
        return False

    def count(self) -> int:
        """Count total records."""
        return self.db.query(self.model).count()
```

**2.3.2 Transaction Repository**

Create `src/database/repositories/transaction_repository.py`:

```python
"""
Repository for transaction data access.
"""
from typing import List, Optional
from datetime import datetime
from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from src.database.models import Transaction, Anomaly
from src.database.repositories.base_repository import BaseRepository


class TransactionRepository(BaseRepository[Transaction]):
    """Repository for blockchain transactions."""

    def __init__(self, db: Session):
        super().__init__(Transaction, db)

    def get_by_hash(self, hash: str) -> Optional[Transaction]:
        """Get transaction by hash."""
        return self.db.query(Transaction).filter(Transaction.hash == hash).first()

    def get_by_address(
        self,
        address: str,
        is_sender: bool = True,
        skip: int = 0,
        limit: int = 100
    ) -> List[Transaction]:
        """Get transactions for an address (as sender or receiver)."""
        if is_sender:
            filter_col = Transaction.from_address
        else:
            filter_col = Transaction.to_address

        return self.db.query(Transaction)\
            .filter(filter_col == address)\
            .order_by(Transaction.timestamp.desc())\
            .offset(skip)\
            .limit(limit)\
            .all()

    def get_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        skip: int = 0,
        limit: int = 1000
    ) -> List[Transaction]:
        """Get transactions in date range."""
        return self.db.query(Transaction)\
            .filter(and_(
                Transaction.timestamp >= start_date,
                Transaction.timestamp <= end_date
            ))\
            .order_by(Transaction.timestamp.asc())\
            .offset(skip)\
            .limit(limit)\
            .all()

    def get_high_value_transactions(
        self,
        min_value: float,
        limit: int = 100
    ) -> List[Transaction]:
        """Get high-value transactions."""
        return self.db.query(Transaction)\
            .filter(Transaction.value >= min_value)\
            .order_by(Transaction.value.desc())\
            .limit(limit)\
            .all()

    def get_statistics(self, start_date: Optional[datetime] = None):
        """Get transaction statistics."""
        query = self.db.query(
            func.count(Transaction.id).label('total'),
            func.sum(Transaction.value).label('total_value'),
            func.avg(Transaction.value).label('avg_value'),
            func.max(Transaction.value).label('max_value'),
            func.min(Transaction.value).label('min_value'),
            func.avg(Transaction.gas_price).label('avg_gas_price')
        )

        if start_date:
            query = query.filter(Transaction.timestamp >= start_date)

        return query.first()._asdict()

    def bulk_insert(self, transactions: List[dict]) -> int:
        """
        Bulk insert transactions (efficient for large datasets).

        Returns:
            Number of records inserted
        """
        try:
            self.db.bulk_insert_mappings(Transaction, transactions)
            self.db.commit()
            return len(transactions)
        except Exception as e:
            self.db.rollback()
            raise e
```

**2.3.3 Anomaly Repository**

Create `src/database/repositories/anomaly_repository.py`:

```python
"""
Repository for anomaly data access.
"""
from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy import and_, desc
from sqlalchemy.orm import Session, joinedload

from src.database.models import Anomaly, SeverityEnum, Transaction
from src.database.repositories.base_repository import BaseRepository


class AnomalyRepository(BaseRepository[Anomaly]):
    """Repository for detected anomalies."""

    def __init__(self, db: Session):
        super().__init__(Anomaly, db)

    def get_with_transaction(self, id: str) -> Optional[Anomaly]:
        """Get anomaly with related transaction data."""
        return self.db.query(Anomaly)\
            .options(joinedload(Anomaly.transaction))\
            .filter(Anomaly.id == id)\
            .first()

    def get_by_severity(
        self,
        severity: SeverityEnum,
        skip: int = 0,
        limit: int = 100
    ) -> List[Anomaly]:
        """Get anomalies by severity level."""
        return self.db.query(Anomaly)\
            .filter(Anomaly.severity == severity)\
            .order_by(Anomaly.detected_at.desc())\
            .offset(skip)\
            .limit(limit)\
            .all()

    def get_unreviewed(
        self,
        limit: int = 100
    ) -> List[Anomaly]:
        """Get anomalies awaiting review."""
        return self.db.query(Anomaly)\
            .filter(Anomaly.reviewed == False)\
            .order_by(
                Anomaly.severity.desc(),  # Critical first
                Anomaly.detected_at.asc()  # Oldest first
            )\
            .limit(limit)\
            .all()

    def get_recent(
        self,
        hours: int = 24,
        limit: int = 100
    ) -> List[Anomaly]:
        """Get anomalies from last N hours."""
        since = datetime.utcnow() - timedelta(hours=hours)
        return self.db.query(Anomaly)\
            .filter(Anomaly.detected_at >= since)\
            .order_by(Anomaly.detected_at.desc())\
            .limit(limit)\
            .all()

    def mark_reviewed(
        self,
        id: str,
        reviewer_id: str,
        is_false_positive: bool
    ) -> Optional[Anomaly]:
        """Mark anomaly as reviewed."""
        anomaly = self.get(id)
        if anomaly:
            anomaly.reviewed = True
            anomaly.reviewed_by = reviewer_id
            anomaly.reviewed_at = datetime.utcnow()
            anomaly.false_positive = is_false_positive
            self.db.commit()
            self.db.refresh(anomaly)
        return anomaly

    def get_false_positive_rate(
        self,
        start_date: Optional[datetime] = None
    ) -> float:
        """Calculate false positive rate."""
        query = self.db.query(Anomaly).filter(Anomaly.reviewed == True)

        if start_date:
            query = query.filter(Anomaly.detected_at >= start_date)

        total = query.count()
        if total == 0:
            return 0.0

        false_positives = query.filter(Anomaly.false_positive == True).count()
        return false_positives / total
```

**Files to Create:**
- `src/database/repositories/__init__.py`
- `src/database/repositories/base_repository.py`
- `src/database/repositories/transaction_repository.py`
- `src/database/repositories/anomaly_repository.py`
- `src/database/repositories/model_repository.py`
- `src/database/repositories/user_repository.py`
- `src/database/repositories/audit_repository.py`

#### 2.4 Integrate Database with API (2 days)

**2.4.1 Update API Endpoints to Use Database**

Update `src/api_server/app.py`:

```python
from sqlalchemy.orm import Session
from src.database import get_db
from src.database.repositories.transaction_repository import TransactionRepository
from src.database.repositories.anomaly_repository import AnomalyRepository
from src.database.models import Transaction as TransactionModel, Anomaly as AnomalyModel

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(
    transaction: TransactionData,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Predict if a transaction is anomalous and store in database."""

    # Create repositories
    transaction_repo = TransactionRepository(db)
    anomaly_repo = AnomalyRepository(db)

    # Check if transaction already exists
    existing_tx = transaction_repo.get_by_hash(transaction.hash)

    if not existing_tx:
        # Store transaction
        tx_model = TransactionModel(
            hash=transaction.hash,
            block_number=transaction.block_number or 0,
            timestamp=datetime.utcnow(),
            from_address=transaction.from_address or "0x0",
            to_address=transaction.to_address,
            value=transaction.value,
            gas=transaction.gas,
            gas_price=transaction.gasPrice,
            nonce=transaction.nonce or 0
        )
        existing_tx = transaction_repo.create(tx_model)

    # Process transaction through stream processor
    result = await stream_processor.process_transaction(transaction.dict())

    # If anomaly detected, store in database
    if result and result.get("is_anomaly"):
        anomaly = AnomalyModel(
            transaction_id=existing_tx.id,
            model_id=app_state["active_model_id"],
            anomaly_score=result.get("confidence", 0.0),
            severity=result.get("severity", "medium"),
            confidence=result.get("confidence", 0.0),
            features_used=result.get("features", {}),
            detected_at=datetime.utcnow()
        )
        anomaly_repo.create(anomaly)

    # Store prediction
    prediction = Prediction(
        transaction_id=existing_tx.id,
        model_version_id=app_state["active_model_id"],
        user_id=user["sub"],
        is_anomaly=result.get("is_anomaly", False),
        confidence=result.get("confidence", 0.0),
        response_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
    )
    db.add(prediction)
    db.commit()

    return PredictionResponse(
        transaction_hash=transaction.hash,
        is_anomaly=result.get("is_anomaly", False),
        confidence=result.get("confidence", 0.0),
        severity=result.get("severity"),
        timestamp=datetime.utcnow()
    )


@app.get("/api/v1/anomalies", response_model=AnomalyListResponse)
async def get_anomalies(
    severity: Optional[str] = None,
    limit: int = 100,
    skip: int = 0,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detected anomalies from database."""
    anomaly_repo = AnomalyRepository(db)

    if severity:
        anomalies = anomaly_repo.get_by_severity(
            SeverityEnum[severity],
            skip=skip,
            limit=limit
        )
    else:
        anomalies = anomaly_repo.get_all(skip=skip, limit=limit)

    return AnomalyListResponse(
        anomalies=[
            AnomalyRecord(
                id=a.id,
                transaction_hash=a.transaction.hash,
                severity=a.severity.value,
                confidence=a.confidence,
                detected_at=a.detected_at,
                reviewed=a.reviewed
            )
            for a in anomalies
        ],
        total=anomaly_repo.count()
    )


@app.get("/api/v1/transactions/{hash}", response_model=TransactionResponse)
async def get_transaction(
    hash: str,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get transaction details."""
    transaction_repo = TransactionRepository(db)
    tx = transaction_repo.get_by_hash(hash)

    if not tx:
        raise HTTPException(status_code=404, detail="Transaction not found")

    return TransactionResponse(
        hash=tx.hash,
        value=tx.value,
        gas=tx.gas,
        gasPrice=tx.gas_price,
        timestamp=tx.timestamp,
        from_address=tx.from_address,
        to_address=tx.to_address
    )
```

**Files to Modify:**
- `src/api_server/app.py` (integrate database)
- `src/streaming/stream_processor.py` (store anomalies in DB)
- `src/api_server/models.py` (add DB-related response models)

#### 2.5 Backup & Restore (2 days)

**2.5.1 Automated Backup CronJob**

Create `k8s/postgresql-backup-cronjob.yaml`:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgresql-backup
  namespace: blockchain-anomaly-prod
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  successfulJobsHistoryLimit: 7
  failedJobsHistoryLimit: 3
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
          - name: backup
            image: postgres:15-alpine
            env:
            - name: PGHOST
              value: postgresql
            - name: PGPORT
              value: "5432"
            - name: PGDATABASE
              valueFrom:
                configMapKeyRef:
                  name: postgresql-config
                  key: POSTGRES_DB
            - name: PGUSER
              valueFrom:
                configMapKeyRef:
                  name: postgresql-config
                  key: POSTGRES_USER
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: postgresql-secret
                  key: POSTGRES_PASSWORD
            - name: BACKUP_BUCKET
              value: "s3://blockchain-anomaly-backups"
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef:
                  name: aws-credentials
                  key: access_key_id
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: aws-credentials
                  key: secret_access_key
            command:
            - /bin/sh
            - -c
            - |
              # Install AWS CLI
              apk add --no-cache aws-cli

              # Create backup filename with timestamp
              BACKUP_FILE="backup-$(date +%Y%m%d-%H%M%S).sql.gz"

              # Dump database and compress
              pg_dump | gzip > /tmp/$BACKUP_FILE

              # Upload to S3
              aws s3 cp /tmp/$BACKUP_FILE $BACKUP_BUCKET/$BACKUP_FILE

              # Cleanup old backups (keep last 30 days)
              aws s3 ls $BACKUP_BUCKET/ | while read -r line; do
                createDate=$(echo $line | awk {'print $1" "$2'})
                createTimestamp=$(date -d"$createDate" +%s)
                olderThan=$(date -d "30 days ago" +%s)
                if [[ $createTimestamp -lt $olderThan ]]; then
                  fileName=$(echo $line | awk {'print $4'})
                  if [[ $fileName != "" ]]; then
                    aws s3 rm $BACKUP_BUCKET/$fileName
                  fi
                fi
              done

              echo "Backup completed: $BACKUP_FILE"
```

**2.5.2 Restore Script**

Create `scripts/restore_database.sh`:

```bash
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
```

**Files to Create:**
- `k8s/postgresql-backup-cronjob.yaml`
- `scripts/restore_database.sh`
- `scripts/test_backup.sh`

### Deliverables

- [ ] PostgreSQL deployed in Kubernetes
- [ ] Database schema created with all tables
- [ ] Migration framework (Alembic) configured
- [ ] Repository pattern implemented
- [ ] API endpoints integrated with database
- [ ] Automated daily backups to S3
- [ ] Restore procedure documented and tested
- [ ] Data retention policies implemented

### Success Criteria

- All data persisted to PostgreSQL
- Zero data loss on pod restarts
- Backup/restore tested successfully
- Query performance < 100ms (p95) for common queries
- Database migrations run automatically on deployment
- 30-day backup retention active

---

## Phase 3: ML Model Lifecycle

**Duration:** 2 weeks
**Priority:** P0 (Critical Blocker)
**Team:** ML Engineer + Backend Engineers

### Objectives

1. Implement full model training pipeline
2. Create model versioning and registry
3. Implement A/B testing framework
4. Add model monitoring and drift detection
5. Implement online learning capabilities

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Model Lifecycle                           │
│                                                             │
│  1. TRAINING                                                │
│     ├─ Fetch historical data                               │
│     ├─ Feature engineering                                  │
│     ├─ Model training (scikit-learn)                        │
│     ├─ Hyperparameter tuning (Optuna)                       │
│     ├─ Model evaluation                                     │
│     └─ Save model artifacts                                 │
│                                                             │
│  2. VERSIONING                                              │
│     ├─ Upload to S3/GCS                                     │
│     ├─ Store metadata in database                           │
│     ├─ Calculate checksum                                   │
│     └─ Version tagging (semver)                             │
│                                                             │
│  3. DEPLOYMENT                                              │
│     ├─ Shadow deployment (0% traffic)                       │
│     ├─ A/B testing (10% → 50% → 100%)                       │
│     ├─ Monitor performance metrics                          │
│     └─ Rollback if degradation                              │
│                                                             │
│  4. MONITORING                                              │
│     ├─ Prediction latency                                   │
│     ├─ Model accuracy (labeled data)                        │
│     ├─ Feature drift detection                              │
│     ├─ Concept drift detection                              │
│     └─ Alert on degradation                                 │
│                                                             │
│  5. RETRAINING                                              │
│     ├─ Triggered by drift detection                         │
│     ├─ Scheduled (weekly/monthly)                           │
│     ├─ Manual trigger by admin                              │
│     └─ Incremental learning (online mode)                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Tasks

#### 3.1 Model Training Pipeline (4 days)

**3.1.1 Training Orchestrator**

Create `src/ml/training/trainer.py`:

```python
"""
ML model training orchestrator.
"""
import os
import pickle
import hashlib
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import optuna
from optuna.samplers import TPESampler

from src.anomaly_detection.isolation_forest import AnomalyDetectorIsolationForest
from src.data_processing.data_transformation import DataTransformer
from src.database.repositories.transaction_repository import TransactionRepository
from src.database.repositories.model_repository import ModelRepository, ModelVersionRepository
from src.database.models import Model, ModelVersion
from src.ml.storage import ModelStorage
from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger(__name__)
config = get_config()


class ModelTrainer:
    """
    Orchestrates ML model training workflow.

    Steps:
    1. Fetch training data
    2. Feature engineering
    3. Hyperparameter tuning (optional)
    4. Train model
    5. Evaluate performance
    6. Save artifacts
    7. Register in model registry
    """

    def __init__(self, db_session):
        self.db = db_session
        self.transaction_repo = TransactionRepository(db_session)
        self.model_repo = ModelRepository(db_session)
        self.model_version_repo = ModelVersionRepository(db_session)
        self.storage = ModelStorage()
        self.transformer = DataTransformer()

    async def train_isolation_forest(
        self,
        model_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        hyperparameter_tuning: bool = True,
        contamination: Optional[float] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Train Isolation Forest model.

        Args:
            model_name: Name for the model
            start_date: Training data start date
            end_date: Training data end date
            hyperparameter_tuning: Whether to tune hyperparameters
            contamination: Expected anomaly proportion (None = auto-tune)

        Returns:
            Tuple of (model_version_id, metrics_dict)
        """
        logger.info(f"Starting training for model: {model_name}")
        training_start = datetime.utcnow()

        # Step 1: Fetch training data
        logger.info("Fetching training data...")
        transactions = self.transaction_repo.get_by_date_range(
            start_date or datetime(2024, 1, 1),
            end_date or datetime.utcnow(),
            limit=100000  # Max 100K for training
        )

        if len(transactions) < 1000:
            raise ValueError(f"Insufficient training data: {len(transactions)} transactions")

        logger.info(f"Loaded {len(transactions)} transactions for training")

        # Step 2: Convert to DataFrame and engineer features
        df = pd.DataFrame([{
            'hash': t.hash,
            'value': t.value,
            'gas': t.gas,
            'gasPrice': t.gas_price,
            'timestamp': t.timestamp
        } for t in transactions])

        # Feature engineering
        df = self.transformer.normalize_column(df, 'value')
        df = self.transformer.normalize_column(df, 'gas')
        df = self.transformer.normalize_column(df, 'gasPrice')

        # Add derived features
        df['value_per_gas'] = df['value'] / (df['gas'] + 1e-10)
        df['hour_of_day'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek

        # Select features
        feature_columns = ['value', 'gas', 'gasPrice', 'value_per_gas', 'hour_of_day', 'day_of_week']
        X = df[feature_columns].values

        # Step 3: Hyperparameter tuning
        best_params = {'contamination': contamination or 0.01, 'n_estimators': 100}

        if hyperparameter_tuning:
            logger.info("Running hyperparameter tuning...")
            best_params = self._tune_hyperparameters(X)
            logger.info(f"Best parameters: {best_params}")

        # Step 4: Train model with best parameters
        logger.info("Training model...")
        detector = AnomalyDetectorIsolationForest(
            contamination=best_params['contamination']
        )
        detector.model.set_params(n_estimators=best_params['n_estimators'])
        detector.train_model(df)

        # Step 5: Evaluate model
        logger.info("Evaluating model...")
        metrics = self._evaluate_model(detector, X)

        training_duration = (datetime.utcnow() - training_start).total_seconds()
        metrics['training_duration_seconds'] = training_duration
        metrics['training_samples'] = len(X)

        logger.info(f"Model metrics: {metrics}")

        # Step 6: Save model artifacts
        logger.info("Saving model artifacts...")
        model_path, checksum = self._save_model_artifacts(
            detector,
            model_name,
            best_params,
            metrics
        )

        # Step 7: Register in model registry
        logger.info("Registering model in registry...")
        model_version_id = self._register_model(
            model_name,
            model_path,
            checksum,
            best_params,
            metrics,
            len(X),
            training_duration
        )

        logger.info(f"Training completed. Model version: {model_version_id}")

        return model_version_id, metrics

    def _tune_hyperparameters(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Hyperparameter tuning using Optuna.

        Args:
            X: Training features

        Returns:
            Best hyperparameters dict
        """
        def objective(trial):
            # Define hyperparameter search space
            contamination = trial.suggest_float('contamination', 0.001, 0.1, log=True)
            n_estimators = trial.suggest_int('n_estimators', 50, 200, step=50)
            max_samples = trial.suggest_categorical('max_samples', ['auto', 256, 512, 1024])

            # Create model
            from sklearn.ensemble import IsolationForest
            model = IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                max_samples=max_samples,
                random_state=42,
                n_jobs=-1
            )

            # Cross-validation
            # Since anomaly detection is unsupervised, we use anomaly score variance
            model.fit(X)
            scores = model.score_samples(X)

            # Objective: Maximize score variance (better separation)
            return np.var(scores)

        # Run optimization
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=20, show_progress_bar=True)

        return study.best_params

    def _evaluate_model(
        self,
        detector: AnomalyDetectorIsolationForest,
        X: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        For unsupervised anomaly detection, we use:
        - Anomaly score distribution metrics
        - Silhouette score (if possible)
        - Number of anomalies detected

        Args:
            detector: Trained detector
            X: Test features

        Returns:
            Metrics dictionary
        """
        # Get anomaly scores
        scores = detector.model.score_samples(X)
        predictions = detector.model.predict(X)  # -1 = anomaly, 1 = normal

        num_anomalies = np.sum(predictions == -1)
        anomaly_rate = num_anomalies / len(predictions)

        metrics = {
            'num_samples': len(X),
            'num_anomalies_detected': int(num_anomalies),
            'anomaly_rate': float(anomaly_rate),
            'mean_anomaly_score': float(np.mean(scores)),
            'std_anomaly_score': float(np.std(scores)),
            'min_anomaly_score': float(np.min(scores)),
            'max_anomaly_score': float(np.max(scores)),
            'score_range': float(np.max(scores) - np.min(scores))
        }

        # If we have labeled data (from reviewed anomalies), calculate precision/recall
        # This would require fetching reviewed anomalies from DB
        # For now, we skip this

        return metrics

    def _save_model_artifacts(
        self,
        detector: AnomalyDetectorIsolationForest,
        model_name: str,
        hyperparameters: Dict,
        metrics: Dict
    ) -> Tuple[str, str]:
        """
        Save model artifacts to storage.

        Args:
            detector: Trained detector
            model_name: Model name
            hyperparameters: Hyperparameters used
            metrics: Evaluation metrics

        Returns:
            Tuple of (storage_path, checksum)
        """
        # Create temporary directory
        import tempfile
        tmp_dir = tempfile.mkdtemp()

        # Save model pickle
        model_file = os.path.join(tmp_dir, 'model.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(detector.model, f)

        # Save metadata
        metadata = {
            'model_name': model_name,
            'model_type': 'isolation_forest',
            'hyperparameters': hyperparameters,
            'metrics': metrics,
            'feature_columns': ['value', 'gas', 'gasPrice', 'value_per_gas', 'hour_of_day', 'day_of_week'],
            'training_timestamp': datetime.utcnow().isoformat(),
            'scikit_learn_version': '1.2.2'
        }

        metadata_file = os.path.join(tmp_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Calculate checksum
        with open(model_file, 'rb') as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        # Upload to storage (S3/GCS)
        storage_path = self.storage.upload_model(
            local_dir=tmp_dir,
            model_name=model_name,
            version=datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        )

        # Cleanup
        import shutil
        shutil.rmtree(tmp_dir)

        logger.info(f"Model artifacts saved to: {storage_path}")

        return storage_path, checksum

    def _register_model(
        self,
        model_name: str,
        storage_path: str,
        checksum: str,
        hyperparameters: Dict,
        metrics: Dict,
        training_size: int,
        training_duration: float
    ) -> str:
        """
        Register model in database registry.

        Args:
            model_name: Model name
            storage_path: Storage location
            checksum: SHA256 checksum
            hyperparameters: Hyperparameters used
            metrics: Evaluation metrics
            training_size: Number of training samples
            training_duration: Training time in seconds

        Returns:
            Model version ID
        """
        # Check if model exists
        model = self.model_repo.get_by_name(model_name)

        if not model:
            # Create new model
            model = Model(
                name=model_name,
                model_type='isolation_forest',
                description=f'Isolation Forest anomaly detector trained on {training_size} transactions'
            )
            model = self.model_repo.create(model)

        # Determine version number
        existing_versions = self.model_version_repo.get_by_model_id(model.id)
        version = f"1.0.{len(existing_versions)}"

        # Create model version
        model_version = ModelVersion(
            model_id=model.id,
            version=version,
            storage_path=storage_path,
            checksum=checksum,
            training_dataset_size=training_size,
            training_duration_seconds=training_duration,
            hyperparameters=hyperparameters,
            metrics=metrics,
            is_deployed=False,
            traffic_percentage=0.0
        )

        model_version = self.model_version_repo.create(model_version)

        logger.info(f"Model registered: {model_name} v{version} (ID: {model_version.id})")

        return model_version.id
```

**Files to Create:**
- `src/ml/__init__.py`
- `src/ml/training/__init__.py`
- `src/ml/training/trainer.py`
- `src/ml/training/evaluator.py`
- `src/ml/storage.py`

Due to length constraints, I'll now save this comprehensive plan and continue in the next response.

Let me save this document now:

