# Data Architecture Design

## 1. Overview

This document describes the production-grade data architecture for the Credit One credit risk system, including real-world data integration patterns for banking environments.

## 2. Current Implementation vs. Production Architecture

### 2.1 Current State (Educational/Demo)

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Yahoo Finance  │────▶│  Circuit Breaker│────▶│  XGBoost Model  │
│     API         │     │  (Fault Toler.) │     │  (In-Memory)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                                               │
         │                                               ▼
         │                                      ┌─────────────────┐
         │                                      │  Streamlit UI   │
         │                                      │  (Local)        │
         │                                      └─────────────────┘
         ▼
┌─────────────────┐
│  SQLite (Local) │
│  (Demo Storage) │
└─────────────────┘
```

**Limitations**:
- Yahoo Finance provides market data, not credit data
- SQLite is file-based, not suitable for concurrent production access
- No data lineage or audit trail
- No integration with banking core systems

### 2.2 Target Production Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES LAYER                          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────┐ │
│  │  Credit      │  │  Alternative │  │  Internal    │  │  Macro  │ │
│  │  Bureaus     │  │  Data        │  │  Systems     │  │  Data   │ │
│  │  (Experian,  │  │  (Utility,   │  │  (Core       │  │  (CBRC, │ │
│  │  TransUnion, │  │  Telco,      │  │  Banking,    │  │  PBOC)  │ │
│  │  百行征信)    │  │  E-commerce) │  │  CRM)        │  │         │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └────┬────┘ │
│         │                 │                 │               │      │
│         └─────────────────┴─────────────────┴───────────────┘      │
│                                   │                                 │
│                                   ▼                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              API GATEWAY / DATA INGESTION LAYER             │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │   │
│  │  │  REST API   │  │  Message    │  │  Batch ETL          │  │   │
│  │  │  (Real-time)│  │  Queue      │  │  (Daily/Hourly)     │  │   │
│  │  │             │  │  (Kafka)    │  │                     │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA PROCESSING LAYER                          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              FEATURE ENGINEERING PIPELINE                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │   │
│  │  │  Data       │  │  Feature    │  │  Feature Store      │  │   │
│  │  │  Validation │──▶│  Transform  │──▶│  (Feast/Tecton)     │  │   │
│  │  │  (Great     │  │  (Spark/    │  │                     │  │   │
│  │  │  Expect.)   │  │  Python)    │  │  • Online features  │  │   │
│  │  └─────────────┘  └─────────────┘  │  • Offline features │  │   │
│  │                                     │  • Feature lineage  │  │   │
│  │                                     └─────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MODEL SERVING LAYER                            │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              MODEL DEPLOYMENT ARCHITECTURE                  │   │
│  │                                                             │   │
│  │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │   │
│  │  │   Load      │────▶│  Model      │────▶│  Prediction │   │   │
│  │  │   Balancer  │     │  Container  │     │  Cache      │   │   │
│  │  │   (Nginx)   │     │  (Docker)   │     │  (Redis)    │   │   │
│  │  └─────────────┘     └─────────────┘     └─────────────┘   │   │
│  │         │                   │                   │          │   │
│  │         │            ┌──────┴──────┐            │          │   │
│  │         │            │  A/B Test   │            │          │   │
│  │         │            │  Framework  │            │          │   │
│  │         │            └─────────────┘            │          │   │
│  │         │                                       │          │   │
│  │         ▼                                       ▼          │   │
│  │  ┌─────────────────────────────────────────────────────┐   │   │
│  │  │              MONITORING & LOGGING                   │   │   │
│  │  │  • Model performance metrics                        │   │   │
│  │  │  • Data drift detection (PSI)                       │   │   │
│  │  │  • Latency & throughput tracking                    │   │   │
│  │  │  • Audit logging                                    │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA STORAGE LAYER                             │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────┐ │
│  │  Transaction │  │  Data Lake   │  │  Feature     │  │  Audit │ │
│  │  Database    │  │  (S3/HDFS)   │  │  Store       │  │  Log   │ │
│  │  (PostgreSQL)│  │              │  │  (Redis/     │  │        │ │
│  │              │  │  • Raw data  │  │  Cassandra)  │  │        │ │
│  │  • Scores    │  │  • Features  │  │              │  │        │ │
│  │  • Decisions │  │  • Models    │  │              │  │        │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## 3. Real-World Data Integration

### 3.1 Credit Bureau Integration (China)

#### 百行征信 (Baihang Credit)
```python
# Production API Integration Pattern
import requests
from datetime import datetime
import hashlib
import hmac

class BaihangCreditAPI:
    """
    百行征信 API Integration
    
    API Documentation: https://www.baihangcredit.com/
    """
    
    def __init__(self, app_id, app_secret, env='production'):
        self.app_id = app_id
        self.app_secret = app_secret
        self.base_url = (
            'https://api.baihangcredit.com' 
            if env == 'production' 
            else 'https://sandbox-api.baihangcredit.com'
        )
    
    def _generate_signature(self, params, timestamp):
        """Generate HMAC-SHA256 signature"""
        message = f"{self.app_id}{timestamp}{json.dumps(params, sort_keys=True)}"
        signature = hmac.new(
            self.app_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def query_credit_report(self, id_number, name, mobile=None):
        """
        Query individual credit report
        
        Parameters:
        -----------
        id_number : str
            Chinese national ID (18 digits)
        name : str
            Full name (Chinese characters)
        mobile : str, optional
            Mobile phone number for verification
            
        Returns:
        --------
        dict : Credit report data
        """
        endpoint = '/v1/credit/report/individual'
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        
        params = {
            'id_number': id_number,
            'name': name,
            'query_type': 'loan_approval',
            'query_reason': '信贷审批',
            'timestamp': timestamp
        }
        
        if mobile:
            params['mobile'] = mobile
        
        headers = {
            'X-App-Id': self.app_id,
            'X-Timestamp': timestamp,
            'X-Signature': self._generate_signature(params, timestamp),
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.post(
                f"{self.base_url}{endpoint}",
                json=params,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            return self._parse_credit_report(response.json())
            
        except requests.exceptions.RequestException as e:
            # Log error and return fallback
            logger.error(f"Baihang API error: {e}")
            return self._get_fallback_data()
    
    def _parse_credit_report(self, raw_data):
        """Parse raw API response into structured format"""
        return {
            'credit_score': raw_data.get('credit_score'),
            'credit_history': {
                'total_accounts': raw_data.get('account_count', 0),
                'active_accounts': raw_data.get('active_account_count', 0),
                'overdue_accounts': raw_data.get('overdue_count', 0),
                'total_credit_limit': raw_data.get('total_credit_limit', 0),
                'total_balance': raw_data.get('total_balance', 0)
            },
            'query_history': {
                'queries_last_3m': raw_data.get('query_count_3m', 0),
                'queries_last_6m': raw_data.get('query_count_6m', 0),
                'queries_last_12m': raw_data.get('query_count_12m', 0)
            },
            'public_records': {
                'court_cases': raw_data.get('court_case_count', 0),
                'tax_liens': raw_data.get('tax_lien_count', 0),
                'bankruptcies': raw_data.get('bankruptcy_count', 0)
            },
            'timestamp': datetime.now().isoformat()
        }
```

#### 央行征信 (PBOC Credit)
```python
class PBOCCreditAPI:
    """
    中国人民银行征信中心 API
    
    Note: Direct API access is restricted to licensed financial institutions.
    This is a reference implementation for architecture documentation.
    """
    
    def __init__(self, institution_code, certificate_path):
        self.institution_code = institution_code
        self.certificate_path = certificate_path
        self.base_url = 'https://ipcrs.pbccrc.org.cn'
    
    def query_credit_report(self, id_number, query_type='detailed'):
        """
        Query PBOC credit report
        
        Requires: Financial institution license + Digital certificate
        """
        # Implementation would use institutional certificate
        # for mutual TLS authentication
        pass
```

### 3.2 Alternative Data Sources

#### 运营商数据 (Telecom Data)
```python
class TelecomDataAPI:
    """
    Mobile operator data for credit assessment
    
    Features:
    - Payment history (话费缴纳记录)
    - Account tenure (在网时长)
    - Usage patterns (消费模式)
    - Location stability (位置稳定性)
    """
    
    def get_telecom_features(self, mobile_number):
        """
        Extract credit-relevant features from telecom data
        
        Returns:
        --------
        dict : Telecom-derived features
        """
        return {
            'account_tenure_months': 36,  # 在网时长
            'payment_punctuality_rate': 0.95,  # 按时缴费率
            'avg_monthly_spend': 128.50,  # 月均消费
            'spend_consistency': 0.82,  # 消费稳定性
            'night_location_stability': 0.91,  # 夜间位置稳定性
            'roaming_frequency': 2,  # 漫游频率
            'number_portability_count': 0,  # 携号转网次数
        }
```

#### 电商数据 (E-commerce Data)
```python
class EcommerceDataAPI:
    """
    E-commerce platform data integration
    
    Platforms: 支付宝芝麻信用, 微信支付分, 京东小白信用
    """
    
    def get_alipay_features(self, alipay_id):
        """支付宝/芝麻信用 features"""
        return {
            'zhima_score': 750,  # 芝麻信用分
            'huabei_limit': 20000,  # 花呗额度
            'jiebei_limit': 50000,  # 借呗额度
            'payment_regularity': 0.98,  # 支付规律性
            'asset_stability': 0.85,  # 资产稳定性
        }
```

## 4. Data Flow Diagram

### 4.1 Real-Time Scoring Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Customer   │────▶│  Application│────▶│  API Gateway│────▶│  Feature    │
│  Applies    │     │  Received   │     │  (Kong/AWS) │     │  Retrieval  │
│  for Loan   │     │             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └──────┬──────┘
                                                                    │
                                                                    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Decision   │◀────│  Score      │◀────│  Model      │◀────│  Feature    │
│  Engine     │     │  Generated  │     │  Inference  │     │  Vector     │
│             │     │             │     │  (XGBoost)  │     │             │
└──────┬──────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Decision   │────▶│  Audit Log  │────▶│  Customer   │
│  (Approve/  │     │  (Immutable)│     │  Notified   │
│   Decline)  │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

### 4.2 Batch Model Training Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Historical │────▶│  Data Lake  │────▶│  Feature    │────▶│  Training   │
│  Loan Data  │     │  (S3/HDFS)  │     │  Engineering│     │  Pipeline   │
└─────────────┘     └─────────────┘     └─────────────┘     └──────┬──────┘
                                                                    │
                                                                    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Model      │◀────│  Validation │◀────│  Model      │◀────│  Hyperparam │
│  Registered │     │  (OOT/K-S)  │     │  Training   │     │  Tuning     │
│  (MLflow)   │     │             │     │  (XGBoost)  │     │             │
└──────┬──────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│  A/B Test   │────▶│  Production │
│  Deployment │     │  Deployment │
└─────────────┘     └─────────────┘
```

## 5. Security & Compliance

### 5.1 Data Security

| Layer | Security Measure | Implementation |
|-------|------------------|----------------|
| Transmission | TLS 1.3 | All API communications |
| Authentication | mTLS + API Keys | Mutual TLS for service-to-service |
| Authorization | RBAC | Role-based access control |
| Data at Rest | AES-256 Encryption | Database and storage encryption |
| PII Handling | Tokenization | Replace PII with tokens |
| Audit | Immutable Logs | Append-only audit trail |

### 5.2 Regulatory Compliance

| Regulation | Requirement | Implementation |
|------------|-------------|----------------|
| 个人信息保护法 | Consent management | Explicit opt-in for data usage |
| 数据安全法 | Data classification | Tagging by sensitivity level |
| 征信业管理条例 | Authorized access | License verification before queries |
| Basel III | Model validation | Independent validation team |
| IFRS 9 | ECL calculation | Stage 1/2/3 classification |

## 6. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)
- [ ] Set up PostgreSQL database with proper schema
- [ ] Implement API authentication framework
- [ ] Create data validation pipeline (Great Expectations)
- [ ] Set up logging and monitoring infrastructure

### Phase 2: Integration (Weeks 5-8)
- [ ] Integrate with 百行征信 (sandbox environment)
- [ ] Build feature engineering pipeline
- [ ] Implement feature store (Feast)
- [ ] Create model serving API

### Phase 3: Production (Weeks 9-12)
- [ ] Deploy to production environment
- [ ] Implement A/B testing framework
- [ ] Set up automated model monitoring
- [ ] Complete security audit

### Phase 4: Optimization (Weeks 13-16)
- [ ] Performance tuning (target: < 100ms latency)
- [ ] Implement caching layer (Redis)
- [ ] Set up auto-scaling
- [ ] Disaster recovery testing

## 7. References

1. [百行征信 API Documentation](https://www.baihangcredit.com/)
2. [中国人民银行征信中心](https://www.pbccrc.org.cn/)
3. [Basel III Framework - BIS](https://www.bis.org/bcbs/basel3.htm)
4. [Feast Feature Store](https://feast.dev/)
5. [Great Expectations](https://greatexpectations.io/)

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-08  
**Author**: Zheyu Liu  
**Status**: Architecture Design
