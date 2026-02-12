# Model Governance Framework

## 1. Overview

This document outlines the model risk management framework for the Credit One credit risk system, aligned with **SR 11-7** (Federal Reserve Guidance on Model Risk Management) and **Basel III** requirements.

## 2. Three Lines of Defense

### 2.1 First Line: Model Development

**Responsibility**: Build, implement, and maintain models

| Activity | Owner | Deliverable |
|----------|-------|-------------|
| Business Requirements | Product Manager | Requirements Document |
| Data Preparation | Data Engineer | Data Quality Report |
| Model Development | Data Scientist | Model Documentation |
| Implementation | ML Engineer | Production Code |
| Initial Testing | QA Engineer | Test Results |

**Key Controls**:
- Version control for all code (Git)
- Code review requirements (minimum 2 approvers)
- Automated testing (unit, integration, end-to-end)
- Documentation standards compliance

### 2.2 Second Line: Model Validation

**Responsibility**: Independent validation of model soundness

| Activity | Owner | Deliverable |
|----------|-------|-------------|
| Conceptual Soundness | Model Validator | Validation Report |
| Input Data Validation | Data Validator | Data Assessment |
| Performance Testing | Model Validator | Performance Report |
| Sensitivity Analysis | Model Validator | Sensitivity Report |
| Ongoing Monitoring | MRM Team | Monitoring Dashboard |

**Independence Requirements**:
- Validation team must be independent from development
- Reporting line to CRO (Chief Risk Officer)
- Budget independence from business units

### 2.3 Third Line: Internal Audit

**Responsibility**: Independent assurance on model risk management

| Activity | Owner | Frequency |
|----------|-------|-----------|
| Process Review | Internal Audit | Annual |
| Control Testing | Internal Audit | Quarterly |
| Regulatory Compliance | Compliance Officer | Continuous |
| Issue Tracking | Audit Committee | Monthly |

## 3. Model Lifecycle Management

### 3.1 Model Development Phase

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Business   │────▶│  Data       │────▶│  Model      │────▶│  Testing    │
│  Case       │     │  Collection │     │  Development│     │  & Validation│
└─────────────┘     └─────────────┘     └─────────────┘     └──────┬──────┘
                                                                    │
                                                                    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Model      │◀────│  Approval   │◀────│  Documentation│◀──│  Independent │
│  Inventory  │     │  (MRC)      │     │  Complete   │     │  Validation  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

**Required Documentation**:
1. **Model Development Document (MDD)**
   - Business purpose and use cases
   - Conceptual framework
   - Data sources and quality assessment
   - Methodology selection rationale
   - Model limitations and assumptions

2. **Model Validation Report (MVR)**
   - Independent validation findings
   - Testing results and conclusions
   - Recommendations and conditions
   - Approval or rejection decision

3. **Model Risk Rating**
   - High Risk: Material impact, complex methodology
   - Medium Risk: Moderate impact, standard methodology
   - Low Risk: Limited impact, simple methodology

### 3.2 Model Deployment Phase

**Pre-Deployment Checklist**:

```markdown
□ Technical Requirements
  □ Code review completed and approved
  □ Unit tests passing (coverage > 80%)
  □ Integration tests passing
  □ Performance tests meeting SLA (< 100ms p99)
  □ Security scan completed (no critical vulnerabilities)
  □ Documentation complete and reviewed

□ Model Validation
  □ Independent validation approved
  □ All validation findings addressed
  □ Model risk rating assigned
  □ Monitoring framework implemented
  □ Escalation procedures defined

□ Business Requirements
  □ Business sign-off obtained
  □ Training completed for users
  □ Operating procedures documented
  □ Rollback plan tested

□ Regulatory Requirements
  □ Compliance review completed
  □ Audit trail configured
  □ Data privacy requirements met
  □ Regulatory reporting configured
```

### 3.3 Model Monitoring Phase

**Ongoing Monitoring Requirements**:

| Metric | Frequency | Threshold | Action |
|--------|-----------|-----------|--------|
| Model Performance (AUC) | Daily | Drop > 0.03 | Alert |
| Population Stability (PSI) | Weekly | PSI > 0.25 | Investigation |
| Prediction Distribution | Daily | Shift > 2 std | Alert |
| Feature Drift | Weekly | PSI > 0.20 | Review |
| Latency | Real-time | p99 > 100ms | Escalate |
| Error Rate | Real-time | > 0.1% | Escalate |

**Monitoring Dashboard**:
```python
class ModelMonitoringDashboard:
    """
    Real-time model monitoring dashboard
    """
    
    def __init__(self, model_id):
        self.model_id = model_id
        self.metrics = {
            'performance': ['auc', 'ks', 'gini'],
            'stability': ['psi', 'csi'],
            'operations': ['latency', 'throughput', 'error_rate']
        }
    
    def generate_daily_report(self):
        """Generate daily monitoring report"""
        return {
            'date': datetime.now().date(),
            'model_id': self.model_id,
            'performance_metrics': self._get_performance_metrics(),
            'stability_metrics': self._get_stability_metrics(),
            'operational_metrics': self._get_operational_metrics(),
            'alerts': self._get_active_alerts(),
            'recommendations': self._generate_recommendations()
        }
```

### 3.4 Model Retirement Phase

**Retirement Triggers**:
- Model performance degradation beyond acceptable limits
- Business requirement changes
- Regulatory requirement changes
- Technology obsolescence
- Replacement model available

**Retirement Process**:
1. **Retirement Proposal**: Document rationale and impact
2. **Stakeholder Review**: Business, Risk, Compliance approval
3. **Migration Plan**: Data migration, process changes
4. **Communication**: Notify all users and stakeholders
5. **Execution**: Gradual rollout with monitoring
6. **Post-Retirement**: Archive model artifacts, close monitoring

## 4. Model Risk Assessment

### 4.1 Risk Rating Matrix

| Factor | Low (1) | Medium (2) | High (3) |
|--------|---------|------------|----------|
| **Materiality** | < $1M impact | $1M - $10M | > $10M |
| **Complexity** | Linear regression | Ensemble methods | Deep learning, NLP |
| **Data Quality** | Proven sources | Some uncertainty | Unverified sources |
| **Validation** | Full validation | Limited validation | No validation |
| **Monitoring** | Comprehensive | Basic | None |

**Risk Score = Sum of all factors**
- 5-8: Low Risk (Annual review)
- 9-12: Medium Risk (Semi-annual review)
- 13-15: High Risk (Quarterly review)

### 4.2 Risk Mitigation Strategies

| Risk Type | Mitigation Strategy | Owner |
|-----------|---------------------|-------|
| Model Risk | Independent validation, ongoing monitoring | MRM Team |
| Data Risk | Data quality checks, lineage tracking | Data Engineering |
| Implementation Risk | Code review, automated testing | ML Engineering |
| Operational Risk | SLA monitoring, failover procedures | DevOps |
| Regulatory Risk | Compliance reviews, audit trails | Compliance |

## 5. Model Inventory

### 5.1 Inventory Requirements

Every model must be registered with:

```yaml
model_inventory_entry:
  model_id: "PD_XGB_2024_001"
  model_name: "XGBoost PD Prediction Model"
  model_version: "2.1.0"
  
  ownership:
    business_owner: "Head of Retail Credit"
    technical_owner: "Lead Data Scientist"
    risk_owner: "Chief Risk Officer"
  
  classification:
    model_type: "Credit Risk - PD"
    risk_rating: "High"
    materiality: "$50M annual decisions"
  
  lifecycle:
    development_date: "2024-01-15"
    validation_date: "2024-02-20"
    deployment_date: "2024-03-01"
    last_review_date: "2024-06-01"
    next_review_date: "2024-09-01"
    retirement_date: null
  
  usage:
    business_process: "Loan Application Scoring"
    decision_type: "Approve/Decline/Refer"
    geographic_scope: "Hong Kong, Mainland China"
    customer_segments: ["SME", "Corporate"]
  
  documentation:
    development_doc: "link/to/mdd"
    validation_report: "link/to/mvr"
    monitoring_dashboard: "link/to/dashboard"
    code_repository: "github.com/..."
```

### 5.2 Model Registry Implementation

```python
from mlflow.tracking import MlflowClient
import mlflow

class ModelRegistry:
    """
    MLflow-based model registry with governance controls
    """
    
    def __init__(self, tracking_uri):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
    
    def register_model(self, model_name, model_path, 
                       validation_status, risk_rating):
        """
        Register model with governance metadata
        
        Parameters:
        -----------
        model_name : str
            Unique model identifier
        model_path : str
            Path to model artifact
        validation_status : str
            'APPROVED', 'CONDITIONAL', 'REJECTED'
        risk_rating : str
            'HIGH', 'MEDIUM', 'LOW'
        """
        # Only allow registration if validation passed
        if validation_status != 'APPROVED':
            raise ValueError(f"Cannot register model with status: {validation_status}")
        
        # Register model
        result = mlflow.register_model(model_path, model_name)
        
        # Add governance tags
        self.client.set_model_version_tag(
            name=model_name,
            version=result.version,
            key="validation_status",
            value=validation_status
        )
        
        self.client.set_model_version_tag(
            name=model_name,
            version=result.version,
            key="risk_rating",
            value=risk_rating
        )
        
        self.client.set_model_version_tag(
            name=model_name,
            version=result.version,
            key="approval_date",
            value=datetime.now().isoformat()
        )
        
        return result
    
    def transition_model_stage(self, model_name, version, 
                               new_stage, approver):
        """
        Transition model to new stage with approval
        
        Stages: None → Staging → Production → Archived
        """
        # Check approver permissions
        if not self._check_approver_permission(approver, new_stage):
            raise PermissionError(f"{approver} cannot approve {new_stage} transition")
        
        # Transition stage
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=new_stage
        )
        
        # Log transition
        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key=f"{new_stage.lower()}_approved_by",
            value=approver
        )
```

## 6. Regulatory Compliance

### 6.1 SR 11-7 Compliance Checklist

| Requirement | Implementation | Evidence |
|-------------|----------------|----------|
| Model Inventory | MLflow Registry | Registry logs |
| Model Validation | Independent validation team | Validation reports |
| Ongoing Monitoring | Automated dashboards | Monitoring logs |
| Documentation Standards | Template + Review | MDD/MVR documents |
| Three Lines of Defense | Org structure + RACI | Org chart |
| Board Reporting | Quarterly MRM reports | Board minutes |

### 6.2 Basel III Compliance

| Pillar | Requirement | Implementation |
|--------|-------------|----------------|
| Pillar 1 | PD/LGD/EAD estimation | Documented methodology |
| Pillar 1 | Model validation | Independent validation |
| Pillar 2 | Stress testing | Scenario analysis |
| Pillar 2 | ICAAP | Model risk in capital planning |
| Pillar 3 | Disclosure | Public model risk disclosures |

### 6.3 IFRS 9 Compliance

| Requirement | Implementation |
|-------------|----------------|
| 12-month ECL (Stage 1) | PD × LGD × EAD |
| Lifetime ECL (Stage 2) | Significant credit deterioration trigger |
| SICR Assessment | 30+ days past due or score decline |
| Forward-looking | Macro-economic scenarios |
| Collective assessment | Portfolio-level adjustments |

## 7. Escalation Procedures

### 7.1 Issue Escalation Matrix

| Issue Severity | Response Time | Escalation Path |
|----------------|---------------|-----------------|
| Critical (Production down) | 15 minutes | On-call → Engineering Manager → CTO |
| High (Performance degraded) | 1 hour | On-call → Team Lead → Engineering Manager |
| Medium (Alert threshold) | 4 hours | Team Lead → Weekly review |
| Low (Monitoring observation) | 24 hours | Weekly review → Monthly report |

### 7.2 Model Performance Degradation

**Trigger**: AUC drops by > 0.05 or PSI > 0.25

```
Hour 0:     Automated alert triggered
Hour 1:     On-call engineer investigates
Hour 4:     If unresolved, escalate to Model Owner
Hour 8:     If unresolved, escalate to MRM Team
Hour 24:    If unresolved, escalate to CRO
Day 3:      Emergency model review meeting
Day 7:      Decision: Retrain, Replace, or Retire
```

## 8. Training and Awareness

### 8.1 Required Training

| Role | Training | Frequency |
|------|----------|-----------|
| Model Developers | SR 11-7, Model Risk | Annual |
| Model Validators | Validation Techniques | Annual |
| Business Users | Model Interpretation | Annual |
| Senior Management | Model Risk Oversight | Annual |
| Board Members | Model Risk Governance | Annual |

### 8.2 Training Records

Maintain training records for audit purposes:
- Attendance logs
- Training materials
- Assessment results
- Certification status

## 9. Appendices

### Appendix A: Model Documentation Template
### Appendix B: Validation Report Template
### Appendix C: Monitoring Dashboard Specifications
### Appendix D: Escalation Contact List

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-08  
**Author**: Zheyu Liu  
**Approved By**: [To be signed by CRO]  
**Next Review Date**: 2026-05-08
