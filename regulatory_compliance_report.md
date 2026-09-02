# Regulatory Compliance Report
FirstBank Nigeria Limited — Cross-Sell Propensity Analytics Project
BAN6800 Final Project

Extends the Regulatory Compliance Checklist first published in the
Module 2 report with the additional frameworks named in the Final Project
brief (GDPR, CCPA, HIPAA), assessed for actual applicability rather than
checked automatically.

| Regulation | Applicability | Status / Action |
|---|---|---|
| **Nigeria Data Protection Act (NDPA), 2023** | Directly applicable — FirstBank is a Nigerian entity processing Nigerian customers' data | Processing basis documented and reviewed by the Data Protection Officer (Module 1). Pseudonymization and access controls implemented in the Module 3 pipeline. |
| **GDPR (EU) 2016/679** | Conditionally applicable — only to the subset of FirstBank customers who are EU residents or diaspora account holders | Right to explanation supported via the SHAP-based Explainability Framework (Module 4) and the Explanation Store (Final Project, Section 5). No EU-specific data residency measures have been implemented; if EU customer volume becomes material, a formal GDPR Article 22 (automated decision-making) review is required before scaling. |
| **CCPA (California Consumer Privacy Act)** | Not currently applicable — FirstBank has no California-resident customer base identified in this dataset or business scope | No action required at this stage. Documented here to close the assignment's compliance checklist explicitly, not to imply a compliance program is running for a jurisdiction with no customers. |
| **HIPAA** | Not applicable — this system processes financial transaction data, not protected health information, and FirstBank is not a covered entity under HIPAA | No action required. Documented for completeness. |
| **Central Bank of Nigeria (CBN) guidelines** | Directly applicable | Segment-driven offers reviewed to confirm no breach of consumer protection or pricing rules (Module 2). The Algorithmic Impact Assessment (Section 3 of this report package) confirms this model does not touch credit or pricing decisions, which keeps it outside CBN's stricter credit-decisioning guidance. |
| **PCI DSS** | Directly applicable to the source systems, not to this model | Card transaction fields are handled under FirstBank's existing cardholder data controls and are not duplicated into this project's analytics storage (Module 2). |

## Honest Note on Scope

Two of the four frameworks named in the Final Project brief (CCPA, HIPAA)
are not applicable to this system as currently scoped. Rather than
manufacture compliance activity against inapplicable law, this report
states that plainly: applying CCPA or HIPAA controls to a system with no
California residents and no health data would be compliance theater, not
compliance. If FirstBank's customer base or product scope changes, this
report should be revisited.
