"""
audit_log.py
Privacy Audit Logging (Module 3, Section 6).

Every pipeline stage that reads, writes, or transforms customer data calls
audit_event(), producing a structured, append-only log of who/what/when for
each data access or transformation. This satisfies the "Privacy Audit Logging
for all data access and transformations" requirement and supports the
accountability commitment in the Module 1 Ethical AI Charter.
"""
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

LOG_DIR = Path("/home/claude/m3/repo/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "privacy_audit_log.jsonl"

logger = logging.getLogger("privacy_audit")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler(LOG_FILE)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

PIPELINE_ACTOR = "pipeline-service-account"  # non-human actor; RM/analyst access would log their SSO id

def audit_event(stage: str, dataset: str, rows: int = None, note: str = "", actor: str = PIPELINE_ACTOR):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actor": actor,
        "stage": stage,
        "dataset": dataset,
        "rows_affected": rows,
        "note": note,
    }
    logger.info(json.dumps(entry))
    return entry
