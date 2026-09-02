"""
render_ge_report.py
Renders the Great Expectations validation_result.json into a clean HTML
report, screenshotted for the PPTX deck (Module 3 deliverable requirement:
"Great Expectations suite screenshots").
"""
import json
from pathlib import Path

GE_DIR = Path("/home/claude/m3/repo/great_expectations")

with open(GE_DIR / "validation_result.json") as f:
    data = json.load(f)

rows = ""
for r in data["results"]:
    badge = '<span class="pass">PASS</span>' if r["success"] else '<span class="fail">FAIL</span>'
    col = r["column"] or "(table-level)"
    rows += f"<tr><td>{r['expectation_type']}</td><td>{col}</td><td>{badge}</td></tr>\n"

stats = data["statistics"]
html = f"""
<html><head><style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; background:#F7F8FA; padding:24px; }}
h1 {{ color:#1F3864; font-size:20px; margin-bottom:4px;}}
.sub {{ color:#595959; font-size:13px; margin-bottom:18px;}}
table {{ border-collapse: collapse; width: 100%; background:white; box-shadow: 0 1px 3px rgba(0,0,0,0.1);}}
th {{ background:#1F3864; color:white; text-align:left; padding:8px 12px; font-size:13px;}}
td {{ padding:7px 12px; font-size:13px; border-bottom:1px solid #eee; color:#333;}}
.pass {{ color:#1a7f37; font-weight:600; background:#e6f4ea; padding:2px 10px; border-radius:10px; font-size:12px;}}
.fail {{ color:#c0392b; font-weight:600; background:#fdecea; padding:2px 10px; border-radius:10px; font-size:12px;}}
.summary {{ display:flex; gap:14px; margin-bottom:18px; }}
.card {{ background:white; border-radius:8px; padding:12px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.card .n {{ font-size:22px; font-weight:700; color:#1F3864; }}
.card .l {{ font-size:11px; color:#595959; text-transform:uppercase; letter-spacing:.03em;}}
</style></head><body>
<h1>Great Expectations Validation &mdash; FirstBank Customer Analytics Table</h1>
<div class="sub">Suite: firstbank_customer_analytics_suite &nbsp;|&nbsp; Batch: customer_analytics_table.csv</div>
<div class="summary">
  <div class="card"><div class="n">{stats['evaluated_expectations']}</div><div class="l">Expectations Run</div></div>
  <div class="card"><div class="n" style="color:#1a7f37">{stats['successful_expectations']}</div><div class="l">Passed</div></div>
  <div class="card"><div class="n" style="color:#c0392b">{stats['unsuccessful_expectations']}</div><div class="l">Failed</div></div>
  <div class="card"><div class="n">{stats['success_percent']:.0f}%</div><div class="l">Success Rate</div></div>
</div>
<table>
<tr><th>Expectation</th><th>Column</th><th>Result</th></tr>
{rows}
</table>
</body></html>
"""
with open(GE_DIR / "validation_report.html", "w") as f:
    f.write(html)
print("Rendered validation_report.html")
