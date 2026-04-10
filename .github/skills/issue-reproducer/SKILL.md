---
name: issue-reproducer
description: >
  Bug reproduction specialist workflow. Use when a bug report is unclear, unconfirmed,
  or needs reproduction before investigation. Transforms vague reports into precise,
  repeatable steps with evidence. Does NOT fix code — reproduction and documentation only.
license: Apache-2.0
metadata:
  author: octobots
  version: "0.1.0"
---

# Issue Reproducer

Transform vague bug reports into precise, reproducible documentation with evidence. Post all findings on the GitHub issue — that is the source of truth.

**REPRODUCTION & DOCUMENTATION ONLY — do NOT fix code.**

## Workflow

### Phase 1: Issue Intake

```bash
gh issue view <NUMBER>
```

Read everything. Assess:
- **Clear steps provided** → follow them exactly
- **Partial steps** → fill in gaps systematically
- **No steps** → explore the feature area
- **Intermittent** → multiple attempts with timing variation

```bash
gh issue comment <NUMBER> --body "🔄 **Reproduction attempt started**: Reading the report and setting up."
```

### Phase 2: Environment Setup

Identify: target URL/endpoint, authentication requirements, prerequisite data or state. Document the environment — reproduction must be repeatable.

### Phase 3: Reproduction Attempts

#### UI Issues — Playwright MCP
```
browser_navigate(url) → browser_snapshot() → [follow steps] → browser_take_screenshot() → browser_console_messages() → browser_network_requests()
```

#### API Issues
```bash
curl -s -X POST http://localhost:PORT/api/endpoint \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"field": "value"}' -w "\n%{http_code}"
```

#### Logic Issues
```python
from module import function
result = function(edge_case_input)
print(f"Expected: X, Got: {result}")
```

#### Intermittent Issues
Run 5–10 times, vary timing, try different data values. Document success/failure rate.

### Phase 4: Technical Hints

Gather clues for RCA:
- Console errors — exact messages and stack traces
- Network — failed requests, unexpected status codes
- Patterns — when it works vs. when it doesn't
- Data patterns — specific inputs that trigger the bug

```bash
gh issue comment <NUMBER> --body "📝 **Technical Observations**\n\n- Console: \`...\`\n- Network: \`...\`"
```

### Phase 5: Confirmation Gate

**CONFIRMED → Ready for RCA:**
```bash
gh issue comment <NUMBER> --body "$(cat <<'EOF'
## ✅ Reproduction Status: CONFIRMED

**Reproduction Rate:** X/Y attempts
**Method:** [Playwright / curl / Python script]

### Steps to Reproduce
1. [Precise step]

### Expected / Actual
**Expected:** ...
**Actual:** ...

### Evidence
- Console: `[exact error]`
- API response: `[status + body]`

### Technical Hints for RCA
- Suspected area: `src/module.py`
- Trigger condition: [specific input/state]

✅ READY FOR RCA
EOF
)"
```

**CANNOT REPRODUCE:**
```bash
gh issue comment <NUMBER> --body "## ⚠️ CANNOT REPRODUCE\n\n**Attempts:** X\n**Info needed:** ..."
```

## What You DON'T Do

- Edit source code or create fixes
- Create PRs or branches
- Proceed to RCA if not confirmed
- Close issues
