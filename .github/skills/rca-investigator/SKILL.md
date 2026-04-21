---
name: rca-investigator
description: >
  Root Cause Analysis workflow. Use when a bug is confirmed and needs deep investigation:
  tracing code paths, analyzing dependencies, identifying the exact location and cause.
  Does NOT fix code — investigation and documentation only.
license: Apache-2.0
metadata:
  author: octobots
  version: "0.1.0"
---

# RCA Investigator

Investigate confirmed bugs with methodical precision. All findings go on the GitHub issue.

**INVESTIGATION ONLY — do NOT edit code, create PRs, or implement fixes.**

## Methodology: 4 Phases

### Phase 1: Ticket Acquisition

```bash
gh issue view <NUMBER>
```

Read all comments. Look for: reproduction steps (from issue-reproducer if available), error messages, stack traces, environment details, previous investigation notes.

```bash
gh issue comment <NUMBER> --body "🔍 **RCA Investigation Started**"
```

### Phase 2: Codebase Investigation

**2a. Locate the entry point:**
```bash
grep -rn "function_name\|error_message" src/ --include="*.py" -l
```

**2b. Trace the execution path:**
```
Entry point (route/handler)
  → Service layer (business logic)
    → Data layer (database/API calls)
      → Where it breaks
```

**2c. Examine the failure point:** What assumption is violated? What input causes it? Is the error here or in a dependency?

**2d. Search for patterns:**
```bash
grep -rn "function_name(" src/ --include="*.py"   # call sites
git --no-pager blame src/module.py -L 40,60        # who changed this
git --no-pager log --oneline -10 -- src/module.py  # recent changes
```

### Phase 3: Root Cause Determination

Classify:

| Category | Example |
|----------|---------|
| **Logic** | Wrong condition, off-by-one |
| **Data** | Null handling, type mismatch |
| **Concurrency** | Race condition, missing lock |
| **Configuration** | Wrong default, missing env var |
| **Integration** | API contract changed |
| **Resource** | Pool exhausted, OOM |

Confidence: **Confirmed** (see the exact bug) / **Highly likely** (strong evidence) / **Suspected** (needs more info).

### Phase 4: Impact Analysis & Report

**Usage analysis:**
```bash
grep -rn "broken_function" src/ --include="*.py"
grep -rn "from module import\|import module" src/ --include="*.py"
```

**Post RCA report:**
```bash
gh issue comment <NUMBER> --body "$(cat <<'EOF'
## 🔍 Root Cause Analysis Report

### Executive Summary
[One paragraph: what's broken and why]

### Classification
- **Category:** [Logic / Data / Concurrency / Config / Integration / Resource]
- **Confidence:** [Confirmed / Highly Likely / Suspected]
- **Severity:** [Critical / Major / Minor]

### Root Cause
**Location:** `src/module.py:42-58`
**Function:** `process_request()`
**Cause:** [Detailed description]

### Execution Path
```
POST /api/resource
  → routes.py:handler()
    → service.py:process()  ← BUG HERE (line 47)
```

### Impact Analysis
- **Affected areas:** [list]
- **Regression risk:** [Low / Medium / High]
- **Existing test coverage:** [Yes/No]

### Suggested Remediation
1. [Specific fix]
2. [Test to add]
EOF
)"
```

## What You DON'T Do

- Edit source code
- Create PRs or branches
- Run the application
- Make architectural decisions
- Close or reassign issues
