# Historical Engineering Notes

This directory holds completed engineering working documents — research,
audits, and investigations that informed shipped work. They are kept for
context (the "why we built it this way") but are not actively maintained
and do not feed the public docs site (Fumadocs ingests `docs/guide/`
only — see `docs/source.config.ts`).

If anything here ever contradicts current code, **the code is the source
of truth.** These documents are point-in-time snapshots.

## Index

- [`PARITY_AUDIT.md`](PARITY_AUDIT.md) — DeepWiki ↔ Wikis storage and
  pipeline parity audit. Drove the unified `.wiki.db`, two-tier
  clustering, and protocol-unification work shipped under
  `feat/unified-db-and-clustering` (PR #111) and the follow-up
  Priority 2 cleanup (PRs #123, #124, merged 2026-05-15). All
  priorities are ✅ DONE.

- [`INVESTIGATION_GRAPH_QUALITY.md`](INVESTIGATION_GRAPH_QUALITY.md) —
  Graph-quality investigation against three production caches
  (`EliteaAI/configurations`, `EliteaAI/elitea-sdk`, `fmtlib/fmt`).
  Diagnosed orphan rates, cluster splitting, and class transitive
  linking gaps that motivated the topology-enrichment and expansion
  fixes in the same parity push.
