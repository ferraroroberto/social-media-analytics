# Project Instructions

Canonical instructions for AI coding agents working in this repository. Claude Code reads this file directly as project memory. Other agents (Cursor, Codex, etc.) reach it via the one-line `AGENTS.md` pointer.

## This repository
End-to-end data science project to analyze and predict social media content performance.
See `README.md` for setup, layout, and usage.

## Internal architecture

[`docs/architecture.mmd`](docs/architecture.mmd) is a hand-authored Mermaid diagram of this repo's own internal structure (entry points, `src/data` → `src/features` → `src/models` pipeline, external dependencies). Update it in the same PR as any material structural change (a new entry point, a pipeline stage added/moved/removed, a script relocated) — same anti-staleness contract as this repo's `.fleet.toml` `description` field. It is not auto-generated and not covered by any lint/test gate.
