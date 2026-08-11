# CLAUDE.md — analytic_signal_gbm

Project-specific guidance for Claude Code, supplementing the global
`~/.claude/CLAUDE.md` (which resolves to `claude-infra/CLAUDE.md`). At present this
file exists to declare the cross-project search roots this project may read; add
further project conventions below as they arise.

## Directory Search Restrictions — cross-project roots

Supplements the global allowlist in `~/.claude/CLAUDE.md`, which covers
`~/.claude/`, `~/.analytic_signal_sst/`, `~/.brainsmash/`,
`~/PycharmProjects/claude-infra/`, `~/data/`, and the **hard** `$HOME` /
`/home/jjlee/` top-level search block. **This project additionally MAY search these
sibling project roots** (globs — always use the most specific path):

- `/Users/jjlee/PycharmProjects/analytic_signal*/`
- `/Users/jjlee/PycharmProjects/brainsmash*/`

Nothing here widens the `$HOME`-top-level block; it only names the
`analytic_signal*` / `brainsmash*` sibling roots this project is allowed to read.
Copied one-to-many from `~/.claude/CLAUDE.md` on 2026-08-11 (that file no longer
lists cross-project roots — they live per-project now).
