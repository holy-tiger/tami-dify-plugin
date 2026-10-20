# Progress Agent Implementation Plan

> **For agentic workers:** Implement inline in the existing feature branch. Use test-driven-development and verification-before-completion. The user requested both design and creation in this session.

**Goal:** Deliver an independent Dify Function Calling strategy with variable-backed iteration budgets and progress checkpoints.

**Architecture:** Vendor the existing official Function Calling engine and its local helpers into a standalone plugin. Keep only validated budget settings and state calculation in `strategies/iteration_budget.py`; serialize runtime state in the invocation layer before token calculation. Progress policies belong to the Agent node's instruction, with copyable recommendations in the plugin README. Enforce finalization by withholding tools and refusing returned tool calls.

**Tech Stack:** Python 3.12, dify_plugin, Pydantic, unittest/pytest, Dify plugin CLI.

### Task 1: Budget contract

- [x] Create `agent-strategies/progress_agent/tests/test_iteration_budget.py`; test string values, defaults, invalid inputs and boundaries. Run with `python -m pytest tests/test_iteration_budget.py`; expect missing module initially.
- [x] Implement `strategies/iteration_budget.py`: `IterationBudget` validates four integer fields; `state(round)` returns serializable budget data without behavior instructions. Defaults are 30/20/5/1; omitted check start contracts for smaller budgets.
- [x] Re-run budget tests; all must pass.

### Task 2: Strategy integration

- [x] Copy the official Function Calling engine and local tool helpers/tests into the independent directory, preserving behavior unrelated to iteration policy.
- [x] Add integration tests in `tests/test_progress_agent.py` for budget injection, checkpoint timing, finalization, early exit, empty output, streaming, history isolation and tool-call batches. Run and observe failures against the original engine.
- [x] Extend parameter model with the budget model. Inject budget state into a copied system message on each call before token accounting. Withhold tools in finalization and reject unexpected tool calls. Emit iteration metadata with the final execution metadata.
- [x] Re-run both new integration tests and copied official regression tests.

### Task 3: Standalone packaging and instructions

- [x] Add `manifest.yaml`, `provider/agent.yaml`, `strategies/function_calling.yaml`, `main.py`, `pyproject.toml`, `uv.lock`, `_assets/icon.svg`, `.env.example`, `.difyignore`, `README.md`, and `README.zh_Hans.md` under the new plugin. Use identity `tami/progress_agent`, all four settings as templated strings, and dependency `dify_plugin>=0.10.2,<0.11`.
- [x] Document exact 30/20/5/1 example, direct variable selection, integer validation, cost/latency limits, historical context, and next-turn continuation. Do not promise an in-place pause.
- [x] Validate manifests and strategy/provider configuration using the installed SDK; verify all references and icon exist.
- [x] Run complete plugin tests, compile checks and Dify CLI packaging. Inspect packaged content, then run tests from the unpacked standalone artifact.
- [x] Review final diff and report artifact paths, configuration, verification results and remaining live-integration limitation.

### Task 4: Move progress policies to the Agent instruction (0.0.2)

- [x] Add an integration regression that checks every round's system message is exactly the user's instruction plus the JSON budget state; observe failure against the bundled behavior prompts.
- [x] Remove `IterationBudget.prompt`, serialize `state(round)` in the invocation layer, and retain hard tool/call limits.
- [x] Rewrite the primary plugin README with Chinese usage, runtime fields and a copyable Agent instruction; synchronize the localized README and plugin descriptions.
- [x] Refresh the lockfile, test, lint, package 0.0.2 and verify the extracted artifact independently. Source and extracted package each pass 55 tests and 44 subtests; Ruff, compile checks and independent review pass.
