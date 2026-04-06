# LEARNINGS

## 1. Review Partial Chunks Against Their Claimed Scope First

- What failed or was discovered: Partial builder handoffs were hard to judge when chunk acceptance and later-plan gaps were mixed together.
- Why it matters: Over-scoping a partial review creates false rejections and noisy feedback.
- What changes next time: Approve or reject the claimed chunk first, then record broader-plan gaps separately.
- Where that change applies: Chunked trio reviews and any phased implementation work.
- Provenance or evidence: In the role-selection task, focused tests and help output were enough to approve the claimed chunk even though a later UI-path gap still existed.

## 2. Probe Local Config Paths Before Spending API Calls

- What failed or was discovered: A claimed interaction path can be locally disabled by config before any live-model call happens.
- Why it matters: Live validation is wasted if the code never enters the path being tested.
- What changes next time: Run cheap config-state and runtime-construction probes before external API or manual interaction testing.
- Where that change applies: Entrypoint reviews, UI toggles, and mode-selection features.
- Provenance or evidence: `--display_ui True` still left `UI` false until a local config probe exposed the gap.

## 3. Use Mixed Live And Deterministic Evidence When Providers Are Flaky

- What failed or was discovered: One real free-model startup plus deterministic local probes was enough to validate the user-facing contract when deeper interaction was blocked by provider limits.
- Why it matters: Waiting for a perfect end-to-end session can block correct decisions for reasons unrelated to product logic.
- What changes next time: Pair a minimal real startup with focused tests and local probes, then record provider failures as validation limits rather than product failures.
- Where that change applies: Reviewer validation for API-backed features.
- Provenance or evidence: The role-selection feature reached the requested human role from `main.py`, but deeper live play hit HTTP 429s.

## 4. Constructor Plumbing Changes Need Direct Probes For Every Touched Class

- What failed or was discovered: Focused log-routing tests passed while `LLMHumanAgent` still failed through MRO into `LLMAgent.__init__`.
- Why it matters: Class-specific constructor bugs can hide behind broader runtime coverage.
- What changes next time: When constructor plumbing or fallback behavior changes, directly instantiate every touched class and cover special modes such as `AmongUs(test=True)`.
- Where that change applies: Agent initialization, multiple inheritance, and log-routing work.
- Provenance or evidence: PR #22 initially looked correct on `main.py` isolation and `HumanAgent` fallback, but fresh review still found `LLMHumanAgent` broken.

## 5. Re-Run The Exact Failing Repro Before Broadening A Follow-Up Review

- What failed or was discovered: Repeating the exact failing constructor and game-init probes gave decisive approval evidence faster than a full re-review.
- Why it matters: It confirms the blocker is closed before spending time on wider validation.
- What changes next time: Keep prior blocker repro commands handy and run them first after a builder follow-up.
- Where that change applies: ESCALATE -> builder -> reviewer loops.
- Provenance or evidence: Re-running `LLMHumanAgent(...)` and `AmongUs(test=True).initialize_game()` quickly turned the PR #22 escalation into approval.

## 6. Classify Automated Review Comments By Current-Head Relevance

- What failed or was discovered: Automated PR review comments can span older commits and mix already-resolved suggestions with still-real regressions.
- Why it matters: Treating every comment as current wastes builder effort and muddies acceptance criteria.
- What changes next time: Fetch inline comments with commit IDs, compare them against current HEAD, and classify each as valid/fixed, valid/no-change, or already resolved in `HANDOFF.md` and `REVIEW.md`.
- Where that change applies: PR review loops that incorporate Copilot or other automated reviewers.
- Provenance or evidence: PR #22 had Copilot comments across multiple commits, but only a subset still applied on the final branch state.

## 7. After Shared Helper Contract Changes, Search Tests For Stale Mocks

- What failed or was discovered: Production code adapted to a new helper return shape while tests still monkeypatched the old one.
- Why it matters: Stale mocks can look like product regressions even though the runtime code is fine.
- What changes next time: Whenever a shared helper signature or return shape changes, grep tests for monkeypatches and stubs of that helper and update them or add intentional compatibility.
- Where that change applies: Utility changes that affect multiple call sites.
- Provenance or evidence: `setup_experiment()` moved to a tuple return and `tests/test_main_role_flag.py` still mocked a string.
