# Judge panel human-agreement ablation

All panels use the same 38 human-rated player-game rows and the same 13-label rubric.
Provider pins and sampling settings are recorded in `run_manifest.json`.

| Panel | Judges | Human κ | Agreement | Inter-judge α | Coverage |
|---|---|---:|---:|---:|---:|
| matched_original3 | paper-gemini-3.1-pro, glm-5.2, paper-claude-opus-4.6 | 0.812 | 92.1% | 0.917 | 100.0% |
| expanded5 | paper-gemini-3.1-pro, glm-5.2, paper-claude-opus-4.6, inkling-small, grok-4.3 | 0.801 | 91.6% | 0.880 | 100.0% |
| agreement_top3 | glm-5.2, deepseek-v4-flash-0731, grok-4.3 | 0.819 | 92.5% | 0.870 | 100.0% |
| noncohort3 | glm-5.2, inkling-small, grok-4.3 | 0.795 | 91.4% | 0.860 | 100.0% |

- `matched_original3` updates the original Gemini–GLM–Claude composition on the matched gold set.
- `expanded5` adds Inkling and Grok to test whether the result survives a broader panel.
- `agreement_top3` uses the three best complete candidates; none is an exact acting model.
- `noncohort3` removes Gemini and Claude, whose labs also appear in the acting cohort.
