# Prompt A/B — spot-check results

- date: 2026-10-07
- backend: mock
- A (incumbent): current extract suffix / no system prompt
- B (challenger): prompts/ab/extract_b.txt + chat_system_b.txt

## Per-case checks (failures listed)

| case | type | A | B | A failures | B failures |
|---|---|---|---|---|---|
| flat-order | extract | 5/5 | 5/5 | — | — |
| nested-customer | extract | 5/5 | 5/5 | — | — |
| array-items | extract | 5/5 | 5/5 | — | — |
| number-formats | extract | 5/5 | 5/5 | — | — |
| date-formats | extract | 5/5 | 5/5 | — | — |
| boolean-flags | extract | 5/5 | 5/5 | — | — |
| long-summary | extract | 5/5 | 5/5 | — | — |
| chat-capital | chat | 2/2 | 2/2 | — | — |
| chat-summarise | chat | 2/2 | 2/2 | — | — |
| chat-explain | chat | 2/2 | 2/2 | — | — |

## Totals

- A: 1.000 pass rate
- B: 1.000 pass rate

## Winner

**A** — tie on the mock backend — incumbent kept

tie-break rule: the challenger must strictly outscore the incumbent to displace production; ties keep A.
rerun with `--backend hf` (or another real backend) to measure differences the mock can't show.

## Winner kept

prompts/ab/winner.txt records the verdict. while the winner is A, production is unchanged — model.py's extract_json already uses the A suffix inline. to adopt B later, swap that suffix for prompts/ab/extract_b.txt.
