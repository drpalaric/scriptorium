# Evaluation report: mapping

- Model: `claude-sonnet-5-5`
- Prompt: `mapping/v3` (sha256 `73d60bd6015e`)
- Parameters: effort `low`, thinking `adaptive`, max_tokens 1024
- Split: `dev-b`, condition `examples`
- Date (UTC): 2026-10-25T02:10:45+00:00
- Mode: live
- Items: 407 scored of 407
- Statuses: {'ok': 407}
- Tokens: 333197 input, 4477 output, 330460 cache-read, 3280 cache-write
- Cost this run: $0.7855 (407 live calls, 0 from cache); prices as of 2026-10-06

## Held-out scores

| Metric | Value | 95% CI (bootstrap over base controls) |
|---|---|---|
| Accuracy | 0.872 | 0.834 to 0.913 |
| Macro-F1 | 0.888 | 0.826 to 0.913 |

## Caveats

- Costs are computed from recorded token counts at listed prices and may differ from the invoice.
- Model output is not deterministic; answers served from the cache are reproducible, new calls may differ.
