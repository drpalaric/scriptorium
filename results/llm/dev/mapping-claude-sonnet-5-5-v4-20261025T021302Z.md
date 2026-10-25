# Evaluation report: mapping

- Model: `claude-sonnet-5-5`
- Prompt: `mapping/v4` (sha256 `03e8fe7ba02b`)
- Parameters: effort `low`, thinking `adaptive`, max_tokens 1024
- Split: `dev-b`, condition `examples_guidance`
- Date (UTC): 2026-10-25T02:13:02+00:00
- Mode: live
- Items: 407 scored of 407
- Statuses: {'ok': 407}
- Tokens: 759636 input, 4477 output, 340132 cache-read, 3376 cache-write
- Cost this run: $1.6405 (407 live calls, 0 from cache); prices as of 2026-10-06

## Held-out scores

| Metric | Value | 95% CI (bootstrap over base controls) |
|---|---|---|
| Accuracy | 0.875 | 0.838 to 0.915 |
| Macro-F1 | 0.898 | 0.837 to 0.923 |

## Caveats

- Costs are computed from recorded token counts at listed prices and may differ from the invoice.
- Model output is not deterministic; answers served from the cache are reproducible, new calls may differ.
