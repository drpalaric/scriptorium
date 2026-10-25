# Evaluation report: mapping

- Model: `claude-sonnet-5-5`
- Prompt: `mapping/v2` (sha256 `5818b16d5bcc`)
- Parameters: effort `low`, thinking `adaptive`, max_tokens 1024
- Split: `dev-b`, condition `parent`
- Date (UTC): 2026-10-25T02:07:21+00:00
- Mode: live
- Items: 407 scored of 407
- Statuses: {'ok': 407}
- Tokens: 73654 input, 4477 output, 320716 cache-read, 0 cache-write
- Cost this run: $0.2562 (407 live calls, 0 from cache); prices as of 2026-10-06

## Held-out scores

| Metric | Value | 95% CI (bootstrap over base controls) |
|---|---|---|
| Accuracy | 0.943 | 0.903 to 0.973 |
| Macro-F1 | 0.934 | 0.872 to 0.966 |

## Caveats

- Costs are computed from recorded token counts at listed prices and may differ from the invoice.
- Model output is not deterministic; answers served from the cache are reproducible, new calls may differ.
