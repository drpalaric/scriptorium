# Evaluation report: mapping

- Model: `claude-sonnet-5-5`
- Prompt: `mapping/v1` (sha256 `d631630734b7`)
- Parameters: effort `low`, thinking `adaptive`, max_tokens 1024
- Split: `dev-b`, condition `none`
- Date (UTC): 2026-10-25T02:04:59+00:00
- Mode: live
- Items: 407 scored of 407
- Statuses: {'ok': 407}
- Tokens: 32874 input, 4587 output, 290191 cache-read, 0 cache-write
- Cost this run: $0.1697 (407 live calls, 0 from cache); prices as of 2026-10-06

## Held-out scores

| Metric | Value | 95% CI (bootstrap over base controls) |
|---|---|---|
| Accuracy | 0.865 | 0.811 to 0.915 |
| Macro-F1 | 0.888 | 0.820 to 0.919 |

## Caveats

- Costs are computed from recorded token counts at listed prices and may differ from the invoice.
- Model output is not deterministic; answers served from the cache are reproducible, new calls may differ.
