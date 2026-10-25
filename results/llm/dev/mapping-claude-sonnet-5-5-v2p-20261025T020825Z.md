# Evaluation report: mapping

- Model: `claude-sonnet-5-5`
- Prompt: `mapping/v2p` (sha256 `d631630734b7`)
- Parameters: effort `low`, thinking `adaptive`, max_tokens 1024
- Split: `dev-b` (enhancements only), condition `parent_only`
- Date (UTC): 2026-10-25T02:08:25+00:00
- Mode: live
- Items: 285 scored of 285
- Statuses: {'ok': 285}
- Tokens: 40495 input, 3135 output, 203205 cache-read, 0 cache-write
- Cost this run: $0.0849 (177 live calls, 108 from cache); prices as of 2026-10-06

## Held-out scores

| Metric | Value | 95% CI (bootstrap over base controls) |
|---|---|---|
| Accuracy | 0.954 | 0.899 to 0.993 |
| Macro-F1 | 0.942 | 0.878 to 0.986 |

## Caveats

- Costs are computed from recorded token counts at listed prices and may differ from the invoice.
- Model output is not deterministic; answers served from the cache are reproducible, new calls may differ.
