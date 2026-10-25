# Evaluation report: mapping

- Model: `claude-sonnet-5-5`
- Prompt: `mapping/v1` (sha256 `d631630734b7`)
- Parameters: effort `low`, thinking `adaptive`, max_tokens 1024
- Date (UTC): 2026-10-25T01:34:46+00:00
- Mode: live
- Items: 184 scored of 184
- Statuses: {'ok': 184}
- Tokens: 14875 input, 2196 output, 128340 cache-read, 2852 cache-write
- Cost this run: $0.0693 (164 live calls, 20 from cache); prices as of 2026-10-06

## Held-out scores

| Metric | Value | 95% CI (bootstrap over base controls) |
|---|---|---|
| Accuracy | 0.875 | 0.807 to 0.937 |
| Macro-F1 | 0.893 | 0.812 to 0.938 |

## Against the scikit-learn baseline (same items)

- sklearn accuracy 0.522, macro-F1 0.503
- LLM minus sklearn, accuracy: +0.353 (95% CI +0.237 to +0.479)
- LLM minus sklearn, macro-F1: +0.391 (95% CI +0.292 to +0.484)
- Both right 89, only LLM 72, only sklearn 7, neither 16

## Caveats

- Costs are computed from recorded token counts at listed prices and may differ from the invoice.
- Model output is not deterministic; answers served from the cache are reproducible, new calls may differ.
