# q06-astra-y-round003

A specialized inference engine for [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) at revision `c1899de289a04d12100db370d81485cdf75e47ca`, built with EngineSmith 0.3.0.dev0.

- Machine: 16 x Trainium (trn2.48xlarge), 64 cores, 24 GiB, Ubuntu 24.04.5 LTS
- Reference: nxdi
- Judged under evaluation version 2
- Decode: not recorded (this run predates the one-forward-per-token rule)
- Source and data: https://github.com/xuanfeiren/enginesmith-q06-astra-y-round003
- Search: 20 rounds by a custom agent command (model us.openai.gpt-6-astra, reasoning effort high)

## Notes from the author

- This run predates the held-out final evaluation and the one-forward-per-token rule (evaluation version 2), so the numbers below come from the search evaluation.
- Round 003 is the best round of this run that generates one token per forward pass. From round 006 on, the agent turned to speculative decoding, whose speed on the evaluation's random-token prompts overstates what it gives on real text, so those rounds (and their scores in the table below) are not what is shared here.
- A clean rebuild of round 003, checked later with the current procedure on fresh held-out prompts (5 rounds, the engine given only the reference's tensor-parallel degree in cores): passed, 9 of 2080 judged steps mismatch; score 2.27x NxD Inference (prefill 2.13x, decode 2.42x). Decode probe 1.10.

## Speed against nxdi (search evaluation)

Score 2.30x; prefill 2.18x, decode 2.43x (reference time / engine time, 65 tokens per request; the fraction of the hardware bound is bound time / engine time, an estimate).

| prompt tokens | prefill | decode | prefill, fraction of the bound | decode, fraction of the bound |
|---:|---:|---:|---:|---:|
| 256 | 2.34x | 2.19x | not computed | not computed |
| 1,024 | 2.77x | 3.20x | not computed | not computed |
| 4,096 | 1.59x | 2.03x | not computed | not computed |

## Speed in absolute terms

Medians over the paired rounds of the search evaluation, this engine and nxdi on the same machine.

| prompt tokens | time to first token, ms | prefill tok/s | decode tok/s |
|---:|---:|---:|---:|
| 256 | 5.1 (nxdi 11.7) | 50,642.4 (21,929.0) | 508.9 (232.0) |
| 1,024 | 13.2 (nxdi 36.8) | 77,305.6 (27,790.5) | 466.3 (145.7) |
| 4,096 | 61.2 (nxdi 97.2) | 66,965.8 (42,120.6) | 371.4 (182.8) |

## Checks on the decode speed

- Natural text: not measured (no task accuracy answers).
- Decode probe: 64 one-token requests took 1.10x as long as one request for the same tokens at 256 prompt tokens (137.9 ms against 125.5 ms; each request 2.04 to 2.49 ms). One forward pass per generated token keeps this near 1; speculative decoding raises it.

## How the search went

- 20 coding sessions by a custom agent command (model us.openai.gpt-6-astra, reasoning effort high); the search took 2045 min.
- First correct engine: round 000, 16 min after the start.
- First engine as fast as nxdi (score 1.0): round 000, 16 min after the start.
- Agent usage: 5,080k input and 2,405k output tokens, 641,356k cache reads, cost not reported (as the agent reported them).

| round | kind | outcome | ended, min | score | prefill | decode | tokens in / out | engine lines |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 000 | seed | accepted (first_passing_candidate) | 15.6 | 2.21x | 2.04x | 2.39x | - / - | - |
| 001 | optimize | accepted (segment_improved) | 112.8 | 2.23x | 2.08x | 2.40x | 305k / 165k | 15,765 |
| 002 | optimize | accepted (case_improved) | 180.4 | 2.23x | 2.11x | 2.35x | 249k / 107k | 18,835 |
| 003 | optimize | accepted (score_improved) | 229.8 | 2.30x | 2.18x | 2.43x | 167k / 53k | 22,103 |
| 004 | optimize | rejected (insufficient_improvement) | 340.0 | 2.29x | 2.19x | 2.40x | 341k / 185k | 30,214 |
| 005 | optimize | accepted (case_improved) | 393.1 | 2.29x | 2.19x | 2.39x | 186k / 64k | 24,700 |
| 006 | optimize | accepted (score_improved) | 511.0 | 3.01x | 2.19x | 4.13x | 296k / 158k | 31,319 |
| 007 | optimize | accepted (score_improved) | 582.6 | 3.26x | 2.24x | 4.72x | 185k / 69k | 36,340 |
| 008 | optimize | rejected (insufficient_improvement) | 652.0 | 3.21x | 2.21x | 4.67x | 163k / 52k | 41,071 |
| 009 | optimize | accepted (score_improved) | 740.3 | 3.68x | 2.18x | 6.19x | 193k / 75k | 42,265 |
| 010 | optimize | rejected (insufficient_improvement) | 819.1 | 3.60x | 2.20x | 5.88x | 168k / 53k | 47,040 |
| 011 | optimize | rejected (insufficient_improvement) | 904.1 | 3.08x | 2.19x | 4.33x | 201k / 82k | 49,624 |
| 012 | optimize | rejected (insufficient_improvement) | 991.8 | 2.83x | 2.20x | 3.64x | 187k / 74k | 49,321 |
| 013 | optimize | accepted (score_improved) | 1,193.3 | 3.87x | 2.19x | 6.83x | 472k / 306k | 65,601 |
| 014 | optimize | accepted (case_improved) | 1,321.3 | 3.91x | 2.21x | 6.92x | 356k / 167k | 74,928 |
| 015 | optimize | rejected (insufficient_improvement) | 1,473.4 | 3.77x | 2.20x | 6.46x | 357k / 184k | 91,389 |
| 016 | optimize | rejected (insufficient_improvement) | 1,583.0 | 3.84x | 2.23x | 6.60x | 239k / 104k | 81,247 |
| 017 | optimize | rejected (insufficient_improvement) | 1,742.4 | 2.73x | 2.21x | 3.36x | 296k / 150k | 96,621 |
| 018 | optimize | rejected (insufficient_improvement) | 1,918.9 | 3.82x | 2.21x | 6.61x | 404k / 222k | 93,612 |
| 019 | optimize | rejected (insufficient_improvement) | 2,045.2 | 3.46x | 2.21x | 5.43x | 317k / 133k | 83,953 |

`history.csv` in https://github.com/xuanfeiren/enginesmith-q06-astra-y-round003 has every round with every prompt length; `enginesmith.json` has the same and more.

## Use it

These are the author's measurements. Check them on a machine like this one (16 x Trainium (trn2.48xlarge); this builds the engine from `source/` and compares it with your local nxdi):

```sh
hf download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca --local-dir models/Qwen3-0.6B
uv run enginesmith verify https://github.com/xuanfeiren/enginesmith-q06-astra-y-round003 --model models/Qwen3-0.6B
```

Build it yourself: `bash build.sh` in `source/`, then run `target/engine --model MODEL_DIR`; it speaks the engine protocol of EngineSmith's `docs/protocol.md` (token ids in, greedy token ids out). Read the source before you run it: it is native code written by a coding agent.

`enginesmith.json` holds every setting and result, and the SHA-256 of each model file.

## License

MIT, see `LICENSE`. Code adapted from other projects keeps its own license and notices (`source/THIRD_PARTY.md` lists it when the agent recorded any).
