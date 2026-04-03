# ARM64 NEON Validation Harness

Built: 2026-03-31
Platform: AWS Graviton3 (AArch64), Ubuntu 22.04
Purpose: Automated correctness + perf regression gate for ARM64 NEON ports

---

## Overview

Three-phase validation for every NEON implementation:

| Phase | Tool | What it checks | Exit on failure |
|---|---|---|---|
| 0 | stderr parse | Hardware PMU backend active (`CONFIG_LINUX_PERF`) | exit 2 |
| 1 | stderr parse | checkasm correctness tests pass (bench suppressed on fail) | exit 3 |
| 2 | stdout --csv parse | Cycle regression vs `baselines.json` + speedup ratio ≥ 0.80x | exit 1 |
| 3 | stdout | Structured JSON result block for Veda auditor | — |

---

## Quick start

```bash
# Prerequisite (once per boot)
sudo sysctl -w kernel.perf_event_paranoid=-1

# Run via Makefile (preferred)
make bench-validate MODULE=audiodsp

# Run directly
python3 bench/arm64_validate.py \
    --checkasm-path tests/checkasm/checkasm \
    --module audiodsp

# Update baseline after an intentional improvement
python3 bench/arm64_validate.py \
    --checkasm-path tests/checkasm/checkasm \
    --module audiodsp \
    --update-baseline
```

---

## CLI reference

```
bench/arm64_validate.py
  --checkasm-path PATH   Path to checkasm binary (required)
  --baselines PATH       Path to baselines JSON (default: bench/baselines.json)
  --module NAME          Module/function prefix to test (e.g. audiodsp, blockdsp)
  --bench-pattern NAME   Override bench function prefix when it differs from --module
                         (see naming divergence section below)
  --threshold FLOAT      Minimum NEON speedup ratio vs C (default: 0.80)
  --update-baseline      Promote live measurements to new baseline (only on PASS)
  --json-only            Print only JSON result block (no table)
```

---

## Regression gates

**Cycle regression**: `new_cycles > baseline_cycles * 1.20` → `REGRESSED` (gate fail)

**Ratio threshold**: `neon_speedup < 0.80x` → `BELOW_THRESHOLD` (gate fail)
- Skipped for entries with `"skip_threshold": true` in baselines.json

**Known regression**: entries with `"regression": true` → `KNOWN_REGRESSION` (yellow, visible but not a gate fail)

---

## baselines.json schema

```json
{
  "_meta": {
    "platform": "Graviton3",
    "perf_backend": "CONFIG_LINUX_PERF",
    "perf_event_paranoid_required": -1
  },
  "functions": {
    "audiodsp.scalarproduct_int16": {
      "c":    {"cycles": 4134.8},
      "neon": {"cycles": 688.1, "ratio": 6.01}
    },
    "blockdsp.fill_block_tab[0]": {
      "c":    {"cycles": 20.6},
      "neon": {"cycles": 24.9, "ratio": 0.82,
               "regression": true,
               "regression_note": "flagged for investigation"}
    },
    "flac_lpc_16_13": {
      "c":    {"cycles": 3428.6},
      "neon": {"cycles": 3456.9, "ratio": 0.99,
               "skip_threshold": true,
               "skip_note": "dispatch threshold pred_order >= 20; slow by design"}
    }
  }
}
```

---

## Naming divergence

checkasm's `--test=` pattern matches the test group name; `--bench=` matches the
`check_func` registered name. These differ for some modules. All divergences are
now embedded in `MODULE_BENCH_PATTERNS` in `arm64_validate.py` — `make bench-validate`
handles them automatically.

| Module | Bench pattern(s) |
|---|---|
| alacdsp | `alac_` |
| lpc | `apply_welch_window`, `autocorr` |
| llviddsp | `add_` |
| flacdsp | `flac_` |
| vp3dsp | `v_loop_filter`, `h_loop_filter` |
| vf_blend | `""` (all functions) |
| mlpdsp | `mlp_` |
| takdsp | `decorrelate_` |
| hpeldsp | `put_pixels_tab`, `avg_pixels_tab`, `put_no_rnd_pixels_tab`, `avg_no_rnd_pixels_tab` |
| dcadsp | `lfe_fir` |
| g722dsp | `g722_apply_qmf` |
| utvideodsp | `restore_rgb_planes` |
| llauddsp | `scalarproduct_and_madd` |
| exrdsp | `reorder_pixels`, `predictor` |

---

## Ported module status (2026-04-02)

| Module | Correctness | Perf gate | Key speedups |
|---|---|---|---|
| audiodsp | PASS | PASS | scalarproduct_int16 6.01x |
| blockdsp | PASS | PASS | fill_block_tab[0] KNOWN_REGRESSION (0.93x) |
| alacdsp | PASS 3/3 | PASS | decorrelate_stereo 3.07x, append_extra_bits 4.59x |
| flacdsp | PASS | PASS 52/52 | indep2/4/6/8 × S16/S32 all NEON |
| bswapdsp | PASS 4/4 | PASS | |
| lpc | PASS | PASS 12/12 | autocorr exactly 2.00x |
| llviddsp | PASS | PASS | baselines refreshed |
| vp9dsp | PASS | PASS | vp9_dc_128_8x8 KNOWN_REGRESSION (noise floor) |
| vf_blend | PASS 62/62 | PASS 62/62 | 17 8-bit + 14 16-bit blend modes |
| vp3dsp | PASS 2/2 | PASS | v_loop_filter 1.53x |
| takdsp | PASS | PASS | decorrelate_ls/sr/sm/sf NEON |
| dcadsp | PASS | PASS | lfe_fir float + fixed |
| g722dsp | PASS | PASS | apply_qmf single-pass |
| utvideodsp | PASS | PASS | restore_rgb_planes + 10-bit |
| llauddsp | PASS 2/2 | PASS | int16: 4.53x, int32: 4.97x |
| exrdsp | PASS 2/2 | PASS | reorder_pixels: 10.66x, predictor: 5.35x |
| sbcdsp | PASS 6/6 | PASS 12/12 | sbc_analyze_4/8: 3.84–4.04x; calc_scalefactors: 5.88–7.44x |

---

## Open issues (non-blocking)

1. **`blockdsp.fill_block_tab[0]` KNOWN_REGRESSION**: 0.93x — annotated in baselines.json.
2. **`vp9_dc_128_8x8_8bpp_neon` KNOWN_REGRESSION**: sub-10-cycle PMU noise floor — annotated.
3. **`flac_lpc_33_*` C-only permanently**: 64-bit history; NEON correction overhead > gain.

---

## Design notes

### Why --csv and not default stdout?
Default checkasm `--bench` stdout interleaves bench data with diagnostic lines.
`--csv` gives clean `name,variant,cycles` triples. Filter: `^[^,]+,[^,]+,[0-9]+\.[0-9]+$`.

### Why parse stderr for Phase 0 and Phase 1?
- Phase 0: soft-timer fallback produces identical stdout format to hardware PMU
  but measures TSC ticks, not cycles. The only distinguishing signal is stderr.
- Phase 1: checkasm suppresses all bench output when `num_failed > 0`. An empty
  stdout could mean "no matching functions" or "correctness failed." Stderr
  disambiguates.

### Why exact string keys for function name lookup?
Names like `blockdsp.fill_block_tab[0]` contain literal brackets. Using
`re.compile(name)` for lookup would interpret `[0]` as a character class.
The validator uses `dict[name]` lookup throughout — no regex on function names.
