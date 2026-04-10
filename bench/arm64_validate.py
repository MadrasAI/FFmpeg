#!/usr/bin/env python3
"""
bench/arm64_validate.py — ARM64 NEON validation harness for FFmpeg

Three-phase validation for NEON implementations on Graviton3/AArch64:

  Phase 0: Verify hardware PMU backend (CONFIG_LINUX_PERF + perf_event_paranoid=-1)
           Soft-timer fallback produces the same output format as hardware PMU but
           measures nanoseconds/TSC ticks — silently wrong for baseline comparison.

  Phase 1: Correctness — detect test failures before benchmarking.
           checkasm suppresses bench output entirely when any correctness test fails.
           This phase detects that case via stderr rather than treating empty stdout
           as a missing-benchmark condition.

  Phase 2: Performance — parse --csv bench output, compare against baselines.json.
           Uses --csv flag (not default stdout) for clean field separation.
           Function name lookup uses exact string keys — bracket notation like
           fill_block_tab[0] is safe because we are not using re.compile on names.

  Phase 3: Result output — emit structured JSON for Veda result block.

Usage:
    python3 bench/arm64_validate.py \\
        --checkasm-path tests/checkasm/checkasm \\
        [--module audiodsp] \\
        [--threshold 0.80] \\
        [--update-baseline] \\
        [--json-only]

    # Via Makefile target (preferred):
    make bench-validate MODULE=audiodsp

Requirements:
    sudo sysctl -w kernel.perf_event_paranoid=-1
    (Must be set before configuring FFmpeg for CONFIG_LINUX_PERF to activate)
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERF_BACKEND_HW   = "benchmarking with Linux Perf Monitoring API"
PERF_BACKEND_SOFT = "benchmarking with native FFmpeg timers"
PERF_BACKEND_NONE = "checkasm: --bench is not supported on your system"

# Match correctness failure lines emitted by checkasm to stderr
FAIL_RE = re.compile(r'(\d+) of \d+ tests? (?:have )?failed')

# Match CSV bench output lines: name,variant,cycles
# Using exact field split — NOT re.compile on name — to handle bracket notation safely.
CSV_LINE_RE = re.compile(r'^([^,\s][^,]*),([^,]+),([0-9]+\.[0-9]+)$')

# Cycle count tolerance for baseline comparison (new > baseline * (1 + TOLERANCE) = regression)
DEFAULT_TOLERANCE = 0.20   # 20% regression ceiling
DEFAULT_THRESHOLD = 0.80   # minimum NEON speedup ratio vs C

# ---------------------------------------------------------------------------
# Module bench-pattern map
#
# Some modules have a test-group name (used by --test=) that differs from the
# registered check_func names (used by --bench=). Without this map the harness
# would pass --bench=<module>* and match nothing.
#
# Each value is a list of bench patterns to pass to run_checkasm, one invocation
# per pattern. An empty string "" means "no --bench= filter" (match all bench
# functions for the module) — used when function names have no common prefix.
#
# Add entries here whenever a new module is ported and its bench names diverge
# from the test group name.
# ---------------------------------------------------------------------------

MODULE_BENCH_PATTERNS: dict[str, list[str]] = {
    "alacdsp":  ["alac_"],   # test group "alacdsp" but bench names are "alac_*"
    "lpc":      ["apply_welch_window", "autocorr"],
    "llviddsp": ["add_"],
    "flacdsp":  ["flac_"],   # test group "flacdsp" but bench names are "flac_*"
    "vp3dsp":   ["v_loop_filter", "h_loop_filter"],  # bench names are v_loop_filter*/h_loop_filter*
    "vf_blend": [""],        # blend mode names (addition, average, …) have no shared prefix
    "mlpdsp":   ["mlp_"],    # test group "mlpdsp" but bench names are "mlp_filter_channel*"
    "takdsp":   ["decorrelate_"],
    "sbcdsp":   ["sbc_analyze_", "calc_scalefactors_"],
    "hpeldsp":  ["put_pixels_tab", "avg_pixels_tab", "put_no_rnd_pixels_tab", "avg_no_rnd_pixels_tab"],
    "dcadsp":   ["lfe_fir"],
    "g722dsp":  ["g722_apply_qmf"],
    "utvideodsp": ["restore_rgb_planes"],
    "llauddsp": ["scalarproduct_and_madd"],
    "exrdsp":         ["reorder_pixels", "predictor"],
    "huffyuvencdsp":  ["sub_hfyu_median_pred_int16"],
    "llvidencdsp":    ["diff_bytes", "sub_left_predict", "sub_median_pred"],
    "float_dsp":      ["vector_", "butterflies_", "scalarproduct_"],
    "rv34dsp":        ["rv34_idct_dc_add"],   # rv34_inv_transform_dc kept C-only (0.45x on AArch64 scalar)
    "fixed_dsp":      ["vector_fmul_fixed", "vector_fmul_add_fixed", "vector_fmul_reverse_fixed",
                        "vector_fmul_window_fixed", "vector_fmul_window_scaled_fixed",
                        "butterflies_fixed", "scalarproduct_fixed"],
    "motion":         ["hadamard8_diff"],
    "mpegvideoencdsp": ["pix_sum", "pix_norm1", "denoise_dct", "add_8x8basis", "draw_edges_"],
}

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


# ---------------------------------------------------------------------------
# Baselines I/O
# ---------------------------------------------------------------------------

def load_baselines(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_baselines(path: str, data: dict) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
        f.write('\n')


# ---------------------------------------------------------------------------
# checkasm invocation
# ---------------------------------------------------------------------------

def run_checkasm(
    checkasm_path: str,
    module: str | None,
    bench_pattern: str | None = None,
) -> tuple[str, str, int]:
    """Invoke checkasm with --bench --csv, optionally filtered to a module prefix.

    checkasm flag semantics:
      --test=<pattern>   run correctness tests matching wildcard pattern
      --bench=<pattern>  run benchmarks matching wildcard pattern (implies correctness first)
      --csv              emit bench results as name,variant,cycles lines on stdout

    Combining --test and --bench with the same pattern runs correctness then bench
    in one pass. Bench output is suppressed by checkasm if any correctness test fails.

    bench_pattern overrides the bench filter when the test-group name (used by --test=)
    differs from the check_func registered name (used by --bench=). Example: lpc tests
    are grouped as "lpc" but bench functions are named "apply_welch_window_*"/"autocorr_*".
    """
    cmd = [checkasm_path]
    if module:
        cmd.append(f'--test={module}*')
        if bench_pattern is None:
            # Default: use module name as bench pattern
            cmd.append(f'--bench={module}*')
        elif bench_pattern == "":
            # Empty sentinel: bench all functions (no pattern filter)
            cmd.append('--bench')
        else:
            cmd.append(f'--bench={bench_pattern}*')
    else:
        cmd.append('--bench')
    cmd.append('--csv')

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode


# ---------------------------------------------------------------------------
# Phase 0 — hardware PMU backend verification
# ---------------------------------------------------------------------------

def check_perf_backend(stderr: str) -> tuple[bool, str]:
    """Return (ok, backend_description_line).

    The hardware PMU backend emits exactly:
        "benchmarking with Linux Perf Monitoring API"  (to stderr)
    The soft-timer fallback emits:
        "benchmarking with native FFmpeg timers"

    Both produce identical output format on stdout — the only distinction is this
    stderr line. Without hardware PMU, cycle counts are TSC ticks or nanoseconds
    and must NOT be compared against hardware-PMU baselines.
    """
    for line in stderr.splitlines():
        stripped = line.strip()
        if PERF_BACKEND_HW in stripped:
            return True, stripped
        if PERF_BACKEND_SOFT in stripped:
            return False, stripped
        if PERF_BACKEND_NONE in stripped:
            return False, stripped
    return False, "(no benchmarking backend line detected in checkasm stderr)"


# ---------------------------------------------------------------------------
# Phase 1 — correctness check
# ---------------------------------------------------------------------------

def check_correctness(stderr: str) -> tuple[bool, str | None]:
    """Detect correctness failures from checkasm stderr.

    checkasm suppresses all bench output when num_failed > 0.
    Parsing stderr for the failure count is more reliable than checking for
    empty stdout (which could also mean no matching functions).
    """
    m = FAIL_RE.search(stderr)
    if m:
        return False, m.group(0)
    return True, None


# ---------------------------------------------------------------------------
# Phase 2 — CSV parse + baseline comparison
# ---------------------------------------------------------------------------

def parse_csv_bench(stdout: str) -> dict[str, dict[str, float]]:
    """Parse checkasm --csv bench output into {func_name: {variant: cycles}}.

    CSV format from checkasm source (checkasm.c print_benchs):
        printf("%s%c%s%c%.1f\\n", f->name, sep, cpu_suffix(v->cpu), sep, decicycles/10.0)

    f->name is the literal registered name (may contain brackets, dots, underscores).
    cpu_suffix returns "c" for cpu=0, "neon" for AV_CPU_FLAG_NEON, etc.

    We use exact string keys — no re.compile on func_name — so bracket notation
    like "fill_block_tab[0]" is safe.
    """
    results: dict[str, dict[str, float]] = {}
    for line in stdout.splitlines():
        m = CSV_LINE_RE.match(line.strip())
        if not m:
            continue
        name, variant, cycles_str = m.group(1), m.group(2), m.group(3)
        if name in results and variant in results[name]:
            import warnings
            warnings.warn(
                f"Duplicate bench result for {name}/{variant} across pattern runs; keeping first value",
                stacklevel=2,
            )
            continue
        results.setdefault(name, {})[variant] = float(cycles_str)
    return results


def compare_against_baselines(
    measured: dict[str, dict[str, float]],
    baselines: dict,
    threshold: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[dict]:
    """Compare measured results against baselines.json entries.

    For each (function, variant) pair:
      - Cycle regression: new_cycles > baseline_cycles * (1 + tolerance) → REGRESSED
      - Threshold check:  live neon ratio < threshold → BELOW_THRESHOLD
                          (skipped for entries with skip_threshold=true)
      - Known regression: baseline entry has regression=true → KNOWN_REGRESSION (not a fail)

    Returns list of row dicts, one per (function, variant).
    """
    rows = []
    functions = baselines.get('functions', {})

    for func_name in sorted(measured.keys()):
        variants = measured[func_name]
        func_baseline = functions.get(func_name, {})

        # Compute live speedup ratio from this run (c_cycles / neon_cycles)
        c_cycles  = variants.get('c')
        neon_cycles = variants.get('neon')
        live_ratio = (c_cycles / neon_cycles) if (c_cycles and neon_cycles) else None

        for variant in sorted(variants.keys()):
            new_cycles = variants[variant]
            baseline_entry = func_baseline.get(variant)

            row = {
                'func':             func_name,
                'variant':          variant,
                'new_cycles':       new_cycles,
                'baseline_cycles':  None,
                'baseline_ratio':   None,
                'live_ratio':       live_ratio if variant == 'neon' else None,
                'skip_threshold':   False,
                'status':           'OK',
                'note':             None,
            }

            if baseline_entry:
                row['baseline_cycles'] = baseline_entry.get('cycles')
                row['baseline_ratio']  = baseline_entry.get('ratio')
                row['skip_threshold']  = baseline_entry.get('skip_threshold', False)

                # Known regression annotation — informational, not a gate failure
                if baseline_entry.get('regression'):
                    row['status'] = 'KNOWN_REGRESSION'
                    row['note']   = baseline_entry.get('regression_note', 'known regression')

                # Cycle regression check: new cycles more than tolerance above baseline
                if row['baseline_cycles'] and new_cycles > row['baseline_cycles'] * (1 + tolerance):
                    row['status'] = 'REGRESSED'
                    row['note']   = (
                        f"new={new_cycles:.1f} > baseline={row['baseline_cycles']:.1f} "
                        f"* {1+tolerance:.2f} = {row['baseline_cycles'] * (1+tolerance):.1f}"
                    )
            else:
                row['status'] = 'NO_BASELINE'

            # Threshold check for neon variants (skip if annotated or no live ratio)
            if (variant == 'neon'
                    and live_ratio is not None
                    and not row['skip_threshold']
                    and row['status'] == 'OK'):
                if live_ratio < threshold:
                    row['status'] = 'BELOW_THRESHOLD'
                    row['note']   = f"speedup={live_ratio:.2f}x < threshold={threshold:.2f}x"

            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Phase 3 — reporting
# ---------------------------------------------------------------------------

def print_report(rows: list[dict], module: str | None, threshold: float) -> list[dict]:
    """Print human-readable validation report. Returns list of gate-failure rows."""
    label = module or 'all modules'
    print(f"\n{BOLD}ARM64 NEON Validation — {label}{RESET}")
    print("=" * 72)

    gate_failures = [r for r in rows if r['status'] in ('REGRESSED', 'BELOW_THRESHOLD')]
    known_regressions = [r for r in rows if r['status'] == 'KNOWN_REGRESSION']

    col_w = 56
    for r in rows:
        func_str   = f"{r['func']}_{r['variant']}"
        cycles_str = f"{r['new_cycles']:>8.1f}"

        if r['baseline_cycles']:
            pct   = (r['new_cycles'] / r['baseline_cycles'] - 1.0) * 100
            delta = f"  ({pct:+.1f}% vs baseline)"
        else:
            delta = "  (no baseline)"

        ratio_str = f"  [{r['live_ratio']:.2f}x]" if r['live_ratio'] else ""

        status_colour = {
            'OK':               GREEN,
            'KNOWN_REGRESSION': YELLOW,
            'REGRESSED':        RED,
            'BELOW_THRESHOLD':  RED,
            'NO_BASELINE':      DIM,
        }.get(r['status'], RESET)

        status_str = f"{status_colour}{r['status']}{RESET}"
        print(f"  {func_str:<{col_w}} {cycles_str}{ratio_str}{delta}  {status_str}")
        if r['note'] and r['status'] != 'OK':
            print(f"    {DIM}{r['note']}{RESET}")

    counts = {s: sum(1 for r in rows if r['status'] == s) for s in
              ('OK', 'KNOWN_REGRESSION', 'REGRESSED', 'BELOW_THRESHOLD', 'NO_BASELINE')}

    print()
    print(f"  {GREEN}PASS: {counts['OK']}{RESET}  "
          f"{YELLOW}KNOWN_REGRESSION: {counts['KNOWN_REGRESSION']}{RESET}  "
          f"{RED}REGRESSED: {counts['REGRESSED']}  BELOW_THRESHOLD: {counts['BELOW_THRESHOLD']}{RESET}  "
          f"{DIM}NO_BASELINE: {counts['NO_BASELINE']}{RESET}")
    print(f"  threshold={threshold:.2f}x  tolerance=+{DEFAULT_TOLERANCE*100:.0f}%")
    print()

    return gate_failures


def build_result_json(
    rows: list[dict],
    module: str | None,
    backend_line: str,
    correctness_ok: bool,
    gate_failures: list[dict],
) -> dict:
    outcome = "SUCCESS" if (correctness_ok and not gate_failures) else "FAILURE"
    pass_count = sum(1 for r in rows if r['status'] == 'OK')
    total = len(rows)

    return {
        "result": {
            "attempt_description": f"ARM64 NEON validation — {module or 'all modules'}",
            "outcome": outcome,
            "exact_parameters": {
                "module_pattern": module,
                "perf_backend":   backend_line,
                "threshold":      DEFAULT_THRESHOLD,
                "tolerance":      DEFAULT_TOLERANCE,
            },
            "what_worked": f"{pass_count}/{total} functions within threshold" if pass_count else None,
            "what_failed": (
                "; ".join(f"{r['func']}_{r['variant']}: {r['note']}" for r in gate_failures)
                if gate_failures else None
            ),
            "error_message": None,
            "test_results": {
                "correctness":  "PASS" if correctness_ok else "FAIL",
                "performance":  ("PASS" if not gate_failures
                                 else f"FAIL — {len(gate_failures)} regression(s)"),
                "perf_backend": backend_line,
            },
        }
    }


# ---------------------------------------------------------------------------
# Baseline update
# ---------------------------------------------------------------------------

def update_baselines(baselines: dict, measured: dict, path: str) -> None:
    """Promote measured results to new baseline entries, preserving annotations."""
    functions = baselines.setdefault('functions', {})
    baselines['_meta']['last_updated'] = datetime.now().strftime('%Y-%m-%d')

    for func_name, variants in measured.items():
        c_cycles = variants.get('c')
        functions.setdefault(func_name, {})

        for variant, cycles in variants.items():
            existing = functions[func_name].get(variant, {})
            new_entry: dict = {'cycles': round(cycles, 1)}

            if variant == 'neon' and c_cycles:
                new_entry['ratio'] = round(c_cycles / cycles, 3)

            # Preserve existing regression/skip annotations verbatim
            for key in ('regression', 'regression_note', 'skip_threshold', 'skip_note'):
                if key in existing:
                    new_entry[key] = existing[key]

            functions[func_name][variant] = new_entry

    save_baselines(path, baselines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='ARM64 NEON validation harness — correctness + perf comparison + result output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--checkasm-path', required=True,
                        help='Path to checkasm binary (e.g. tests/checkasm/checkasm)')
    parser.add_argument('--baselines', default='bench/baselines.json',
                        help='Path to baselines JSON (default: bench/baselines.json)')
    parser.add_argument('--module', default=None,
                        help='Module/function prefix to validate (e.g. audiodsp, blockdsp, flac)')
    parser.add_argument('--bench-pattern', default=None, dest='bench_pattern',
                        help='Explicitly override bench function prefix. Known modules resolve '
                             'automatically via MODULE_BENCH_PATTERNS; only needed for new or '
                             'unregistered modules where test-group name differs from bench names.')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help=f'Minimum NEON speedup ratio vs C (default: {DEFAULT_THRESHOLD})')
    parser.add_argument('--update-baseline', action='store_true',
                        help='After successful validation, promote results to new baseline')
    parser.add_argument('--json-only', action='store_true',
                        help='Suppress table output, print only the JSON result block')
    args = parser.parse_args()

    # Pre-flight: verify inputs exist
    if not os.path.isfile(args.checkasm_path):
        print(f"ERROR: checkasm not found: {args.checkasm_path}", file=sys.stderr)
        print("Build it first:  make tests/checkasm/checkasm", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.baselines):
        print(f"ERROR: baselines file not found: {args.baselines}", file=sys.stderr)
        sys.exit(1)

    baselines = load_baselines(args.baselines)

    if not args.json_only:
        print(f"checkasm: {args.checkasm_path}")
        print(f"baselines: {args.baselines}")
        print(f"module: {args.module or 'all'}  threshold: {args.threshold:.2f}x")
        print("Running checkasm...", flush=True)

    # Resolve bench patterns: explicit override > MODULE_BENCH_PATTERNS > default (module name)
    if args.bench_pattern is not None:
        patterns: list[str | None] = [args.bench_pattern]
    elif args.module and args.module in MODULE_BENCH_PATTERNS:
        patterns = MODULE_BENCH_PATTERNS[args.module]
    else:
        patterns = [None]   # None → run_checkasm uses module name as bench pattern

    # Run one checkasm invocation per pattern and merge stdout + stderr
    all_stdout: list[str] = []
    all_stderr: list[str] = []
    for pat in patterns:
        out, err, _ = run_checkasm(args.checkasm_path, args.module, pat)
        all_stdout.append(out)
        all_stderr.append(err)

    stdout = "\n".join(all_stdout)
    stderr = "\n".join(all_stderr)

    # ── Phase 0: hardware PMU backend ──────────────────────────────────────
    backend_ok, backend_line = check_perf_backend(stderr)
    if not backend_ok:
        print(f"\n{RED}ERROR: Hardware PMU backend not active.{RESET}", file=sys.stderr)
        print(f"  Detected:  {backend_line}", file=sys.stderr)
        print(f"  Cycle counts are not comparable to hardware-PMU baselines.", file=sys.stderr)
        print(f"  Fix:  sudo sysctl -w kernel.perf_event_paranoid=-1", file=sys.stderr)
        print(f"  Then reconfigure FFmpeg (CONFIG_LINUX_PERF is set at configure time).", file=sys.stderr)
        sys.exit(2)

    # ── Phase 1: correctness ───────────────────────────────────────────────
    correctness_ok, fail_msg = check_correctness(stderr)
    if not correctness_ok:
        print(f"\n{RED}ERROR: Correctness failures: {fail_msg}{RESET}", file=sys.stderr)
        print("  checkasm suppresses bench output on failure. Fix correctness first.", file=sys.stderr)
        print(f"\n--- checkasm stderr ---\n{stderr}--- end ---", file=sys.stderr)
        sys.exit(3)

    # ── Phase 2: parse + compare ───────────────────────────────────────────
    measured = parse_csv_bench(stdout)
    if not measured:
        print(f"\n{YELLOW}WARNING: No bench output parsed.{RESET}", file=sys.stderr)
        print("  Possible causes:", file=sys.stderr)
        print("  - Module pattern matched no functions", file=sys.stderr)
        print("  - checkasm built without the relevant CONFIG_ flags", file=sys.stderr)
        print(f"\n--- checkasm stderr ---\n{stderr}--- end ---", file=sys.stderr)
        sys.exit(4)

    rows = compare_against_baselines(measured, baselines, args.threshold)

    # ── Phase 3: report ────────────────────────────────────────────────────
    if not args.json_only:
        gate_failures = print_report(rows, args.module, args.threshold)
    else:
        gate_failures = [r for r in rows if r['status'] in ('REGRESSED', 'BELOW_THRESHOLD')]

    result = build_result_json(rows, args.module, backend_line, correctness_ok, gate_failures)
    print(json.dumps(result, indent=2))

    # ── Baseline update ────────────────────────────────────────────────────
    if args.update_baseline:
        if gate_failures:
            print(f"\n{RED}Cannot update baseline: {len(gate_failures)} gate failure(s) present.{RESET}",
                  file=sys.stderr)
            sys.exit(1)
        update_baselines(baselines, measured, args.baselines)
        if not args.json_only:
            print(f"\n{GREEN}Baseline updated: {args.baselines}{RESET}")

    sys.exit(1 if gate_failures else 0)


if __name__ == '__main__':
    main()
