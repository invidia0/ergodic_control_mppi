"""Check every number asserted in Sec. V of the manuscript against the campaign CSVs.

Sec. V quotes roughly seventy numbers, all typed by hand from these files. Nothing else in
the repo would notice if one were mistranscribed, or if a re-flown campaign moved one out
from under the prose -- the manuscript lives in a separate Overleaf checkout, so a test
cannot reach it. This is the check: the claims are listed explicitly below, and a drift in
either the data or the typing shows up as a BAD line.

**Update this table in the same commit that changes a number in Sec. V.** A claim here that
no longer matches the manuscript is worse than no claim, because it reports OK.

One transcription error was caught this way on the first run (SVES open-tier tours given as
8 where the twelve-seed median is 7.5).

    uv run python scripts/verify_manuscript_numbers.py
"""
import csv, collections, importlib.util, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

spec = importlib.util.spec_from_file_location('fr', 'scripts/final_report.py')
fr = importlib.util.module_from_spec(spec); sys.modules['fr'] = fr; spec.loader.exec_module(fr)
rf = sys.modules['report_figures']

MANUSCRIPT = Path('69f1b707cd917a58478ed643/main.tex')
#: The manuscript is in the tree now, so a claim can be bound to the text that makes it
#: rather than only to the data. Numbers are typeset several ways, so a claim counts as
#: found if any rendering of it occurs: bare, with a thousands separator, or as a LaTeX
#: scientific literal. Absence is reported, never fatal -- a value can legitimately reach
#: the page through a generated table -- but an unfound claim is not verifying any prose.
_TEXT = MANUSCRIPT.read_text(encoding='utf-8') if MANUSCRIPT.is_file() else None


def _renderings(value: float) -> list[str]:
    """Every spelling of ``value`` this manuscript plausibly uses."""
    out = []
    for text in (f"{value:g}", f"{value:,g}", f"{abs(value):g}"):
        out.append(text)
        if text.startswith('0.'):
            out.append(text[1:])          # .55 as well as 0.55
    if value and abs(value) < 1e-3:
        exponent = int(np.floor(np.log10(abs(value))))
        mantissa = value / 10 ** exponent
        out += [f"{mantissa:.2f}", f"{mantissa:.3g}"]
    return out


def in_manuscript(claimed: float) -> bool:
    return _TEXT is not None and any(r in _TEXT for r in _renderings(claimed))


ok = bad = missing = 0
def check(label, claimed, actual, tol=0.005):
    global ok, bad, missing
    good = abs(claimed - actual) <= tol * max(1.0, abs(actual))
    found = in_manuscript(claimed)
    missing += not found
    mark = 'OK ' if good else 'BAD'
    print(f"  {mark} {label:<46} claimed {claimed:<12} actual {actual:.4g}"
          f"{'' if found else '   [not found in main.tex]'}")
    ok, bad = ok + good, bad + (not good)

t = rf.load_final(Path('results/uav/ablation_final.csv'))
recs = {r['arm']: r for r in fr.analyse(t)}
print("--- ablation ---")
check("arms", 37, len(recs), 0)
check("axes", 19, len(set(r['axis'] for r in recs.values())), 0)
check("cells per arm", 36, recs['T_150']['cells'], 0)
for arm, eff, sens in (("memory_off", -3.16, 15.1), ("plan_off", -0.94, 1.75),
                       ("ceiling_0", -0.34, 2.93), ("release_off", -0.09, 2.98),
                       ("transit_1", -0.54, 3.04), ("h_0.47", -0.36, 4.4),
                       ("h_5.0", -0.24, 6.7), ("alpha_0.9", -1.72, 5.8),
                       ("T_150", 0.55, 3.28), ("T_500", -0.20, 1.10),
                       ("T_750", -0.55, 2.48), ("ceiling_0.5", -0.03, 0.61)):
    check(f"{arm} effect", eff, recs[arm]['median_effect'], 0.02)
    check(f"{arm} sensitivity", sens, recs[arm]['sensitivity'], 0.02)
check("holm-significant sub-3sigma", 11,
      sum(1 for r in recs.values() if r['holm'] and r['sensitivity'] < 3.0), 0)
check("null verdicts", 20, sum(1 for r in recs.values() if r['verdict'] == 'null'), 0)
check("promotions", 1, sum(1 for r in recs.values() if r['verdict'] == 'promoted'), 0)
check("ceiling_0.5 agreement", 4, recs['ceiling_0.5']['agreement'], 0)
check("h_2.35 open effect", 0.14,
      {r['arm']: r for r in fr.analyse(rf.load_final(Path('results/uav/ablation_open.csv')))}['h_2.35']['median_effect'], 0.02)

# tours: baseline is 72 rows over two lane widths, so halve it for a 36-cell comparison
rows = list(csv.DictReader(open('results/uav/ablation_final.csv')))
by = collections.defaultdict(list)
for r in rows: by[r['arm']].append(r)
def tours(a):
    v = by[a]
    s = sum(int(float(r['all_modes_reached'])) + float(r['mode_cycles']) for r in v)
    return s / (len(v) / 36)
check("ceiling_0 tour collapse factor", 4.0, tours('baseline') / tours('ceiling_0'), 0.10)
mo = np.array([float(r['mode_cycles']) for r in by['memory_off']])
bl = np.array([float(r['mode_cycles']) for r in by['baseline'] if r['lanes'] == '36'])
check("memory_off median tours", 2, np.median(mo), 0)
check("baseline median tours", 5, np.median(bl), 0)

# T_150 path length and clearance
for m, claimed in (('path_length_m', 952), ('min_clearance_m', 0.95)):
    a, b, _ = rf.paired_final(t, 'T_150', m)
    check(f"T_150 {m}", claimed, np.median(a), 0.01)
for m, claimed in (('path_length_m', 678), ('min_clearance_m', 0.98)):
    a, b, _ = rf.paired_final(t, 'T_150', m)
    check(f"baseline {m}", claimed, np.median(b), 0.01)

print("--- baselines ---")
def load(p):
    d = collections.defaultdict(dict)
    for r in csv.DictReader(open(p)):
        d[r['method']][(r['map'], r['seed'])] = r
    return d
for tier, path, claims in (
    ('open', 'results/uav/baselines_open.csv',
     {'fmec': (2.57, -0.14), 'hedac': (1.39, 1.20), 'sves': (1.08, 0.04), 'smc': (-0.63, 1.37)}),
    ('clutter', 'results/uav/baselines_clutter.csv',
     {'fmec': (1.23, -0.52), 'hedac': (0.18, 0.61), 'sves': (0.70, 0.33), 'smc': (-1.24, 1.20)})):
    d = load(path); ours = d['ours']
    for m, (ce, co) in claims.items():
        cells = sorted(set(ours) & set(d[m]))
        for metric, claimed in (('fourier_ergodic', ce), ('occupancy_mse', co)):
            a = np.array([float(d[m][c][metric]) for c in cells])
            b = np.array([float(ours[c][metric]) for c in cells])
            check(f"{tier} {m} {metric}", claimed, np.median(np.log2(a / b)), 0.02)
    for m, claimed in (('sves', 36.1), ('smc', 27.8), ('fmec', 13.9), ('hedac', 8.3), ('ours', 0.0)):
        if tier != 'clutter': continue
        v = list(d[m].values())
        check(f"clutter {m} collision %", claimed,
              100 * sum(1 for r in v if int(r['collisions'])) / len(v), 0.02)
    if tier == 'clutter':
        for m, claimed in (('ours', 0.99), ('sves', 0.317), ('smc', 0.313), ('fmec', 0.335), ('hedac', 0.337)):
            check(f"clutter {m} clearance", claimed,
                  np.median([float(r['min_clearance_m']) for r in d[m].values()]), 0.02)
        check("clutter ours path m", 683, np.median([float(r['path_length_m']) for r in d['ours'].values()]), 0.01)
        check("clutter sves modes", 31, sum(int(float(r['all_modes_reached'])) for r in d['sves'].values()), 0)
    else:
        check("open ours tours", 6, np.median([float(r['mode_cycles']) for r in d['ours'].values()]), 0)
        check("open sves tours", 7.5, np.median([float(r['mode_cycles']) for r in d['sves'].values()]), 0)

print(f"\n{ok} verified, {bad} WRONG, {missing} not located in the manuscript text")
if _TEXT is None:
    print(f"NOTE: {MANUSCRIPT} is absent, so no claim was bound to the prose")
if bad:
    raise SystemExit(f"{bad} of {ok + bad} manuscript numbers do not match the data")
