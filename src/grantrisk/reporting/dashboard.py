"""The dashboard (SPEC-E3-01, DEC-11): one HTML file that works from the local disk, offline.

Every script and style is embedded; the data are the stored results, embedded as JSON.
Views: models (the sortable comparison table and, for the selected model run, its
confusion matrix, ROC curves, fold spread and error sizes), extraction, labels,
coverage and factor distributions, and provenance.
"""

from __future__ import annotations

import collections
import html
import json
from typing import Any

from grantrisk.labelling.scoring import FACTORS, LABELS
from grantrisk.reporting.chain import Data, sort_sources
from grantrisk.reporting.common import PERIODS, model_name
from grantrisk.reporting.tables import MODEL_DEFINITIONS

DISPLAY_DIGITS = 4  # the embedded numbers are rounded for display only; the stored results keep full precision


def _r(x: float | None) -> float | None:
    return None if x is None else round(float(x), DISPLAY_DIGITS)


def _key(key: tuple[str, str, str]) -> str:
    return "/".join(key)


def payload(data: Data, today: str) -> dict[str, Any]:
    """Everything the page shows, from the stored runs."""
    out: dict[str, Any] = {
        "created": today,
        "code_version": data.code_version,
        "n_documents": len(data.labels),
        "runs": [{"label": label, "run_id": run_id, "stage": data.run_info[label]["stage"],
                  "finished": data.run_info[label]["finished_at"], "code_version": data.run_info[label]["code_version"]}
                 for label, run_id in data.chain.runs().items()],
        "missing": data.chain.missing(),
        "labels_order": list(LABELS),
        "models": None, "extraction": None, "label_distributions": None,
    }
    if data.chain.e2 and data.comparison:
        details = {}
        for key, m in data.models.items():
            details[_key(key)] = {
                "name": model_name(key),
                "fold_f1": [[s, _r(v)] for s, v in m.fold_f1],
                "confusion": {t: {p: _r(v) for p, v in row.items()} for t, row in m.confusion.items()},
                "confusion_share": {t: {p: _r(v) for p, v in row.items()} for t, row in m.confusion_share.items()},
                "roc": {c: [_r(v) for v in m.roc.get(c, [])] for c in (*LABELS, "macro") if m.roc.get(c)},
                "roc_auc": {c: _r(v) for c, v in m.roc_auc.items()},
                "errors": {str(e): {"mean": _r(v["mean_count"]), "sd": _r(v["sd_count"])} for e, v in m.errors.items()},
            }
        grid = collections.defaultdict(dict)
        for g in data.grid:
            grid[g["scheme"]][f"{g['axis']}:{g['key']}"] = _r(g["f1_macro_mean"])
        out["models"] = {
            "definitions": MODEL_DEFINITIONS,
            "rows": [{"key": _key((r["scheme"], r["representation"], r["classifier"])), "scheme": r["scheme"],
                      "rank": r["rank"], "representation": r["representation"], "classifier": r["classifier"],
                      "f1": _r(r["f1_macro_mean"]), "f1_sd": _r(r["f1_macro_sd"]), "accuracy": _r(r["accuracy_mean"]),
                      "auc": _r(r["roc_auc_ovr_macro_mean"]), "kappa": _r(r["kappa_quadratic_mean"]),
                      "e2": _r(r["error2_share"]), "baseline": bool(r["is_baseline"]), "best": bool(r["is_best"]),
                      "below": bool(r["not_above_baseline"])} for r in data.comparison],
            "details": details,
            "grid": dict(grid),
            "best": _key(data.best) if data.best else None,
        }
        cells = collections.defaultdict(lambda: collections.defaultdict(dict))
        for r in data.label_distributions:
            cells[r["subset"]][r["tercile_label"]][r["fixed_label"]] = r["n"]
        out["label_distributions"] = {s: {t: dict(v) for t, v in row.items()} for s, row in cells.items()}
    if data.chain.e1 and data.e1_agreements:
        sources = sort_sources([r["source"] for r in data.e1_agreements])
        out["extraction"] = {
            "sources": sources,
            "factors": [*FACTORS, "all"],
            "cells": {s: {r["factor"]: {"agreement": _r(r["agreement"]), "low": _r(r["agreement_low"]),
                                        "high": _r(r["agreement_high"]), "agree": r["agree"],
                                        "n": r["agree"] + r["disagree"] + r["not_found"]}
                          for r in data.e1_agreements if r["source"] == s} for s in sources},
            "labels": [{"source": r["source"], "kind": r["label_kind"], "agreement": _r(r["agreement"]),
                        "low": _r(r["agreement_low"]), "high": _r(r["agreement_high"]), "agree": r["agree"],
                        "n": r["n_documents"], "kappa": _r(r["kappa_quadratic"]),
                        "errors": json.loads(r["error_sizes_json"])}
                       for r in sorted(data.e1_label_agreements,
                                       key=lambda r: (sources.index(r["source"]) if r["source"] in sources else 99,
                                                      r["label_kind"]))],
        }
    origins = (data.l3_report.get("coverage") or {}).get("by_factor") or {}
    points = {f: collections.Counter() for f in FACTORS}
    for (_, f), r in data.factor_points.items():
        points[f]["imputed" if r["origin"] == "mean" else str(int(r["points"]))] += 1
    out["coverage"] = {
        f: {"determined": origins.get(f, {}).get("band", 0) + origins.get(f, {}).get("manual", 0),
            "rule": origins.get(f, {}).get("top_rule", 0) + origins.get(f, {}).get("loan_rule", 0),
            "imputed": origins.get(f, {}).get("mean", 0),
            "points": {k: points[f].get(k, 0) for k in ("0", "1", "2", "3", "imputed")}}
        for f in FACTORS
    }
    out["periods"] = list(PERIODS)
    return out


STYLE = """
:root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--seq:222,62%;
--accent-bg:#cde2fb}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--surface:#1a1a19;
--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--accent-bg:#104281}}
:root[data-theme="dark"]{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--accent-bg:#104281}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:16px 16px 0;max-width:1200px;margin:auto}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:24px 0 8px}h3{font-size:14px;margin:16px 0 6px}
.sub{color:var(--ink2);margin:0 0 12px}
nav{display:flex;gap:4px;flex-wrap:wrap;border-bottom:1px solid var(--grid)}
nav button{background:none;border:0;border-bottom:2px solid transparent;color:var(--ink2);padding:8px 12px;
font:inherit;cursor:pointer}
nav button[aria-selected="true"]{color:var(--ink);border-bottom-color:var(--s1);font-weight:600}
main{max-width:1200px;margin:auto;padding:0 16px 32px}
section[hidden]{display:none}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:8px;padding:12px;margin:12px 0;overflow-x:auto}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.cards .card{margin:0}
table{border-collapse:collapse;font-variant-numeric:tabular-nums;width:100%}
th,td{padding:4px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th:first-child,td:first-child,.l{text-align:left}
th{color:var(--ink2);font-weight:600}
th.sort{cursor:pointer}th.sort::after{content:" ↕";color:var(--muted)}
tr.pick{cursor:pointer}tr.pick:hover{background:var(--grid)}
tr.sel{outline:2px solid var(--s1);outline-offset:-2px}
.tag{display:inline-block;padding:0 6px;border-radius:4px;font-size:12px;border:1px solid var(--ring);color:var(--ink2)}
.tag.best{border-color:var(--s1);color:var(--ink);font-weight:600}
.note{color:var(--ink2);font-size:13px;max-width:900px}
.heat td.v{color:var(--ink)}
svg text{fill:var(--ink2);font-size:11px}
svg .axis{stroke:var(--axis)}svg .gridl{stroke:var(--grid)}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;color:var(--ink2);margin:4px 0}
.legend i{display:inline-block;width:12px;height:3px;border-radius:2px;vertical-align:middle;margin-right:4px}
.tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--ring);border-radius:6px;
padding:6px 8px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.15);display:none;z-index:9}
select{font:inherit;color:var(--ink);background:var(--surface);border:1px solid var(--ring);border-radius:6px;padding:2px 6px}
footer{max-width:1200px;margin:auto;padding:0 16px 24px;color:var(--muted);font-size:12px}
"""

SCRIPT = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const L = D.labels_order, SERIES = {low:'var(--s1)', medium:'var(--s2)', high:'var(--s3)', macro:'var(--ink2)'};
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const f = (x, d = 3) => x === null || x === undefined ? '–' : Number(x).toFixed(d);
const tip = $('#tip');
function showTip(e, text) { tip.innerHTML = text; tip.style.display = 'block';
  tip.style.left = Math.min(e.clientX + 12, innerWidth - 220) + 'px'; tip.style.top = (e.clientY + 12) + 'px'; }
function hideTip() { tip.style.display = 'none'; }
// Sequential blue: a share in [0, 1] as a cell background; the number stays in the text ink.
const shade = x => x === null || x === undefined ? 'transparent' : `hsla(var(--seq),55%,${(0.08 + 0.55 * x).toFixed(3)})`;

// --- tabs
document.querySelectorAll('nav button').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('nav button').forEach(x => x.setAttribute('aria-selected', x === b));
  document.querySelectorAll('main section').forEach(s => s.hidden = s.id !== b.dataset.view);
}));

// --- models
let sortKey = 'rank', sortDir = 1, selected = D.models ? (D.models.best || D.models.rows[0].key) : null;
function modelTable(scheme) {
  const cols = [['rank','Rank'],['representation','Representation'],['classifier','Classifier'],['f1','macro-F1'],
    ['f1_sd','± SD'],['accuracy','Accuracy'],['auc','ROC-AUC'],['kappa','κw'],['e2','|e| = 2']];
  const rows = D.models.rows.filter(r => r.scheme === scheme).slice().sort((a, b) => {
    const x = a[sortKey], y = b[sortKey];
    if (x === y) return a.rank - b.rank;
    if (x === null) return 1; if (y === null) return -1;
    return (x < y ? -1 : 1) * sortDir; });
  let h = '<table><thead><tr>' + cols.map(([k, t]) => `<th class="sort" data-k="${k}">${t}</th>`).join('') +
    '<th class="l">Note</th></tr></thead><tbody>';
  for (const r of rows) {
    const note = r.best ? '<span class="tag best">best</span>' : r.baseline ? '<span class="tag">baseline</span>' :
      r.below ? '<span class="tag">not above the baseline</span>' : '';
    h += `<tr class="pick${r.key === selected ? ' sel' : ''}" data-key="${esc(r.key)}">` +
      cols.map(([k]) => `<td${k === 'representation' || k === 'classifier' ? ' class="l"' : ''}>${
        typeof r[k] === 'number' && k !== 'rank' ? f(r[k]) : esc(r[k] ?? '–')}</td>`).join('') + `<td class="l">${note}</td></tr>`;
  }
  return h + '</tbody></table>';
}
function renderModels() {
  if (!D.models) { $('#models').innerHTML = '<div class="card">Not available: no E2 run in this chain.</div>'; return; }
  const schemes = [...new Set(D.models.rows.map(r => r.scheme))];
  let h = `<p class="note">${esc(D.models.definitions)} Click a column to sort, a row to show its details.</p>`;
  for (const s of schemes) h += `<h2>${s === 'stratified' ? 'Model comparison (standard scheme)' :
    'Robustness check: ' + esc(s) + ' scheme'}</h2><div class="card" data-scheme="${esc(s)}">${modelTable(s)}</div>`;
  h += '<h2>Grid view: macro-F1, representation × classifier</h2><div class="card">' + gridTable() + '</div>';
  h += '<h2 id="sel-title"></h2><div class="cards"><div class="card" id="m-conf"></div><div class="card" id="m-roc"></div>' +
       '<div class="card" id="m-folds"></div><div class="card" id="m-err"></div></div>';
  $('#models').innerHTML = h;
  document.querySelectorAll('#models th.sort').forEach(th => th.addEventListener('click', () => {
    sortDir = sortKey === th.dataset.k ? -sortDir : (['rank','e2'].includes(th.dataset.k) ? 1 : -1);
    sortKey = th.dataset.k; renderModels(); }));
  document.querySelectorAll('#models tr.pick').forEach(tr => tr.addEventListener('click', () => {
    selected = tr.dataset.key; renderModels(); }));
  renderDetail();
}
function gridTable() {
  const g = D.models.grid.stratified || {}, rows = D.models.rows.filter(r => r.scheme === 'stratified');
  const reps = [...new Set(rows.map(r => r.representation))], clfs = [...new Set(rows.map(r => r.classifier))];
  const cell = (rep, clf) => (rows.find(r => r.representation === rep && r.classifier === clf) || {}).f1;
  let h = '<table class="heat"><thead><tr><th>Representation</th>' + clfs.map(c => `<th>${esc(c)}</th>`).join('') +
    '<th>Row mean</th></tr></thead><tbody>';
  for (const rep of reps) h += `<tr><td>${esc(rep)}</td>` + clfs.map(c => { const v = cell(rep, c);
    return `<td class="v" style="background:${shade(v)}" title="${esc(rep)}/${esc(c)}: ${f(v)}">${f(v)}</td>`; }).join('') +
    `<td><b>${f(g['representation:' + rep])}</b></td></tr>`;
  h += '<tr><td><b>Column mean</b></td>' + clfs.map(c => `<td><b>${f(g['classifier:' + c])}</b></td>`).join('') + '<td></td></tr>';
  return h + '</tbody></table><p class="note">Row means leave out the majority baseline; column means run over the representations.</p>';
}
function renderDetail() {
  const m = D.models.details[selected]; if (!m) return;
  $('#sel-title').textContent = 'Selected model run: ' + m.name;
  // Confusion matrix: counts divided by the number of repeats, and row shares.
  let h = '<h3>Confusion matrix (rows: tercile label; columns: predicted)</h3><table class="heat"><thead><tr><th></th>' +
    L.map(p => `<th>${p}</th>`).join('') + '</tr></thead><tbody>';
  for (const t of L) h += `<tr><td>${t}</td>` + L.map(p => `<td class="v" style="background:${shade(m.confusion_share[t][p])}">${
    f(m.confusion[t][p], 1)} <span class="note">(${f(100 * (m.confusion_share[t][p] ?? 0), 0)}%)</span></td>`).join('') + '</tr>';
  h += '</tbody></table><p class="note">Counts: the out-of-fold predictions of all repeats, divided by the number of repeats; they sum to the number of documents. In brackets: the share of the row (row-normalised).</p>';
  $('#m-conf').innerHTML = h;
  $('#m-roc').innerHTML = '<h3>ROC curves (one-vs-rest, pooled over the repeats)</h3>' + rocChart(m);
  $('#m-folds').innerHTML = '<h3>macro-F1 on each of the test folds</h3>' + foldChart(m);
  $('#m-err').innerHTML = '<h3>Error sizes (predicted − tercile label)</h3>' + errorChart(m);
  bindRocHover(m);
}
const W = 320, H = 240, P = {l: 40, r: 12, t: 10, b: 30};
function axes(xTicks, yTicks, sx, sy, xl, yl) {
  let s = '';
  for (const y of yTicks) s += `<line class="gridl" x1="${P.l}" x2="${W - P.r}" y1="${sy(y)}" y2="${sy(y)}"/><text x="${P.l - 6}" y="${sy(y) + 4}" text-anchor="end">${y}</text>`;
  for (const x of xTicks) s += `<text x="${sx(x)}" y="${H - P.b + 16}" text-anchor="middle">${x}</text>`;
  s += `<line class="axis" x1="${P.l}" x2="${W - P.r}" y1="${H - P.b}" y2="${H - P.b}"/>`;
  s += `<text x="${(P.l + W - P.r) / 2}" y="${H - 2}" text-anchor="middle">${xl}</text>`;
  s += `<text transform="translate(10 ${(P.t + H - P.b) / 2}) rotate(-90)" text-anchor="middle">${yl}</text>`;
  return s;
}
function rocChart(m) {
  const sx = x => P.l + x * (W - P.l - P.r), sy = y => H - P.b - y * (H - P.t - P.b);
  let s = `<svg id="roc" viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="ROC curves">` +
    axes([0, 0.5, 1], [0, 0.5, 1], sx, sy, 'false positive rate', 'true positive rate');
  s += `<line class="gridl" x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(1)}" stroke-dasharray="3 3"/>`;
  for (const c of [...L, 'macro']) { const ys = m.roc[c]; if (!ys) continue;
    const pts = ys.map((y, i) => `${sx(i / (ys.length - 1)).toFixed(1)},${sy(y).toFixed(1)}`).join(' ');
    s += `<polyline fill="none" stroke="${SERIES[c]}" stroke-width="2"${c === 'macro' ? ' stroke-dasharray="5 3"' : ''} points="${pts}"/>`; }
  s += `<line id="roc-x" class="axis" y1="${P.t}" y2="${H - P.b}" x1="-10" x2="-10"/><rect id="roc-hit" x="${P.l}" y="${P.t}" width="${W - P.l - P.r}" height="${H - P.t - P.b}" fill="transparent"/></svg>`;
  s += '<div class="legend">' + [...L, 'macro'].map(c => `<span><i style="background:${SERIES[c]}"></i>${c} (AUC ${f(m.roc_auc[c])})</span>`).join('') + '</div>';
  return s;
}
function bindRocHover(m) {
  const hit = $('#roc-hit'), line = $('#roc-x'); if (!hit) return;
  hit.addEventListener('mousemove', e => {
    const box = hit.getBoundingClientRect(), x = Math.max(0, Math.min(1, (e.clientX - box.left) / box.width));
    const n = (m.roc.macro || m.roc.low || []).length; if (!n) return;
    const i = Math.round(x * (n - 1)), px = P.l + (i / (n - 1)) * (W - P.l - P.r);
    line.setAttribute('x1', px); line.setAttribute('x2', px);
    showTip(e, `FPR ${f(i / (n - 1), 3)}<br>` + [...L, 'macro'].filter(c => m.roc[c]).map(c => `${c}: TPR ${f(m.roc[c][i])}`).join('<br>'));
  });
  hit.addEventListener('mouseleave', () => { hideTip(); line.setAttribute('x1', -10); line.setAttribute('x2', -10); });
}
function foldChart(m) {
  const vals = m.fold_f1.filter(v => v[1] !== null), n = vals.length;
  if (!n) return '<p>No fold values.</p>';
  const mean = vals.reduce((a, v) => a + v[1], 0) / n;
  const sd = n > 1 ? Math.sqrt(vals.reduce((a, v) => a + (v[1] - mean) ** 2, 0) / (n - 1)) : 0;
  const sx = i => P.l + (i + 0.5) * (W - P.l - P.r) / n, sy = y => H - P.b - y * (H - P.t - P.b);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="macro-F1 per fold">` +
    axes([], [0, 0.25, 0.5, 0.75, 1], sx, sy, 'test folds (repeat.fold)', 'macro-F1');
  s += `<rect x="${P.l}" width="${W - P.l - P.r}" y="${sy(Math.min(1, mean + sd))}" height="${Math.max(0, sy(Math.max(0, mean - sd)) - sy(Math.min(1, mean + sd)))}" fill="var(--accent-bg)" opacity=".6"/>`;
  s += `<line x1="${P.l}" x2="${W - P.r}" y1="${sy(mean)}" y2="${sy(mean)}" stroke="var(--s1)" stroke-width="2"/>`;
  vals.forEach(([scope, v], i) => { s += `<circle cx="${sx(i)}" cy="${sy(v)}" r="4" fill="var(--s1)" stroke="var(--surface)" stroke-width="2"><title>fold ${esc(scope)}: ${f(v)}</title></circle>`; });
  return s + `</svg><p class="note">Mean ${f(mean)} (line) ± SD ${f(sd)} (band, n − 1) over ${n} folds.</p>`;
}
function errorChart(m) {
  const keys = ['-2', '-1', '0', '+1', '+2'], vals = keys.map(k => (m.errors[k] || {}).mean ?? 0);
  const top = Math.max(1, ...vals) * 1.15, bw = (W - P.l - P.r) / keys.length;  // headroom for the value labels
  const sx = i => P.l + (i + 0.5) * bw, sy = y => H - P.b - y / top * (H - P.t - P.b);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="error sizes">` +
    axes([], [0, Math.round(top / 2), Math.floor(top)], sx, sy, 'error size', 'documents (mean over repeats)');
  keys.forEach((k, i) => { const v = vals[i], y = sy(v), sd = (m.errors[k] || {}).sd;
    s += `<rect x="${sx(i) - bw * 0.3}" width="${bw * 0.6}" y="${y}" height="${H - P.b - y}" rx="4" fill="${k === '0' ? 'var(--muted)' : 'var(--s1)'}"><title>error ${k}: ${f(v, 1)} documents (SD ${f(sd, 1)})</title></rect>`;
    s += `<text x="${sx(i)}" y="${H - P.b + 16}" text-anchor="middle">${k}</text><text x="${sx(i)}" y="${y - 4}" text-anchor="middle">${f(v, 1)}</text>`; });
  return s + '</svg><p class="note">Documents per error size, averaged over the repeats; 0 is a correct prediction.</p>';
}

// --- extraction
function renderExtraction() {
  const x = D.extraction, el = $('#extraction');
  if (!x) { el.innerHTML = '<div class="card">Not available: no E1 run in this chain.</div>'; return; }
  let h = '<h2>Agreement with the gold points, per factor and source</h2><p class="note">agree / (agree + disagree + not found) on the gold documents; brackets: 95% Wilson interval. Cell shade: the agreement.</p><div class="card"><table class="heat"><thead><tr><th>Factor</th>' +
    x.sources.map(s => `<th>${esc(s)}</th>`).join('') + '</tr></thead><tbody>';
  for (const fc of x.factors) h += `<tr><td>${fc === 'all' ? '<b>all factors</b>' : esc(fc)}</td>` + x.sources.map(s => {
    const c = (x.cells[s] || {})[fc]; if (!c) return '<td>–</td>';
    return `<td class="v" style="background:${shade(c.agreement)}" title="${esc(s)} / ${esc(fc)}: ${c.agree} of ${c.n}">${f(c.agreement)} <span class="note">[${f(c.low, 2)}, ${f(c.high, 2)}]</span></td>`;
  }).join('') + '</tr>';
  h += '</tbody></table></div><h2>Label-level agreement (SPEC-E1-04)</h2><p class="note">The label computed from each source\'s values against the label computed from the expert\'s points. Error: source − reference.</p><div class="card"><table><thead><tr><th>Source</th><th>Label</th><th>Agreement</th><th>κw</th><th>−2</th><th>−1</th><th>0</th><th>+1</th><th>+2</th></tr></thead><tbody>';
  for (const r of x.labels) h += `<tr><td>${esc(r.source)}</td><td>${r.kind}</td><td>${f(r.agreement)} (${r.agree}/${r.n}) [${f(r.low, 2)}, ${f(r.high, 2)}]</td><td>${f(r.kappa)}</td>` +
    ['-2', '-1', '0', '1', '2'].map(k => `<td>${r.errors[k] ?? r.errors[k.replace(/^(\d)/, '+$1')] ?? 0}</td>`).join('') + '</tr>';
  el.innerHTML = h + '</tbody></table></div>';
}

// --- labels
function renderLabels() {
  const el = $('#labels'), dist = D.label_distributions;
  if (!dist) { el.innerHTML = '<div class="card">Not available: no E2 run in this chain.</div>'; return; }
  const subsets = ['all', ...D.periods];
  let h = '<h2>Tercile and fixed label</h2><p class="note">Rows: the tercile label (the ML target); columns: the fixed-threshold label (documentation only). <label>Subset <select id="subset">' +
    subsets.map(s => `<option>${s}</option>`).join('') + '</select></label></p><div class="card" id="xtab"></div>';
  el.innerHTML = h;
  const draw = () => { const s = $('#subset').value, c = dist[s] || {};
    const get = (t, fx) => ((c[t] || {})[fx] || 0), n = L.reduce((a, t) => a + L.reduce((b, fx) => b + get(t, fx), 0), 0);
    let t = '<table class="heat"><thead><tr><th>Tercile \\ fixed</th>' + L.map(fx => `<th>${fx}</th>`).join('') + '<th>Total</th></tr></thead><tbody>';
    for (const tl of L) { const row = L.reduce((a, fx) => a + get(tl, fx), 0);
      t += `<tr><td>${tl}</td>` + L.map(fx => `<td class="v" style="background:${shade(n ? get(tl, fx) / n * 3 : 0)}">${get(tl, fx)}</td>`).join('') + `<td><b>${row}</b></td></tr>`; }
    t += '<tr><td><b>Total</b></td>' + L.map(fx => `<td><b>${L.reduce((a, tl) => a + get(tl, fx), 0)}</b></td>`).join('') + `<td><b>${n}</b></td></tr>`;
    $('#xtab').innerHTML = t + '</tbody></table>'; };
  $('#subset').addEventListener('change', draw); draw();
}

// --- coverage and factor distributions
function renderCoverage() {
  const el = $('#coverage'), cov = D.coverage, n = D.n_documents;
  const parts = [['determined', 'var(--s1)', 'determined (from a value or the expert)'], ['rule', 'var(--s2)', 'set by a rule (TOP rule DEC-31, loan rule DEC-40)'], ['imputed', 'var(--s3)', 'imputed (factor mean)']];
  const bw = 900, rh = 22, left = 120;
  let s = `<svg viewBox="0 0 ${bw + left + 50} ${Object.keys(cov).length * rh + 10}" width="100%" role="img" aria-label="coverage per factor">`;
  Object.entries(cov).forEach(([fc, c], i) => { let x = left; const y = 4 + i * rh;
    s += `<text x="${left - 6}" y="${y + 13}" text-anchor="end">${esc(fc)}</text>`;
    for (const [k, col, lab] of parts) { const w = n ? c[k] / n * bw : 0; if (w <= 0) continue;
      s += `<rect x="${x}" y="${y}" width="${Math.max(0, w - 2)}" height="${rh - 6}" rx="2" fill="${col}"><title>${esc(fc)}: ${lab} ${c[k]} of ${n}</title></rect>`; x += w; }
    s += `<text x="${left + bw + 6}" y="${y + 13}">${n ? f(c.determined / n * 100, 0) : '–'}%</text>`; });
  s += '</svg><div class="legend">' + parts.map(([, col, lab]) => `<span><i style="background:${col}"></i>${lab}</span>`).join('') + '</div>';
  let t = '<table class="heat"><thead><tr><th>Factor</th><th>0</th><th>1</th><th>2</th><th>3</th><th>imputed</th></tr></thead><tbody>';
  for (const [fc, c] of Object.entries(cov)) t += `<tr><td>${esc(fc)}</td>` + ['0', '1', '2', '3', 'imputed'].map(k =>
    `<td class="v" style="background:${shade(n ? c.points[k] / n : 0)}" title="${esc(fc)} ${k}: ${c.points[k]} of ${n}">${c.points[k]} <span class="note">(${n ? f(c.points[k] / n * 100, 0) : '–'}%)</span></td>`).join('') + '</tr>';
  el.innerHTML = `<h2>Coverage per factor (SPEC-L3-07)</h2><p class="note">How the points of each factor were set, over the ${n} documents of the L3 run; the percentage is the share determined.</p><div class="card">${s}</div>` +
    `<h2>Distribution of points per factor</h2><p class="note">Documents per point value; a near-constant factor carries little information (ISS-15).</p><div class="card">${t}</tbody></table></div>`;
}

// --- provenance
function renderProvenance() {
  let h = '<h2>Runs this page was built from</h2><div class="card"><table><thead><tr><th>Part</th><th class="l">Run</th><th class="l">Finished</th><th class="l">Code version</th></tr></thead><tbody>';
  for (const r of D.runs) h += `<tr><td>${esc(r.label)}</td><td class="l"><code>${esc(r.run_id)}</code></td><td class="l">${esc(r.finished || '')}</td><td class="l"><code>${esc(r.code_version)}</code></td></tr>`;
  h += '</tbody></table></div>';
  h += '<h2>Parts not in this chain</h2><div class="card">' + (D.missing.length ? '<ul>' + D.missing.map(m => `<li>${esc(m)}</li>`).join('') + '</ul>' : 'None.') + '</div>';
  $('#provenance').innerHTML = h;
}
renderModels(); renderExtraction(); renderLabels(); renderCoverage(); renderProvenance();
"""


def build(data: Data, today: str) -> str:
    """The HTML text of the dashboard."""
    body_json = json.dumps(payload(data, today), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    runs = ", ".join(f"{label} {run_id}" for label, run_id in data.chain.runs().items())
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Grant Risk Dashboard</title>
<style>{STYLE}</style>
</head>
<body>
<header>
<h1>Grant Risk Estimator: results</h1>
<p class="sub">{len(data.labels)} documents. The ML reference is the tercile label (DEC-07); every number comes from the stored runs named under Provenance.</p>
<nav role="tablist">
<button data-view="models" aria-selected="true">Models</button>
<button data-view="extraction" aria-selected="false">Extraction</button>
<button data-view="labels" aria-selected="false">Labels</button>
<button data-view="coverage" aria-selected="false">Coverage</button>
<button data-view="provenance" aria-selected="false">Provenance</button>
</nav>
</header>
<main>
<section id="models"></section>
<section id="extraction" hidden></section>
<section id="labels" hidden></section>
<section id="coverage" hidden></section>
<section id="provenance" hidden></section>
</main>
<div class="tip" id="tip"></div>
<script type="application/json" id="data">{body_json}</script>
<script>{SCRIPT}</script>
<footer>Provenance: built from {html.escape(runs)}. Code version {html.escape(data.code_version)}. Created {today}.</footer>
</body>
</html>
"""
