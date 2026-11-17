// Loads results.json (written by run_pipeline.py) and fills the page. No private data, no API keys.
const $ = (id) => document.getElementById(id);
const fmt = (x, d = 3) => Number(x).toFixed(d);
const pct = (x) => Number(x).toFixed(1) + "%";
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const money = (x) => "$" + Math.round(x).toLocaleString("en-US");
let R, weights = [], charts = {};

fetch("results.json").then((r) => r.json()).then(init).catch((e) => {
  document.querySelector("main").insertAdjacentHTML("afterbegin",
    `<p class="banner">Could not load results.json (${e}). Run <code>python run_pipeline.py</code> and refresh. If you opened this file directly, serve it with <code>python -m http.server</code> from the docs folder.</p>`);
});

function row(cells, cls = "") {
  return `<tr class="${cls}">` + cells.map((c, i) => `<td class="${i ? "num" : ""}">${c}</td>`).join("") + "</tr>";
}
function bars(el, items, valueKey, labelKey, digits = 4) {
  const max = Math.max(...items.map((i) => Math.abs(i[valueKey])), 1e-12);
  el.innerHTML = items.map((i) => `<div class="bar-row"><span>${i[labelKey]}</span>
    <div><div class="bar ${i[valueKey] < 0 ? "neg" : ""}" style="width:${(Math.abs(i[valueKey]) / max) * 100}%"></div></div>
    <span>${fmt(i[valueKey], digits)}</span></div>`).join("");
}
function lineChart(id, labels, sets) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart($(id), {
    type: "line",
    data: { labels, datasets: sets.map((s) => ({ ...s, pointRadius: 0, borderWidth: 1.6, tension: 0.15 })) },
    options: { responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      scales: { x: { ticks: { maxTicksLimit: 8, color: css("--muted") }, grid: { display: false } },
                y: { title: { display: true, text: "Annualised volatility (%)", color: css("--muted") }, ticks: { color: css("--muted") } } },
      plugins: { legend: { labels: { color: css("--ink") } } } },
  });
}

function init(d) {
  R = d;
  const m = d.meta;
  if (m.synthetic_demo) {
    $("demo-banner").hidden = false;
    $("demo-banner").textContent = "DEMO: these results come from synthetic data used to test the code. Run python run_pipeline.py to replace them with real results.";
  }
  document.querySelectorAll(".v-train").forEach((e) => (e.textContent = m.split.train_end.slice(0, 4)));
  $("generated").textContent = `Results generated ${m.generated}. Data: ${m.data_source}.`;
  $("data-text").textContent = `${m.tickers.length} large US stocks (${m.tickers.join(", ")}) plus the SPY index, ${m.quality.first_date} to ${m.quality.last_date} (${m.quality.rows} trading days). Each row of the model table is one stock on one day: ${m.split.n_train} training rows, ${m.split.n_val} validation rows and ${m.split.n_test} test rows.`;
  $("quality-table").querySelector("tbody").innerHTML = Object.entries(m.quality)
    .filter(([k]) => !["first_date", "last_date"].includes(k))
    .map(([k, v]) => row([k.replace(/_/g, " "), v])).join("");

  // results
  const best = m.best_model;
  const byName = Object.fromEntries(d.metrics.map((x) => [x.model, x]));
  const pers = byName["Persistence (last 21d vol)"], har = byName["HAR (3 vol lags, OLS)"], b = byName[best];
  $("results-text").textContent = `The model chosen on validation data was ${best}. On the test period its RMSE is ${fmt(b.rmse)} versus ${fmt(pers.rmse)} for persistence and ${fmt(har.rmse)} for the HAR benchmark. ` +
    (b.rmse < har.rmse ? `It beats both baselines.` : b.rmse < pers.rmse ? `It beats persistence but not the simpler HAR benchmark, so the extra complexity is not clearly paying off.` : `It does not beat persistence, so a learned model adds little here.`);
  $("metrics-table").querySelector("tbody").innerHTML = d.metrics.map((x) =>
    row([x.model + (x.model === best ? " (chosen)" : ""), fmt(x.val_rmse), fmt(x.rmse), fmt(x.mae), fmt(x.r2), (x.skill_vs_persistence * 100).toFixed(1) + "%"], x.model === best ? "best" : "")).join("");
  charts.metrics = new Chart($("metrics-chart"), {
    type: "bar",
    data: { labels: d.metrics.map((x) => x.model), datasets: [{ label: "Test RMSE (lower is better)", data: d.metrics.map((x) => x.rmse),
      backgroundColor: d.metrics.map((x) => (x.model === best ? css("--ml") : css("--base"))) }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, ticks: { color: css("--muted") } }, x: { ticks: { color: css("--muted") } } } },
  });
  bars($("importance"), d.rf_importance, "importance", "feature", 5);

  // stock explorer
  $("ticker").innerHTML = m.tickers.map((t) => `<option>${t}</option>`).join("");
  $("ticker").addEventListener("change", drawStock);
  drawStock();

  // errors
  $("regime-table").querySelector("tbody").innerHTML = d.by_regime.map((r) => row([r.regime, r.n, fmt(r.rmse_model), fmt(r.rmse_persistence)])).join("");
  $("ticker-table").querySelector("tbody").innerHTML = d.per_ticker.map((r) => row([r.ticker, fmt(r.rmse_model), fmt(r.rmse_persistence)])).join("");
  $("worst-table").querySelector("tbody").innerHTML = d.worst_errors.map((r) => row([r.date, r.ticker, pct(r.actual), pct(r.model), pct(r.persistence)])).join("");

  initPortfolio();
  initBacktest();
}

function drawStock() {
  const s = R.series[$("ticker").value];
  lineChart("stock-chart", s.dates, [
    { label: "Actual (next 21 days)", data: s.actual, borderColor: css("--real") },
    { label: "ML forecast", data: s.model, borderColor: css("--ml") },
    { label: "Persistence (last 21 days)", data: s.persistence, borderColor: css("--base"), borderDash: [5, 4] },
  ]);
}

// ---------- portfolio ----------
function initPortfolio() {
  const c = R.current, t = c.tickers;
  $("asof").textContent = `Forecasts made from data up to ${c.as_of}, for the following 21 trading days.`;
  $("sliders").innerHTML = t.map((tk, i) => `<div class="slider-row"><label for="w${i}" style="margin:0;font-weight:500">${tk}</label>
    <input id="w${i}" type="range" min="0" max="100" value="10"><output id="o${i}">10%</output></div>`).join("");
  $("presets").innerHTML = Object.keys(c.presets).map((k) => `<button type="button" data-p="${k}">${k}</button>`).join("");
  $("presets").addEventListener("click", (e) => {
    const k = e.target.dataset.p; if (!k) return;
    c.presets[k].forEach((v, i) => ($("w" + i).value = Math.round(v * 100)));
    update();
  });
  t.forEach((_, i) => $("w" + i).addEventListener("input", update));
  $("pv").addEventListener("input", update);
  update();
}

function portVol(w, vols, corr) {
  let v = 0;
  for (let i = 0; i < w.length; i++) for (let j = 0; j < w.length; j++) v += w[i] * w[j] * vols[i] * vols[j] * corr[i][j];
  return Math.sqrt(v);
}

function update() {
  const c = R.current, n = c.tickers.length;
  const raw = c.tickers.map((_, i) => +$("w" + i).value);
  const sum = raw.reduce((a, b) => a + b, 0);
  raw.forEach((v, i) => ($("o" + i).textContent = sum ? ((v / sum) * 100).toFixed(0) + "%" : "0%"));
  if (!sum) { ["p-ml", "p-pers", "p-var", "hero-vol"].forEach((id) => ($(id).textContent = "–")); $("contrib").innerHTML = "<p class='note'>Set at least one weight above zero.</p>"; return; }
  const w = raw.map((v) => v / sum);
  const ml = portVol(w, c.model_vol, c.corr), pe = portVol(w, c.persistence_vol, c.corr);
  $("p-ml").textContent = pct(ml); $("p-pers").textContent = pct(pe); $("hero-vol").textContent = pct(ml);
  const dailySd = ml / 100 / Math.sqrt(252);
  $("p-var").textContent = money(1.645 * dailySd * (+$("pv").value || 0));
  // risk contribution_i = w_i * (Sigma w)_i / sigma_p^2
  const sig = c.model_vol.map((v) => v / 100);
  const sw = sig.map((si, i) => si * sig.reduce((a, sj, j) => a + c.corr[i][j] * sj * w[j], 0));
  const sp2 = w.reduce((a, wi, i) => a + wi * sw[i], 0);
  const contrib = c.tickers.map((tk, i) => ({ t: tk, share: (w[i] * sw[i]) / sp2 * 100 })).filter((x) => x.share > 0.05);
  bars($("contrib"), contrib.sort((a, b) => b.share - a.share), "share", "t", 1);
}

// ---------- portfolio backtest ----------
function initBacktest() {
  const names = Object.keys(R.portfolio_backtest);
  $("bt-select").innerHTML = names.map((n) => `<option>${n}</option>`).join("");
  $("bt-select").addEventListener("change", drawBacktest);
  drawBacktest();
}
function drawBacktest() {
  const p = R.portfolio_backtest[$("bt-select").value];
  lineChart("bt-chart", p.dates, [
    { label: "Realised (next 21 days)", data: p.realised, borderColor: css("--real") },
    { label: "ML forecast", data: p.ml, borderColor: css("--ml") },
    { label: "Persistence", data: p.persistence, borderColor: css("--base"), borderDash: [5, 4] },
  ]);
  $("bt-table").querySelector("tbody").innerHTML = Object.entries(p.summary).map(([k, v]) =>
    row([k, fmt(v.rmse_log), (v.var95_exceedance * 100).toFixed(1) + "%"])).join("");
}
