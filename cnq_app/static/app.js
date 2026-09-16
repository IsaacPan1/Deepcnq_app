"use strict";

const state = {
  csvPath: null,
  fileName: null,
  columns: [],
  numeric: [],
  binary: [],
  features: new Set(),
  featureChips: new Map(),
  models: new Set(["KAN_gaps"]),
  idCol: "",
  eventPositive: null,   // value meaning "event" for a two-value event column
  eventValues: null,     // the two distinct values, when mapping is needed
  timeUnit: "",
  guess: null,
  suggestedPreset: null,
  presetTouched: false,
  config: null,
  lastValidation: null,
  acks: new Set(),       // acknowledged warning keys
  jobId: null,
  poll: null,
  vTimer: null,
  ready: false,          // scientific stack + repo available (health check passed)
  healthMsg: "",         // why uploads are blocked, shown inline when not ready
};

const $ = (id) => document.getElementById(id);

// Display names for the package's model identifiers (values sent to the server are unchanged).
const MODEL_INFO = {
  KAN_gaps:             { name: "KAN-CNQ",       desc: "Spline network, non-crossing" },
  TransformerPS_gaps:   { name: "Trans-CNQ",     desc: "Attention across covariates, non-crossing" },
  Transformer_KAN_gaps: { name: "TransKAN-CNQ",  desc: "Attention with spline blocks, non-crossing" },
  MLP_multiQ_gaps:      { name: "MLP-CNQ",       desc: "Baseline MLP, non-crossing" },
  MLP_multiQ:           { name: "MLP multi-quantile", desc: "Baseline; quantiles can cross" },
  MLP_singleQ:          { name: "MLP single-quantile", desc: "One network per level" },
};

function makeChip(label, on, onToggle, desc) {
  const chip = document.createElement("button");
  chip.type = "button";
  chip.className = "chip" + (on ? " on" : "");
  chip.setAttribute("aria-pressed", String(on));
  if (desc) {
    chip.innerHTML = `<span class="chip-name"></span><span class="chip-desc"></span>`;
    chip.querySelector(".chip-name").textContent = label;
    chip.querySelector(".chip-desc").textContent = desc;
  } else {
    chip.textContent = label;
  }
  chip.onclick = () => {
    const now = onToggle();
    chip.classList.toggle("on", now);
    chip.setAttribute("aria-pressed", String(now));
  };
  return chip;
}

function formatElapsed(sec) {
  const s = Math.max(0, Math.round(Number(sec) || 0));
  const m = Math.floor(s / 60);
  return m ? `${m} min ${String(s % 60).padStart(2, "0")} s` : `${s} s`;
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch { data = { raw: text }; }
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

// ---- init ----
// Returns {ok, msg}: ok=false means the scientific stack/repo isn't ready, and
// msg is a short inline explanation to show where the user tries to load data.
async function checkHealth() {
  try {
    const h = await api("/api/health");
    const banner = $("health-banner");
    if (h.repo && !h.repo.found) {
      banner.innerHTML = "The <code>deepcnq</code> repository was not found, so the app " +
        "cannot train. Place it at <code>../deepcnq</code> next to this app, or set the " +
        "<code>CNQ_REPO</code> environment variable, then reload.";
      banner.classList.remove("hidden");
      return { ok: false, msg: "Can't load data: the deepcnq repository wasn't found — see the banner above, fix it and reload." };
    }
    if (!h.ok) {
      const pkgs = (h.missing_pip || []);
      banner.innerHTML = "Missing Python package(s): " +
        pkgs.map((p) => `<code>${p}</code>`).join(", ") +
        ". Install them, then reload — training will fail until they are available.";
      banner.classList.remove("hidden");
      return { ok: false, msg: "Can't load data: missing Python package(s) — " +
        pkgs.join(", ") + ". Fix the environment and reload." };
    }
  } catch (e) { /* health check itself failing shouldn't block the page */ }
  return { ok: true, msg: "" };
}

async function init() {
  const health = await checkHealth();
  state.ready = health.ok;
  state.healthMsg = health.msg;
  // Wire the upload affordances unconditionally so the dropzone and the "Use
  // sample data" button always respond — when the stack isn't ready they show
  // state.healthMsg instead of failing silently.
  $("file-input").addEventListener("change", onUpload);
  $("use-sample").addEventListener("click", useSample);
  const dz = document.querySelector(".dropzone");
  ["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, () => dz.classList.add("is-over")));
  ["dragleave", "drop"].forEach((t) => dz.addEventListener(t, () => dz.classList.remove("is-over")));
  if (!health.ok) return;  // the rest (config, model/preset chips, run) needs the stack
  state.config = await api("/api/config");
  buildModelChips();
  buildPresetSelect();
  document.querySelectorAll('input[name=mode]').forEach((r) =>
    r.addEventListener("change", onModeChange));
  $("quantile-grid").addEventListener("change", onQuantileChange);
  $("quantile-custom").addEventListener("input", onQuantileChange);
  $("preset-select").addEventListener("change", () => { state.presetTouched = true; });
  onQuantileChange();
  $("duration-col").addEventListener("change", onMappingSelectChange);
  $("event-col").addEventListener("change", onEventColChange);
  $("id-col").addEventListener("change", onMappingSelectChange);
  $("time-unit").addEventListener("input", () => { state.timeUnit = $("time-unit").value; });
  ["split-train", "split-valid", "split-test", "n-splits", "seed"].forEach((id) =>
    $(id).addEventListener("change", scheduleValidate));
  $("run-btn").addEventListener("click", onRun);
  $("cancel-btn").addEventListener("click", onCancel);
}

function buildModelChips() {
  const wrap = $("model-list");
  wrap.innerHTML = "";
  state.config.models.forEach((m) => {
    const info = MODEL_INFO[m] || { name: m, desc: "" };
    const chip = makeChip(info.name, state.models.has(m), () => {
      if (state.models.has(m)) state.models.delete(m); else state.models.add(m);
      updateGating();
      return state.models.has(m);
    }, info.desc || m);
    chip.title = m;
    wrap.appendChild(chip);
  });
}

function presetLabel(name) {
  const meta = (state.config.preset_meta || {})[name];
  return (meta && meta.display) || name;
}

function buildPresetSelect() {
  const sel = $("preset-select");
  sel.innerHTML = "";
  state.config.preset_names.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name; opt.textContent = presetLabel(name);
    sel.appendChild(opt);
  });
}

// Mark the closest preset "suggested" and pre-select it (until the user chooses).
function applySuggestion(name) {
  state.suggestedPreset = name || null;
  const sel = $("preset-select");
  [...sel.options].forEach((opt) => { opt.textContent = presetLabel(opt.value); });
  if (!name) return;
  const opt = [...sel.options].find((o) => o.value === name);
  if (opt) opt.textContent = presetLabel(name) + " — suggested";
  if (!state.presetTouched && opt) sel.value = name;
}

// ---- upload ----
async function onUpload(ev) {
  await handleFile(ev.target.files[0]);
}

// Fetch the bundled sample CSV and run it through the same upload flow, so the
// user can try the app with one click (no download-then-drag round trip).
async function useSample() {
  const status = $("upload-status");
  if (!state.ready) { blockedMsg(status); return; }
  status.classList.remove("is-error");
  status.textContent = "Loading sample data…";
  try {
    const res = await fetch("/api/sample");
    if (!res.ok) throw new Error((await res.text()) || res.statusText);
    const blob = await res.blob();
    await handleFile(new File([blob], "sample_survival.csv", { type: "text/csv" }));
  } catch (e) {
    status.classList.add("is-error");
    status.textContent = "Could not load sample data: " + e.message;
  }
}

function blockedMsg(status) {
  status.classList.add("is-error");
  status.textContent = state.healthMsg ||
    "The app isn't ready yet — check the message at the top of the page, then reload.";
}

// Upload a File (from the picker, a drop, or the sample button) and open the
// mapping step. Refuses with a clear message when the stack isn't ready.
async function handleFile(file) {
  if (!file) return;
  const status = $("upload-status");
  const dz = document.querySelector(".dropzone");
  status.classList.remove("is-error");
  if (!state.ready) { blockedMsg(status); return; }
  status.textContent = `Uploading ${file.name}…`;
  dz.querySelector(".dz-title").textContent = file.name;
  dz.querySelector(".dz-hint").textContent = "Choose a different file";
  try {
    const buf = await file.arrayBuffer();
    const info = await api("/api/upload", {
      method: "POST",
      headers: { "Content-Type": "text/csv", "X-Filename": file.name },
      body: buf,
    });
    state.csvPath = info.csv_path;
    state.fileName = info.file_name || file.name;
    state.columns = info.columns;
    state.numeric = info.numeric_columns;
    state.binary = info.binary_columns;
    state.guess = info.guess || null;
    status.textContent = `Loaded ${info.n_rows.toLocaleString()} rows and ${info.columns.length} columns.`;
    dz.classList.add("has-file");
    renderPreview(info.preview);
    buildMapping();
    revealFrom("step-map");
    // Use the validation the server already ran with the guessed mapping, then
    // re-validate live as the user adjusts the mapping.
    if (info.validation) { state.lastValidation = info.validation; renderValidation(info.validation); }
    scheduleValidate();
  } catch (e) {
    dz.classList.remove("has-file");
    status.classList.add("is-error");
    status.textContent = "Upload failed: " + e.message;
  }
}

function renderPreview(preview) {
  const tbl = $("preview-table");
  tbl.innerHTML = "";
  const thead = document.createElement("tr");
  preview.columns.forEach((c) => {
    const th = document.createElement("th"); th.textContent = c; thead.appendChild(th);
  });
  tbl.appendChild(thead);
  preview.rows.forEach((row) => {
    const tr = document.createElement("tr");
    row.forEach((v) => {
      const td = document.createElement("td"); td.textContent = v === null ? "" : v;
      tr.appendChild(td);
    });
    tbl.appendChild(tr);
  });
  $("preview-wrap").classList.remove("hidden");
}

// ---- mapping ----
function buildMapping() {
  const g = state.guess || {};
  const dur = $("duration-col"), evt = $("event-col"), id = $("id-col");
  dur.innerHTML = ""; evt.innerHTML = ""; id.innerHTML = "";
  const numeric = state.numeric.length ? state.numeric : state.columns;
  numeric.forEach((c) => dur.appendChild(new Option(c, c)));
  state.columns.forEach((c) => evt.appendChild(new Option(c, c)));
  id.appendChild(new Option("(none)", ""));
  state.columns.forEach((c) => id.appendChild(new Option(c, c)));

  dur.value = g.duration_col || numeric[0];
  evt.value = g.event_col || state.columns[0];
  id.value = g.id_col || "";
  state.idCol = id.value;
  state.eventPositive = null;
  state.eventValues = null;
  state.timeUnit = "";
  $("time-unit").value = "";
  $("event-map").classList.add("hidden");

  const feat = $("feature-list");
  feat.innerHTML = "";
  state.features = new Set(g.feature_cols || []);
  state.featureChips = new Map();
  state.columns.forEach((c) => {
    const chip = makeChip(c, state.features.has(c), () => {
      if (state.features.has(c)) state.features.delete(c); else state.features.add(c);
      scheduleValidate();
      return state.features.has(c);
    });
    state.featureChips.set(c, chip);
    feat.appendChild(chip);
  });

  revealFrom("step-config");
  revealFrom("step-run");
}

function setFeature(col, on) {
  if (on) state.features.add(col); else state.features.delete(col);
  const chip = state.featureChips.get(col);
  if (chip) { chip.classList.toggle("on", on); chip.setAttribute("aria-pressed", String(on)); }
}

function onMappingSelectChange() {
  state.idCol = $("id-col").value;
  // A column used as duration / event / ID should not double as a feature.
  [$("duration-col").value, $("event-col").value, state.idCol].forEach((c) => {
    if (c && state.features.has(c)) setFeature(c, false);
  });
  scheduleValidate();
}

function onEventColChange() {
  // A different event column: forget any prior mapping and let validation re-detect.
  state.eventPositive = null;
  state.eventValues = null;
  $("event-map").classList.add("hidden");
  onMappingSelectChange();
}

function onModeChange() {
  const mode = document.querySelector('input[name=mode]:checked').value;
  $("preset-block").classList.toggle("hidden", mode !== "preset");
  $("custom-block").classList.toggle("hidden", mode !== "custom");
}

// ---- quantile grid (mirror of grids.py so the count/warning update live) ----
const Q_STANDARD = [0.1, 0.25, 0.5, 0.75, 0.9];
const Q_REQUIRED = [0.1, 0.5, 0.9];

function seq(a, b, step) { const out = []; for (let i = a; i <= b; i += step) out.push(i); return out; }
function round4(x) { return Math.round(x * 1e4) / 1e4; }

function customLevels() {
  return $("quantile-custom").value.split(",").map((s) => parseFloat(s.trim()))
    .filter((x) => !Number.isNaN(x));
}

function buildGrid(kind, custom) {
  let base = [];
  if (kind === "standard") base = Q_STANDARD.slice();
  else if (kind === "every10") base = seq(10, 90, 10).map((i) => i / 100);
  else if (kind === "every5") base = seq(5, 95, 5).map((i) => i / 100);
  else if (kind === "every1") base = seq(1, 99, 1).map((i) => i / 100);
  else if (kind === "custom") base = custom || customLevels();
  const all = base.concat(Q_REQUIRED).map(round4).filter((v) => v > 0 && v < 1);
  return [...new Set(all)].sort((a, b) => a - b);
}

function onQuantileChange() {
  const kind = $("quantile-grid").value;
  $("quantile-custom-wrap").classList.toggle("hidden", kind !== "custom");
  const grid = buildGrid(kind);
  const preview = grid.length <= 9 ? grid.map((g) => g).join(", ")
    : `${grid[0]} to ${grid[grid.length - 1]}`;
  $("quantile-info").textContent = `${grid.length} quantile level${grid.length === 1 ? "" : "s"}: ${preview}`;
  const warn = $("quantile-warning");
  if (grid.some((v) => v < 0.05 || v > 0.95)) {
    warn.textContent = "Extreme levels rely on very few observed events and may be " +
      "unreliable, especially with heavy censoring.";
    warn.classList.remove("hidden");
  } else {
    warn.classList.add("hidden");
  }
}

// ---- validation ----
function currentRatio() {
  return [+$("split-train").value || 0, +$("split-valid").value || 0, +$("split-test").value || 0];
}

function scheduleValidate() {
  clearTimeout(state.vTimer);
  state.vTimer = setTimeout(runValidation, 250);
}

async function runValidation() {
  if (!state.csvPath) return;
  const body = {
    csv_path: state.csvPath,
    duration_col: $("duration-col").value,
    event_col: $("event-col").value,
    feature_cols: [...state.features],
    id_col: state.idCol || null,
    event_positive: state.eventPositive,
    ratio: currentRatio(),
    seed: +$("seed").value || 42,
    n_splits: +$("n-splits").value || 1,
  };
  try {
    const result = await api("/api/validate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    state.lastValidation = result;
    renderValidation(result);
  } catch (e) {
    // Leave the previous panel; surface the problem near the button.
    $("run-error").textContent = "Could not validate: " + e.message;
  }
}

function warnKey(w) { return `${w.code}:${w.column || ""}`; }

function renderValidation(result) {
  const panel = $("validation-panel");
  panel.classList.remove("hidden");

  // Event mapping control (two non-0/1 values).
  const twoVal = (result.errors || []).find((e) => e.code === "event_two_values");
  if (twoVal) state.eventValues = twoVal.values;
  if (state.eventValues) renderEventMap(state.eventValues);
  else $("event-map").classList.add("hidden");

  // Errors (no checkbox).
  const errBox = $("v-errors");
  errBox.innerHTML = "";
  (result.errors || []).forEach((e) => {
    if (e.code === "event_two_values") return;  // handled by the mapping control
    const div = document.createElement("div");
    div.className = "v-item";
    div.innerHTML = `<span class="v-icon">✕</span><span></span>`;
    div.querySelector("span:last-child").textContent = e.message;
    errBox.appendChild(div);
  });

  // Warnings (each with an "I understand" checkbox).
  const warnBox = $("v-warnings");
  warnBox.innerHTML = "";
  const liveKeys = new Set();
  (result.warnings || []).forEach((w) => {
    const key = warnKey(w);
    liveKeys.add(key);
    const div = document.createElement("div");
    div.className = "v-item";
    const label = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = state.acks.has(key);
    cb.addEventListener("change", () => {
      if (cb.checked) state.acks.add(key); else state.acks.delete(key);
      updateGating();
    });
    const text = document.createElement("span");
    text.innerHTML = `${escapeHtml(w.message)} <span class="v-ack">I understand</span>`;
    label.appendChild(cb);
    label.appendChild(text);
    div.innerHTML = `<span class="v-icon">!</span>`;
    div.appendChild(label);
    warnBox.appendChild(div);
  });
  // Drop acks for warnings that no longer apply.
  [...state.acks].forEach((k) => { if (!liveKeys.has(k)) state.acks.delete(k); });

  if (result.suggested_preset) applySuggestion(result.suggested_preset);
  renderSummary(result.summary || {});
  updateGating();
}

function renderEventMap(values) {
  const wrap = $("event-map");
  wrap.classList.remove("hidden");
  const opts = $("event-map-options");
  opts.innerHTML = "";
  values.forEach((v, i) => {
    const label = document.createElement("label");
    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "event-positive";
    radio.value = String(i);
    radio.checked = state.eventPositive !== null && String(v) === String(state.eventPositive);
    radio.addEventListener("change", () => { state.eventPositive = v; runValidation(); });
    label.appendChild(radio);
    label.appendChild(document.createTextNode(" " + String(v)));
    opts.appendChild(label);
  });
}

function renderSummary(s) {
  const box = $("v-summary");
  if (s.rows_before === undefined) { box.innerHTML = ""; return; }
  const parts = [
    `Rows: <b>${fmtNum(s.rows_used)}</b> used`,
    `<span>${fmtNum(s.rows_excluded)} excluded</span>`,
    `Events: <b>${fmtNum(s.n_events)}</b>`,
    s.censoring_pct != null ? `Censoring: <b>${s.censoring_pct}%</b>` : null,
    (s.duration_min != null) ? `Duration: <b>${fmtG(s.duration_min)}–${fmtG(s.duration_max)}</b>` : null,
    `Features: <b>${fmtNum(s.n_features)}</b>`,
  ].filter(Boolean);
  box.innerHTML = parts.join(" · ");
}

function fmtNum(n) { return (n == null) ? "—" : Number(n).toLocaleString(); }
function fmtG(n) { return (n == null) ? "—" : (+n).toPrecision(3).replace(/\.?0+$/, ""); }
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function canRun() {
  const v = state.lastValidation;
  if (!v) return false;
  if (state.models.size === 0) return false;
  if ((v.errors || []).length) return false;
  return (v.warnings || []).every((w) => state.acks.has(warnKey(w)));
}

function updateGating() {
  const btn = $("run-btn");
  const ok = canRun();
  btn.disabled = !ok;
  const v = state.lastValidation;
  const hint = $("run-error");
  if (ok || !v) { if (hint.dataset.gate) { hint.textContent = ""; delete hint.dataset.gate; } return; }
  hint.dataset.gate = "1";
  if (state.models.size === 0) hint.textContent = "Select at least one model.";
  else if ((v.errors || []).length) hint.textContent = "Fix the data errors above before training.";
  else hint.textContent = "Confirm the warnings above (tick “I understand”) to enable training.";
}

// ---- run ----
function collectConfig() {
  const mode = document.querySelector('input[name=mode]:checked').value;
  const kind = $("quantile-grid").value;
  const cfg = {
    csv_path: state.csvPath,
    file_name: state.fileName,
    duration_col: $("duration-col").value,
    event_col: $("event-col").value,
    feature_cols: [...state.features],
    id_col: state.idCol || null,
    event_positive: state.eventPositive,
    time_unit: state.timeUnit || "",
    models: [...state.models],
    quantile_grid: { kind, custom: kind === "custom" ? customLevels() : null },
    ratio: currentRatio(),
    n_splits: +$("n-splits").value,
    seed: +$("seed").value,
    deterministic: $("deterministic").checked,
    mode,
  };
  if (mode === "preset") {
    cfg.preset = $("preset-select").value;
  } else {
    cfg.custom = {
      hidden_dim: +$("c-hidden").value, layers: +$("c-layers").value,
      dropout: +$("c-dropout").value, grid_size: +$("c-grid").value,
      learning_rate: +$("c-lr").value, weight_decay: +$("c-wd").value,
      batch_size: +$("c-batch").value, maximum_epochs: +$("c-epochs").value,
      patience: +$("c-patience").value,
    };
  }
  return cfg;
}

async function onRun() {
  if (!canRun()) { updateGating(); return; }
  $("run-error").textContent = ""; delete $("run-error").dataset.gate;
  $("step-results").classList.add("hidden");
  const cfg = collectConfig();
  try {
    const res = await api("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cfg),
    });
    state.jobId = res.job_id;
    $("run-btn").disabled = true;
    $("cancel-btn").classList.remove("hidden");
    $("progress-wrap").classList.remove("hidden");
    $("progress-wrap").dataset.state = "pending";
    $("bar-fill").style.width = "0%";
    startPolling();
  } catch (e) {
    $("run-error").textContent = e.message;
  }
}

async function onCancel() {
  if (!state.jobId) return;
  await api("/api/cancel?id=" + state.jobId, { method: "POST" });
  $("progress-step").textContent = "Cancelling…";
}

function startPolling() {
  clearInterval(state.poll);
  state.poll = setInterval(pollStatus, 800);
  pollStatus();
}

async function pollStatus() {
  if (!state.jobId) return;
  let s;
  try { s = await api("/api/status?id=" + state.jobId); }
  catch { return; }
  updateProgress(s);
  if (["done", "cancelled", "error"].includes(s.state)) {
    clearInterval(state.poll);
    updateGating();
    $("cancel-btn").classList.add("hidden");
    if (s.state === "done") showResults();
    else if (s.state === "error") $("run-error").textContent = "Failed: " + s.error;
  }
}

function updateProgress(s) {
  $("progress-wrap").dataset.state = s.state;
  $("progress-step").textContent = s.step + (s.state === "cancelled" ? " (cancelled)" : "");
  $("bar-fill").style.width = Math.round(s.progress * 100) + "%";
  const spinner = document.querySelector("#progress-wrap .spinner");
  const running = ["pending", "running"].includes(s.state);
  spinner.style.visibility = running ? "visible" : "hidden";
  let detail = "";
  if (s.phase === "train" && s.current_model) {
    const name = (MODEL_INFO[s.current_model] || {}).name || s.current_model;
    detail = `${name}, split ${s.current_split} of ${s.n_splits}, ` +
      `epoch ${s.epoch}${s.max_epochs ? " of up to " + s.max_epochs : ""} ` +
      `(fit ${s.unit_index} of ${s.total_units}). `;
  }
  detail += `Elapsed ${formatElapsed(s.elapsed)}.`;
  $("progress-detail").textContent = detail;
}

function showResults() {
  $("dl-report").href = "/api/report?id=" + state.jobId;
  $("dl-zip").href = "/api/zip?id=" + state.jobId;
  $("report-frame").src = "/api/report?id=" + state.jobId;
  $("step-results").classList.remove("hidden");
  $("step-results").scrollIntoView({ behavior: "smooth" });
}

function revealFrom(id) { $(id).classList.remove("hidden"); }

init().catch((e) => alert("Init failed: " + e.message));
