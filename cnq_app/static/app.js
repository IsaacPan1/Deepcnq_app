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
  // save-model panel (Train results)
  runModels: [],         // models the finished run trained (for the Save panel)
  runSplits: 1,          // repeated splits the finished run used
  saveJobId: null,
  // predict tab
  predictModel: null,    // {name} for a saved bundle or {path} for an uploaded one
  predictSummary: null,  // model summary (features, quantiles, training size, ...)
  predictModels: [],     // saved-model list from /api/models
  predictCsvPath: null,
  predictColumns: [],
  predictLastValidation: null,
  predictJobId: null,
  vpTimer: null,         // predict-validation debounce
};

const $ = (id) => document.getElementById(id);

// The shipped demo model is listed under this reserved name (see server.py).
const DEMO_MODEL_NAME = "Demo model (simulated data)";

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
  $("use-demo").addEventListener("click", useDemoTrain);
  // Drag-over styling for every dropzone (train upload, predict model, predict data).
  document.querySelectorAll(".dropzone").forEach((dz) => {
    ["dragenter", "dragover"].forEach((t) => dz.addEventListener(t, () => dz.classList.add("is-over")));
    ["dragleave", "drop"].forEach((t) => dz.addEventListener(t, () => dz.classList.remove("is-over")));
  });
  initTabs();
  initPredict();
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

// Load the demo training data and pre-fill the mapping (time / event / ID / unit).
async function useDemoTrain() {
  const status = $("upload-status");
  if (!state.ready) { blockedMsg(status); return; }
  status.classList.remove("is-error");
  status.textContent = "Loading demo training data…";
  try {
    const res = await fetch("/api/demo/train");
    if (!res.ok) throw new Error((await res.text()) || res.statusText);
    const blob = await res.blob();
    await handleFile(new File([blob], "demo_train.csv", { type: "text/csv" }));
    // Pre-fill the known demo mapping on top of the server's guess.
    const setIf = (id, val) => {
      if ([...$(id).options].some((o) => o.value === val)) $(id).value = val;
    };
    setIf("duration-col", "time");
    setIf("event-col", "event");
    setIf("id-col", "subject_id");
    $("time-unit").value = "months";
    state.timeUnit = "months";
    onMappingSelectChange();  // drop id/dur/event from features and re-validate
  } catch (e) {
    status.classList.add("is-error");
    status.textContent = "Could not load demo data: " + e.message;
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
  state.runModels = [...cfg.models];   // remembered for the Save panel on the results
  state.runSplits = cfg.n_splits;
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
  initSavePanel();
  $("step-results").scrollIntoView({ behavior: "smooth" });
}

function revealFrom(id) { $(id).classList.remove("hidden"); }

// ===========================================================================
// Tabs
// ===========================================================================
function initTabs() {
  $("tab-train").addEventListener("click", () => showTab("train"));
  $("tab-predict").addEventListener("click", () => showTab("predict"));
}

function showTab(which) {
  const train = which === "train";
  $("train-view").classList.toggle("hidden", !train);
  $("predict-view").classList.toggle("hidden", train);
  $("tab-train").classList.toggle("on", train);
  $("tab-predict").classList.toggle("on", !train);
  $("tab-train").setAttribute("aria-selected", String(train));
  $("tab-predict").setAttribute("aria-selected", String(!train));
  if (!train) loadSavedModels();  // keep the saved-model list fresh when opening Predict
}

// ===========================================================================
// Shared helpers for background (save / predict) jobs
// ===========================================================================
function pollJob(jobId, cb) {
  const timer = setInterval(async () => {
    let s;
    try { s = await api("/api/status?id=" + jobId); } catch { return; }
    if (cb.onProgress) cb.onProgress(s);
    if (["done", "cancelled", "error"].includes(s.state)) {
      clearInterval(timer);
      if (s.state === "done" && cb.onDone) cb.onDone(s);
      else if (s.state === "error" && cb.onError) cb.onError(s.error || "failed");
      else if (s.state === "cancelled" && cb.onCancelled) cb.onCancelled(s);
    }
  }, 700);
  return timer;
}

function renderProgress(wrapId, stepId, fillId, detailId, s) {
  const wrap = $(wrapId);
  wrap.classList.remove("hidden");
  wrap.dataset.state = s.state;
  $(stepId).textContent = s.step + (s.state === "cancelled" ? " (cancelled)" : "");
  $(fillId).style.width = Math.round((s.progress || 0) * 100) + "%";
  const spinner = wrap.querySelector(".spinner");
  if (spinner) spinner.style.visibility = ["pending", "running"].includes(s.state) ? "visible" : "hidden";
  if (detailId && $(detailId)) $(detailId).textContent = "Elapsed " + formatElapsed(s.elapsed);
}

function fillTable(tbl, columns, rows) {
  tbl.innerHTML = "";
  const head = document.createElement("tr");
  (columns || []).forEach((c) => {
    const th = document.createElement("th"); th.textContent = c; head.appendChild(th);
  });
  tbl.appendChild(head);
  (rows || []).forEach((row) => {
    const tr = document.createElement("tr");
    row.forEach((v) => {
      const td = document.createElement("td");
      td.textContent = v === null || v === undefined ? ""
        : (typeof v === "number" && !Number.isInteger(v) ? (+v).toPrecision(4) : v);
      tr.appendChild(td);
    });
    tbl.appendChild(tr);
  });
}

function fillColSelect(sel, cols, withNone) {
  sel.innerHTML = "";
  if (withNone) sel.appendChild(new Option("(none)", ""));
  cols.forEach((c) => sel.appendChild(new Option(c, c)));
}

function autoSelectByName(sel, cols, re) {
  const m = cols.find((c) => re.test(c));
  if (m) sel.value = m;
}

function vItem(icon, message) {
  const div = document.createElement("div");
  div.className = "v-item";
  div.innerHTML = `<span class="v-icon"></span><span></span>`;
  div.querySelector(".v-icon").textContent = icon;
  div.querySelector("span:last-child").textContent = message;
  return div;
}

// ===========================================================================
// Save model panel (on the Train results)
// ===========================================================================
function initSavePanel() {
  const sel = $("save-model");
  sel.innerHTML = "";
  (state.runModels || []).forEach((m) => {
    const info = MODEL_INFO[m] || { name: m };
    sel.appendChild(new Option(info.name || m, m));
  });
  const split = $("save-split");
  split.innerHTML = "";
  for (let i = 0; i < (state.runSplits || 1); i++) {
    split.appendChild(new Option("Split " + (i + 1), String(i)));
  }
  $("save-type").onchange = onSaveTypeChange;
  $("save-btn").onclick = onSave;
  onSaveTypeChange();
  $("save-error").textContent = "";
  $("save-result").classList.add("hidden");
  $("save-progress").classList.add("hidden");
  $("save-panel").classList.remove("hidden");
}

function onSaveTypeChange() {
  $("save-split-wrap").classList.toggle("hidden", $("save-type").value !== "single_split");
}

async function onSave() {
  const name = $("save-name").value.trim();
  $("save-error").textContent = "";
  if (!name) { $("save-error").textContent = "Enter a name for the model."; return; }
  if (!state.jobId) { $("save-error").textContent = "Train a model first."; return; }
  const body = {
    source_job_id: state.jobId,
    model: $("save-model").value,
    bundle_type: $("save-type").value,
    name,
  };
  if (body.bundle_type === "single_split") body.split_index = +$("save-split").value;
  $("save-btn").disabled = true;
  $("save-result").classList.add("hidden");
  try {
    const res = await api("/api/save", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    state.saveJobId = res.job_id;
    pollJob(res.job_id, {
      onProgress: (s) => renderProgress("save-progress", "save-progress-step",
        "save-bar-fill", "save-progress-detail", s),
      onDone: async () => {
        $("save-btn").disabled = false;
        let info = {};
        try { info = await api("/api/results?id=" + state.saveJobId); } catch { /* keep name */ }
        const dlName = info.bundle_name || name;
        $("save-result").classList.remove("hidden");
        $("save-result").innerHTML =
          `Saved <b>${escapeHtml(dlName)}</b>. ` +
          `<a href="/api/models/download?name=${encodeURIComponent(dlName)}" download>` +
          `Download ${escapeHtml(dlName)}.cnqmodel</a> — it now appears in the Predict tab.`;
      },
      onError: (e) => { $("save-btn").disabled = false; $("save-error").textContent = "Save failed: " + e; },
    });
  } catch (e) {
    $("save-btn").disabled = false;
    $("save-error").textContent = "Save failed: " + e.message;
  }
}

// ===========================================================================
// Predict tab
// ===========================================================================
function initPredict() {
  $("p-refresh-models").addEventListener("click", loadSavedModels);
  $("p-delete-model").addEventListener("click", onDeleteModel);
  $("p-model-select").addEventListener("change", onPredictModelSelect);
  $("p-model-file").addEventListener("change", onPredictModelUpload);
  $("p-file-input").addEventListener("change", onPredictDataUpload);
  ["p-id-col", "p-time-col", "p-event-col"].forEach((id) =>
    $(id).addEventListener("change", schedulePredictValidate));
  $("p-run-btn").addEventListener("click", onPredictRun);
  $("p-cancel-btn").addEventListener("click", onPredictCancel);
  $("p-demo-rebuild-btn").addEventListener("click", onDemoRebuild);
  $("p-demo-subjects").addEventListener("click", (ev) => {
    const btn = ev.target.closest("button[data-demo]");
    if (btn) loadDemoSubjects(btn.dataset.demo);
  });
}

async function loadSavedModels() {
  try {
    const res = await api("/api/models");
    state.predictModels = res.models || [];
  } catch { state.predictModels = []; }
  const sel = $("p-model-select");
  const previous = sel.value;
  sel.innerHTML = "";
  const usable = state.predictModels.filter((m) => !m.error);
  sel.appendChild(new Option(usable.length ? "— choose a saved model —" : "— no saved models yet —", ""));
  usable.forEach((m) => sel.appendChild(
    new Option(`${m.name} — ${m.model} (${m.bundle_type})`, m.name)));
  if (previous && usable.some((m) => m.name === previous)) sel.value = previous;
}

function onPredictModelSelect() {
  const name = $("p-model-select").value;
  const status = $("p-model-status");
  status.classList.remove("is-error");
  if (!name) {
    $("p-model-summary").classList.add("hidden");
    $("p-delete-model").classList.add("hidden");
    return;
  }
  const m = state.predictModels.find((x) => x.name === name);
  state.predictModel = { name };
  state.predictSummary = m;
  status.textContent = "";
  renderModelSummary(m);
  const isDemo = !!(m && m.is_demo);
  $("p-delete-model").classList.toggle("hidden", isDemo);  // demo can't be deleted
  $("p-demo-rebuild").classList.toggle("hidden", !isDemo);
  if (isDemo && m.needs_rebuild) {
    // The committed bundle can't load here (e.g. torch mismatch) — offer rebuild,
    // and don't advance until it succeeds.
    $("p-demo-rebuild-msg").textContent = "Demo model needs rebuilding" +
      (m.error ? ` (${m.error})` : "") + ".";
    $("p-demo-rebuild-msg").classList.add("is-error");
    $("p-step-data").classList.add("hidden");
  } else {
    $("p-demo-rebuild-msg").textContent = "This is the simulated demo model.";
    $("p-demo-rebuild-msg").classList.remove("is-error");
    onModelChosen();
  }
}

async function onDeleteModel() {
  const name = state.predictModel && state.predictModel.name;
  if (!name) return;
  if (!confirm(`Delete ${name}? This can't be undone.`)) return;
  const status = $("p-model-status");
  try {
    await api("/api/models?name=" + encodeURIComponent(name), { method: "DELETE" });
  } catch (e) {
    status.classList.add("is-error");
    status.textContent = "Delete failed: " + e.message;
    return;
  }
  state.predictModel = null;
  state.predictSummary = null;
  $("p-model-summary").classList.add("hidden");
  $("p-delete-model").classList.add("hidden");
  status.classList.remove("is-error");
  status.textContent = `Deleted ${name}.`;
  await loadSavedModels();
  $("p-model-select").value = "";
}

async function onPredictModelUpload(ev) {
  const file = ev.target.files[0];
  if (!file) return;
  const status = $("p-model-status");
  status.classList.remove("is-error");
  if (!state.ready) { status.classList.add("is-error"); status.textContent = state.healthMsg; return; }
  status.textContent = `Uploading ${file.name}…`;
  try {
    const buf = await file.arrayBuffer();
    const res = await api("/api/predict/upload-model", {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream", "X-Filename": file.name },
      body: buf,
    });
    state.predictModel = { path: res.model_path };
    state.predictSummary = res.summary;
    status.textContent = `Loaded ${file.name}.`;
    $("p-model-select").value = "";                 // it's an uploaded bundle, not a saved one
    $("p-delete-model").classList.add("hidden");    // delete only applies to saved models
    $("p-demo-rebuild").classList.add("hidden");
    renderModelSummary(res.summary);
    onModelChosen();
  } catch (e) {
    status.classList.add("is-error");
    status.textContent = "Could not load model: " + e.message;
  }
}

function renderModelSummary(m) {
  const box = $("p-model-summary");
  if (!m) { box.classList.add("hidden"); return; }
  const rows = [
    ["Model", m.model],
    ["What it is", (m.bundle_type || "") + (m.n_members ? ` (${m.n_members} member${m.n_members === 1 ? "" : "s"})` : "")],
    ["Trained on", `${m.n ?? "—"} subjects · ${m.n_events ?? "—"} events · ${m.censoring_pct ?? "—"}% censored`],
    ["Features", (m.features || []).join(", ") || "—"],
    ["Quantiles", (m.quantiles || []).map((q) => +q).join(", ") || "—"],
    ["Time unit", m.time_unit || "—"],
    ["Saved", m.created_at || "—"],
    ["deepcnq", m.deepcnq || "—"],
  ];
  const badge = m.is_demo ? `<span class="badge badge-demo">demo · simulated data</span>` : "";
  box.innerHTML = badge + "<table>" + rows.map(([l, v]) =>
    `<tr><td class="lbl">${escapeHtml(l)}</td><td>${escapeHtml(String(v))}</td></tr>`).join("") + "</table>";
  box.classList.remove("hidden");
}

// A model is chosen: reveal the data step, and re-validate if data is already loaded.
function onModelChosen() {
  $("p-step-data").classList.remove("hidden");
  if (state.predictCsvPath) { buildPredictMapping(); schedulePredictValidate(); }
}

// Load one of the demo new-subject files (already on the server) into the flow.
async function loadDemoSubjects(key) {
  const status = $("p-upload-status");
  if (!state.ready) { blockedMsg(status); return; }
  if (!state.predictModel) {
    status.classList.add("is-error");
    status.textContent = "Choose a model first.";
    return;
  }
  status.classList.remove("is-error");
  status.textContent = "Loading demo subjects…";
  try {
    const info = await api("/api/demo/data?name=" + encodeURIComponent(key));
    state.predictCsvPath = info.csv_path;
    state.predictColumns = info.columns;
    status.textContent = `${info.description} (${info.n_rows.toLocaleString()} rows).`;
    fillTable($("p-preview-table"), info.preview.columns, info.preview.rows);
    $("p-preview-wrap").classList.remove("hidden");
    buildPredictMapping();
    if (info.has_outcomes) {
      if (info.time_col) $("p-time-col").value = info.time_col;
      if (info.event_col) $("p-event-col").value = info.event_col;
      $("p-external-block").open = true;   // reveal external-validation mapping
    }
    $("p-step-map").classList.remove("hidden");
    $("p-step-run").classList.remove("hidden");
    schedulePredictValidate();
  } catch (e) {
    status.classList.add("is-error");
    status.textContent = "Could not load demo subjects: " + e.message;
  }
}

// Rebuild the demo bundle (retrains from demo_train.csv) when it can't load here.
async function onDemoRebuild() {
  const msg = $("p-demo-rebuild-msg");
  $("p-demo-rebuild-btn").disabled = true;
  try {
    const res = await api("/api/demo/rebuild", { method: "POST" });
    pollJob(res.job_id, {
      onProgress: (s) => renderProgress("p-demo-rebuild-progress", "p-demo-rebuild-step",
        "p-demo-rebuild-fill", "p-demo-rebuild-detail", s),
      onDone: async () => {
        $("p-demo-rebuild-btn").disabled = false;
        await loadSavedModels();
        $("p-model-select").value = DEMO_MODEL_NAME;
        onPredictModelSelect();
        msg.textContent = "Demo model rebuilt.";
        msg.classList.remove("is-error");
      },
      onError: (e) => {
        $("p-demo-rebuild-btn").disabled = false;
        msg.textContent = "Rebuild failed: " + e;
        msg.classList.add("is-error");
      },
    });
  } catch (e) {
    $("p-demo-rebuild-btn").disabled = false;
    msg.textContent = "Rebuild failed: " + e.message;
    msg.classList.add("is-error");
  }
}

async function onPredictDataUpload(ev) {
  const file = ev.target.files[0];
  if (!file) return;
  const status = $("p-upload-status");
  const dz = $("p-file-input").closest(".dropzone");
  status.classList.remove("is-error");
  if (!state.ready) { status.classList.add("is-error"); status.textContent = state.healthMsg; return; }
  status.textContent = `Uploading ${file.name}…`;
  try {
    const buf = await file.arrayBuffer();
    const info = await api("/api/predict/upload-data", {
      method: "POST",
      headers: { "Content-Type": "text/csv", "X-Filename": file.name },
      body: buf,
    });
    state.predictCsvPath = info.csv_path;
    state.predictColumns = info.columns;
    status.textContent = `Loaded ${info.n_rows.toLocaleString()} rows and ${info.columns.length} columns.`;
    dz.querySelector(".dz-title").textContent = file.name;
    dz.querySelector(".dz-hint").textContent = "Choose a different file";
    dz.classList.add("has-file");
    fillTable($("p-preview-table"), info.preview.columns, info.preview.rows);
    $("p-preview-wrap").classList.remove("hidden");
    buildPredictMapping();
    $("p-step-map").classList.remove("hidden");
    $("p-step-run").classList.remove("hidden");
    schedulePredictValidate();
  } catch (e) {
    status.classList.add("is-error");
    status.textContent = "Upload failed: " + e.message;
  }
}

function buildPredictMapping() {
  const features = (state.predictSummary && state.predictSummary.features) || [];
  const cols = state.predictColumns || [];
  const lower = {};
  cols.forEach((c) => { lower[c.toLowerCase()] = c; });

  const wrap = $("p-feature-map");
  wrap.innerHTML = "";
  const grid = document.createElement("div");
  grid.className = "grid-custom";
  features.forEach((f) => {
    const label = document.createElement("label");
    label.textContent = f;
    const sel = document.createElement("select");
    sel.dataset.feature = f;
    sel.appendChild(new Option("— none —", ""));
    cols.forEach((c) => sel.appendChild(new Option(c, c)));
    sel.value = cols.includes(f) ? f : (lower[f.toLowerCase()] || "");
    sel.addEventListener("change", schedulePredictValidate);
    label.appendChild(sel);
    grid.appendChild(label);
  });
  wrap.appendChild(grid);

  fillColSelect($("p-id-col"), cols, true);
  fillColSelect($("p-time-col"), cols, true);
  fillColSelect($("p-event-col"), cols, true);
  autoSelectByName($("p-time-col"), cols, /time|surv|dur|month|day|year|follow|fu/i);
  autoSelectByName($("p-event-col"), cols, /event|status|death|died|dead|censor|relaps|recur/i);
}

function currentFeatureMap() {
  const map = {};
  $("p-feature-map").querySelectorAll("select[data-feature]").forEach((sel) => {
    if (sel.value) map[sel.dataset.feature] = sel.value;
  });
  return map;
}

function schedulePredictValidate() {
  clearTimeout(state.vpTimer);
  state.vpTimer = setTimeout(runPredictValidation, 250);
}

async function runPredictValidation() {
  if (!state.predictCsvPath || !state.predictModel) return;
  const body = {
    model_ref: state.predictModel,
    csv_path: state.predictCsvPath,
    feature_map: currentFeatureMap(),
    id_col: $("p-id-col").value || null,
    time_col: $("p-time-col").value || null,
    event_col: $("p-event-col").value || null,
  };
  try {
    const res = await api("/api/predict/validate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    state.predictLastValidation = res;
    renderPredictValidation(res);
  } catch (e) {
    $("p-run-error").textContent = "Could not validate: " + e.message;
  }
  updatePredictGating();
}

function renderPredictValidation(res) {
  $("p-validation-panel").classList.remove("hidden");
  const errBox = $("p-v-errors");
  errBox.innerHTML = "";
  (res.errors || []).forEach((e) => errBox.appendChild(vItem("✕", e.message)));
  const warnBox = $("p-v-warnings");
  warnBox.innerHTML = "";
  (res.warnings || []).forEach((w) => warnBox.appendChild(vItem("!", w.message)));
  const s = res.summary || {};
  $("p-v-summary").innerHTML =
    `Rows: <b>${fmtNum(s.rows_used)}</b> to predict · ` +
    `<span>${fmtNum(s.rows_excluded)} excluded</span> · ` +
    `Model features: <b>${fmtNum(s.n_features)}</b>` +
    (s.has_external ? ` · <span class="ok">external validation on</span>` : "");
}

function canPredict() {
  const v = state.predictLastValidation;
  return !!(v && (v.errors || []).length === 0 && state.predictCsvPath && state.predictModel);
}

function updatePredictGating() {
  $("p-run-btn").disabled = !canPredict();
}

async function onPredictRun() {
  if (!canPredict()) { updatePredictGating(); return; }
  $("p-run-error").textContent = "";
  $("p-step-results").classList.add("hidden");
  const body = {
    model_ref: state.predictModel,
    csv_path: state.predictCsvPath,
    feature_map: currentFeatureMap(),
    id_col: $("p-id-col").value || null,
    time_col: $("p-time-col").value || null,
    event_col: $("p-event-col").value || null,
  };
  try {
    const res = await api("/api/predict/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    state.predictJobId = res.job_id;
    $("p-run-btn").disabled = true;
    $("p-cancel-btn").classList.remove("hidden");
    $("p-progress-wrap").classList.remove("hidden");
    $("p-progress-wrap").dataset.state = "pending";
    $("p-bar-fill").style.width = "0%";
    pollJob(res.job_id, {
      onProgress: (s) => renderProgress("p-progress-wrap", "p-progress-step",
        "p-bar-fill", "p-progress-detail", s),
      onDone: showPredictResults,
      onError: (e) => {
        $("p-run-btn").disabled = false;
        $("p-cancel-btn").classList.add("hidden");
        $("p-run-error").textContent = "Failed: " + e;
        maybeOfferDemoRebuild(e);
      },
      onCancelled: () => {
        $("p-run-btn").disabled = false;
        $("p-cancel-btn").classList.add("hidden");
      },
    });
  } catch (e) {
    $("p-run-error").textContent = e.message;
  }
}

async function onPredictCancel() {
  if (!state.predictJobId) return;
  await api("/api/cancel?id=" + state.predictJobId, { method: "POST" });
  $("p-progress-step").textContent = "Cancelling…";
}

// If a predict run failed because the demo bundle couldn't load, offer a rebuild.
function maybeOfferDemoRebuild(err) {
  const isDemo = state.predictModel && state.predictModel.name === DEMO_MODEL_NAME;
  if (isDemo && /bundle|weights|could not read|load|architecture|corrupt/i.test(String(err))) {
    $("p-demo-rebuild").classList.remove("hidden");
    $("p-demo-rebuild-msg").textContent = "Demo model needs rebuilding (" + err + ").";
    $("p-demo-rebuild-msg").classList.add("is-error");
  }
}

async function showPredictResults() {
  $("p-run-btn").disabled = false;
  $("p-cancel-btn").classList.add("hidden");
  let r;
  try { r = await api("/api/results?id=" + state.predictJobId); }
  catch (e) { $("p-run-error").textContent = "Could not load results: " + e.message; return; }

  let summary = `${fmtNum(r.n)} subject(s) predicted`;
  if (r.rows_excluded) summary += ` · ${fmtNum(r.rows_excluded)} excluded (missing features)`;
  if (r.out_of_range_count) summary += ` · ${fmtNum(r.out_of_range_count)} flagged out-of-range`;
  const ext = r.external;
  if (ext && ext.available) {
    summary += ` · external: 80% coverage ${(ext.coverage_80 * 100).toFixed(0)}%`;
    if (ext.uno_c != null) summary += `, Uno C ${(+ext.uno_c).toFixed(3)}`;
    summary += `, pinball ${(+ext.pinball_mean).toFixed(3)}`;
  } else if (ext && !ext.available) {
    summary += ` · external validation not available`;
  }
  $("p-result-summary").textContent = summary;

  fillTable($("p-pred-table"), r.columns, r.preview_rows);
  $("p-dl-predictions").href = "/api/predictions?id=" + state.predictJobId;
  $("p-dl-report").href = "/api/report?id=" + state.predictJobId;
  $("p-dl-zip").href = "/api/zip?id=" + state.predictJobId;
  $("p-report-frame").src = "/api/report?id=" + state.predictJobId;
  $("p-step-results").classList.remove("hidden");
  $("p-step-results").scrollIntoView({ behavior: "smooth" });
}

init().catch((e) => alert("Init failed: " + e.message));
