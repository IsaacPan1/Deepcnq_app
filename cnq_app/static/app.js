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
  tuneJobId: null,       // auto-tune search job
  tuneBest: null,        // winning {model, custom, ...} from the search
  customTouched: false,  // user edited a custom field -> stop auto-filling defaults
  saveJobId: null,
  // predict tab
  predictModelId: null,  // stable model id (demo | upload:<t> | saved stem)
  predictSummary: null,  // the model's list entry / summary (features, source, ...)
  predictModels: [],     // saved-model list from /api/models
  predictCsvPath: null,
  predictColumns: [],
  predictMap: {},          // model feature -> file column (kept across model/file changes)
  predictSuggestions: {},  // model feature -> suggested column (from validate)
  predictLastValidation: null,
  predictJobId: null,
  vpTimer: null,         // predict-validation debounce
  // project tab
  projModelId: null,
  projSummary: null,
  projColumns: [],
  projMap: {},
  projSuggestions: {},
  projCsvPath: null,
  projJobId: null,
};

const $ = (id) => document.getElementById(id);

// The demo model's stable id (ids are never display labels; see server.py / demo.py).
const DEMO_MODEL_ID = "demo";

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
  initProject();
  if (!health.ok) return;  // the rest (config, model chips, run) needs the stack
  state.config = await api("/api/config");
  buildModelChips();
  $("quantile-grid").addEventListener("change", onQuantileChange);
  $("quantile-custom").addEventListener("input", onQuantileChange);
  onQuantileChange();
  $("duration-col").addEventListener("change", onMappingSelectChange);
  $("event-col").addEventListener("change", onEventColChange);
  $("id-col").addEventListener("change", onMappingSelectChange);
  $("time-unit").addEventListener("input", () => { state.timeUnit = $("time-unit").value; });
  ["split-train", "split-valid", "split-test", "n-splits", "seed"].forEach((id) =>
    $(id).addEventListener("change", scheduleValidate));
  $("run-btn").addEventListener("click", onRun);
  $("cancel-btn").addEventListener("click", onCancel);
  // auto-tune
  document.querySelectorAll('input[name=settings-mode]').forEach((r) =>
    r.addEventListener("change", onSettingsModeChange));
  $("tune-btn").addEventListener("click", onFindBest);
  $("tune-cancel-btn").addEventListener("click", onTuneCancel);
  $("tune-use-best").addEventListener("click", onUseBest);
  ["c-hidden", "c-layers", "c-dropout", "c-grid", "c-lr", "c-wd", "c-batch", "c-epochs",
   "c-patience"].forEach((id) => $(id).addEventListener("input", () => { state.customTouched = true; }));
}

function onSettingsModeChange() {
  const mode = document.querySelector('input[name=settings-mode]:checked').value;
  $("manual-block").classList.toggle("hidden", mode !== "manual");
  $("autotune-block").classList.toggle("hidden", mode !== "auto");
}

// Pre-fill the Custom fields with data-driven defaults (until the user edits them).
function applySuggestedCustom(c) {
  if (!c) return;
  const set = (id, v) => { if (v !== undefined && v !== null) $(id).value = v; };
  set("c-hidden", c.hidden_dim); set("c-layers", c.layers); set("c-dropout", c.dropout);
  set("c-grid", c.grid_size); set("c-lr", c.learning_rate); set("c-wd", c.weight_decay);
  set("c-batch", c.batch_size); set("c-epochs", c.maximum_epochs); set("c-patience", c.patience);
}

async function onFindBest() {
  $("tune-error").textContent = "";
  if (!state.csvPath) { $("tune-error").textContent = "Upload data first."; return; }
  const families = [...document.querySelectorAll("#tune-families input:checked")].map((c) => c.value);
  if (!families.length) { $("tune-error").textContent = "Choose at least one model family."; return; }
  const eff = { quick: [6, 1], standard: [12, 1], thorough: [24, 3] }[$("tune-effort").value] || [12, 1];
  const kind = $("quantile-grid").value;
  const body = {
    csv_path: state.csvPath, duration_col: $("duration-col").value, event_col: $("event-col").value,
    feature_cols: [...state.features], id_col: state.idCol || null, event_positive: state.eventPositive,
    quantile_grid: { kind, custom: kind === "custom" ? customLevels() : null },
    ratio: currentRatio(), n_splits: eff[1], seed: +$("seed").value || 42,
    deterministic: $("deterministic").checked, enabled_models: families, n_trials: eff[0],
  };
  $("tune-btn").disabled = true;
  $("tune-results").classList.add("hidden");
  try {
    const res = await api("/api/tune/run", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    state.tuneJobId = res.job_id;
    $("tune-cancel-btn").classList.remove("hidden");
    $("tune-progress-wrap").classList.remove("hidden");
    $("tune-progress-wrap").dataset.state = "pending";
    $("tune-bar-fill").style.width = "0%";
    pollJob(res.job_id, {
      onProgress: (s) => renderProgress("tune-progress-wrap", "tune-progress-step",
        "tune-bar-fill", "tune-progress-detail", s),
      onDone: loadTuneResults,
      onCancelled: loadTuneResults,   // show best-so-far
      onError: (e) => {
        $("tune-btn").disabled = false;
        $("tune-cancel-btn").classList.add("hidden");
        $("tune-error").textContent = "Search failed: " + e;
      },
    });
  } catch (e) {
    $("tune-btn").disabled = false;
    $("tune-error").textContent = "Search failed: " + e.message;
  }
}

async function onTuneCancel() {
  if (!state.tuneJobId) return;
  await api("/api/cancel?id=" + state.tuneJobId, { method: "POST" });
  $("tune-progress-step").textContent = "Cancelling…";
}

async function loadTuneResults() {
  $("tune-btn").disabled = false;
  $("tune-cancel-btn").classList.add("hidden");
  let r;
  try { r = await api("/api/results?id=" + state.tuneJobId); }
  catch (e) { $("tune-error").textContent = "Could not load results: " + e.message; return; }
  state.tuneBest = r.best;
  const lb = r.leaderboard || [];
  const cols = ["#", "Model", "hidden", "layers", "dropout", "lr", "wd", "valid pinball", "test pinball"];
  const rows = lb.map((t) => [t.rank, (MODEL_INFO[t.model] || {}).name || t.model,
    t.custom.hidden_dim, t.custom.layers, t.custom.dropout, t.custom.learning_rate,
    t.custom.weight_decay, (+t.valid_pinball).toFixed(4), (+t.test_pinball).toFixed(4)]);
  fillTable($("tune-table"), cols, rows);
  $("tune-results").classList.remove("hidden");
  $("tune-use-best").disabled = !state.tuneBest;
  if (!lb.length) $("tune-error").textContent = "No trials completed.";
}

function onUseBest() {
  const b = state.tuneBest;
  if (!b) return;
  state.models = new Set([b.model]);           // select the winning model
  buildModelChips();
  applySuggestedCustom(b.custom);              // fill its hyper-parameters
  state.customTouched = true;
  document.querySelector('input[name=settings-mode][value=manual]').checked = true;
  onSettingsModeChange();                      // back to Manual so the user can Start training
  updateGating();
  scheduleValidate();
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

  // Pre-fill Custom hyper-parameters with data-driven defaults until the user edits them.
  if (result.suggested_custom && !state.customTouched) applySuggestedCustom(result.suggested_custom);

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
    mode: "custom",
  };
  cfg.custom = {
    hidden_dim: +$("c-hidden").value, layers: +$("c-layers").value,
    dropout: +$("c-dropout").value, grid_size: +$("c-grid").value,
    learning_rate: +$("c-lr").value, weight_decay: +$("c-wd").value,
    batch_size: +$("c-batch").value, maximum_epochs: +$("c-epochs").value,
    patience: +$("c-patience").value,
  };
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
const TABS = { train: "train-view", predict: "predict-view", project: "project-view" };

function initTabs() {
  Object.keys(TABS).forEach((k) =>
    $("tab-" + k).addEventListener("click", () => showTab(k)));
}

function showTab(which) {
  Object.entries(TABS).forEach(([k, viewId]) => {
    $(viewId).classList.toggle("hidden", k !== which);
    const tab = $("tab-" + k);
    tab.classList.toggle("on", k === which);
    tab.setAttribute("aria-selected", String(k === which));
  });
  if (which === "predict" || which === "project") loadSavedModels();  // keep model lists fresh
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
          `<a href="/api/models/download?id=${encodeURIComponent(dlName)}" download>` +
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
    $(id).addEventListener("change", onPredictColChange));
  $("p-match-position").addEventListener("click", onMatchByPosition);
  $("p-run-btn").addEventListener("click", onPredictRun);
  $("p-cancel-btn").addEventListener("click", onPredictCancel);
  $("p-demo-rebuild-btn").addEventListener("click", onDemoBuild);
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
  populateModelSelect($("p-model-select"));
  populateModelSelect($("pr-model-select"));
}

// Fill a model <select>: option VALUE is the stable id, text is the label.
function populateModelSelect(sel) {
  if (!sel) return;
  const previous = sel.value;
  sel.innerHTML = "";
  const usable = state.predictModels.filter((m) => !m.error);
  sel.appendChild(new Option(usable.length ? "— choose a model —" : "— no models yet —", ""));
  usable.forEach((m) => {
    const label = m.is_demo
      ? m.label + (m.needs_build ? " (not built yet)" : "")
      : `${m.label} — ${m.model || ""} (${m.bundle_type || ""})`;
    sel.appendChild(new Option(label, m.id));
  });
  if (previous && usable.some((m) => m.id === previous)) sel.value = previous;
}

function onPredictModelSelect() {
  const id = $("p-model-select").value;
  const status = $("p-model-status");
  status.classList.remove("is-error");
  if (!id) {
    state.predictModelId = null;
    $("p-model-summary").classList.add("hidden");
    $("p-delete-model").classList.add("hidden");
    $("p-demo-rebuild").classList.add("hidden");
    return;
  }
  const m = state.predictModels.find((x) => x.id === id);
  state.predictModelId = id;
  state.predictSummary = m;
  status.textContent = "";
  renderModelSummary(m);
  const isDemo = !!(m && m.is_demo);
  $("p-delete-model").classList.toggle("hidden", m.source !== "models");  // only saved models
  $("p-demo-rebuild").classList.toggle("hidden", !isDemo);
  const needsBuild = isDemo && m.needs_build;
  const needsRebuild = isDemo && m.needs_rebuild;
  if (needsBuild || needsRebuild) {
    // Not built (or can't load here) — offer a one-click build and don't advance.
    $("p-demo-rebuild-btn").textContent = needsBuild ? "Build demo model" : "Rebuild demo model";
    $("p-demo-rebuild-msg").textContent = needsBuild
      ? "The demo model isn't built yet. Build it now (about 2 minutes)."
      : "The demo model needs rebuilding" + (m.error ? ` (${m.error})` : "") + ".";
    $("p-demo-rebuild-msg").classList.toggle("is-error", !!needsRebuild);
    $("p-step-data").classList.add("hidden");
  } else if (isDemo) {
    $("p-demo-rebuild-btn").textContent = "Rebuild demo model";
    $("p-demo-rebuild-msg").textContent = "This is the simulated demo model.";
    $("p-demo-rebuild-msg").classList.remove("is-error");
    onModelChosen();
  } else {
    onModelChosen();
  }
}

async function onDeleteModel() {
  const id = state.predictModelId;
  const m = state.predictSummary || {};
  if (!id || m.source !== "models") return;
  if (!confirm(`Delete ${m.label || id}? This can't be undone.`)) return;
  const status = $("p-model-status");
  try {
    await api("/api/models?id=" + encodeURIComponent(id), { method: "DELETE" });
  } catch (e) {
    status.classList.add("is-error");
    status.textContent = "Delete failed: " + e.message;
    return;
  }
  state.predictModelId = null;
  state.predictSummary = null;
  $("p-model-summary").classList.add("hidden");
  $("p-delete-model").classList.add("hidden");
  status.classList.remove("is-error");
  status.textContent = `Deleted ${m.label || id}.`;
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
    state.predictModelId = res.id;                  // "upload:<token>" — a real id
    state.predictSummary = { ...res.summary, id: res.id, label: res.label, source: "uploaded" };
    status.textContent = `Loaded ${file.name}.`;
    $("p-model-select").value = "";                 // it's an uploaded bundle, not in the list
    $("p-delete-model").classList.add("hidden");    // delete only applies to saved models
    $("p-demo-rebuild").classList.add("hidden");
    renderModelSummary(state.predictSummary);
    onModelChosen();
  } catch (e) {
    status.classList.add("is-error");
    status.textContent = "Could not load model: " + e.message;
  }
}

function summaryHtml(m) {
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
  return badge + "<table>" + rows.map(([l, v]) =>
    `<tr><td class="lbl">${escapeHtml(l)}</td><td>${escapeHtml(String(v))}</td></tr>`).join("") + "</table>";
}

function attachTemplateButton(box, modelId) {
  const act = document.createElement("p");
  act.className = "map-actions";
  const a = document.createElement("a");
  a.className = "btn btn-outline";
  a.textContent = "Download template for this model";
  a.href = templateUrl(modelId);
  a.setAttribute("download", "");
  act.appendChild(a);
  box.appendChild(act);
}

function renderModelSummary(m) {
  const box = $("p-model-summary");
  if (!m) { box.classList.add("hidden"); return; }
  box.innerHTML = summaryHtml(m);
  attachTemplateButton(box, state.predictModelId);
  box.classList.remove("hidden");
}

// A model is chosen: reveal the data step, and re-validate if data is already loaded.
function onModelChosen() {
  state.predictSuggestions = {};   // suggestions are model-specific
  $("p-step-data").classList.remove("hidden");
  if (state.predictCsvPath) { buildPredictMapping(); schedulePredictValidate(); }
}

// Load one of the demo new-subject files (already on the server) into the flow.
async function loadDemoSubjects(key) {
  const status = $("p-upload-status");
  if (!state.ready) { blockedMsg(status); return; }
  if (!state.predictModelId) {
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
    fillMappingColumns();
    if (info.has_outcomes) {
      if (info.time_col) $("p-time-col").value = info.time_col;
      if (info.event_col) $("p-event-col").value = info.event_col;
      $("p-external-block").open = true;   // reveal external-validation mapping
    }
    buildPredictMapping();
    $("p-step-map").classList.remove("hidden");
    $("p-step-run").classList.remove("hidden");
    schedulePredictValidate();
  } catch (e) {
    status.classList.add("is-error");
    status.textContent = "Could not load demo subjects: " + e.message;
  }
}

// Build (or rebuild) the demo bundle through a job. If a build is already
// running the server returns that job, so clicking twice just attaches to it.
async function onDemoBuild() {
  const msg = $("p-demo-rebuild-msg");
  $("p-demo-rebuild-btn").disabled = true;
  msg.classList.remove("is-error");
  msg.textContent = "Building the demo model… this takes about 2 minutes.";
  try {
    const res = await api("/api/demo/build", { method: "POST" });
    pollJob(res.job_id, {
      onProgress: (s) => renderProgress("p-demo-rebuild-progress", "p-demo-rebuild-step",
        "p-demo-rebuild-fill", "p-demo-rebuild-detail", s),
      onDone: async () => {
        $("p-demo-rebuild-btn").disabled = false;
        await loadSavedModels();
        $("p-model-select").value = DEMO_MODEL_ID;
        onPredictModelSelect();          // now built → reveals the data step
        msg.textContent = "Demo model ready.";
        msg.classList.remove("is-error");
        $("p-demo-rebuild").classList.add("hidden");
      },
      onError: (e) => {
        $("p-demo-rebuild-btn").disabled = false;
        msg.textContent = "Build failed: " + e;
        msg.classList.add("is-error");
      },
      onCancelled: () => {
        $("p-demo-rebuild-btn").disabled = false;
        msg.textContent = "Build cancelled.";
      },
    });
  } catch (e) {
    $("p-demo-rebuild-btn").disabled = false;
    msg.textContent = "Build failed: " + e.message;
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
    fillMappingColumns();
    buildPredictMapping();
    $("p-step-map").classList.remove("hidden");
    $("p-step-run").classList.remove("hidden");
    schedulePredictValidate();
  } catch (e) {
    status.classList.add("is-error");
    status.textContent = "Upload failed: " + e.message;
  }
}

function jsNorm(s) { return String(s).toLowerCase().replace(/[\s_-]+/g, ""); }

// Mirror of predict.auto_feature_map: exact names first, then normalised
// (case- and separator-insensitive); one column never assigned to two features.
function jsAutoMatch(features, cols, used) {
  used = used || new Set();
  const out = {};
  features.forEach((f) => { if (cols.includes(f) && !used.has(f)) { out[f] = f; used.add(f); } });
  const n2c = {};
  cols.forEach((c) => { const n = jsNorm(c); if (!(n in n2c)) n2c[n] = c; });
  features.forEach((f) => {
    if (out[f]) return;
    const c = n2c[jsNorm(f)];
    if (c !== undefined && !used.has(c)) { out[f] = c; used.add(c); }
  });
  return out;
}

// Fill the ID / time / event selects once when the file's columns change.
function fillMappingColumns() {
  state.predictSuggestions = {};   // a new file invalidates old suggestions
  const cols = state.predictColumns || [];
  fillColSelect($("p-id-col"), cols, true);
  fillColSelect($("p-time-col"), cols, true);
  fillColSelect($("p-event-col"), cols, true);
  autoSelectByName($("p-id-col"), cols, /^(subject_?id|patient_?id|case_?id|id)$|_id$/i);
  autoSelectByName($("p-time-col"), cols, /time|surv|dur|month|day|year|follow|fu/i);
  autoSelectByName($("p-event-col"), cols, /event|status|death|died|dead|censor|relaps|recur/i);
}

// Build the feature-mapping table, driven by the model's features. Keeps valid
// manual choices from state.predictMap, then auto-matches the rest by name.
function buildPredictMapping() {
  const summary = state.predictSummary || {};
  const features = summary.features || [];
  const ranges = summary.feature_ranges || {};
  const cols = state.predictColumns || [];

  const map = {};
  const used = new Set();
  features.forEach((f) => {
    const c = state.predictMap[f];
    if (c && cols.includes(c) && !used.has(c)) { map[f] = c; used.add(c); }
  });
  const remaining = features.filter((f) => !map[f]);
  const auto = jsAutoMatch(remaining, cols.filter((c) => !used.has(c)), new Set());
  Object.entries(auto).forEach(([f, c]) => { if (!used.has(c)) { map[f] = c; used.add(c); } });
  state.predictMap = map;

  renderMapTable(features, ranges, cols);
  updateMapStatus();
  updateTemplateAndPosition();
}

function renderMapTable(features, ranges, cols) {
  const tbl = $("p-feature-map");
  tbl.innerHTML = "";
  const head = document.createElement("tr");
  ["Model feature", "Training range", "File column", ""].forEach((h) => {
    const th = document.createElement("th"); th.textContent = h; head.appendChild(th);
  });
  tbl.appendChild(head);
  features.forEach((f) => {
    const tr = document.createElement("tr");
    tr.dataset.feature = f;
    const tdF = document.createElement("td"); tdF.className = "map-feature"; tdF.textContent = f;
    const r = ranges[f];
    const tdR = document.createElement("td"); tdR.className = "map-range";
    tdR.textContent = r ? `${fmtG(r.min)} – ${fmtG(r.max)}` : "—";
    const tdSel = document.createElement("td");
    const sel = document.createElement("select"); sel.dataset.feature = f;
    sel.appendChild(new Option("not mapped", ""));
    cols.forEach((c) => {
      const suggested = state.predictSuggestions[f] === c;
      sel.appendChild(new Option(c + (suggested ? " (suggested)" : ""), c));
    });
    sel.value = state.predictMap[f] || "";
    sel.addEventListener("change", () => onMapSelectChange(f, sel.value));
    tdSel.appendChild(sel);
    const tdS = document.createElement("td"); tdS.className = "map-state";
    tr.appendChild(tdF); tr.appendChild(tdR); tr.appendChild(tdSel); tr.appendChild(tdS);
    tbl.appendChild(tr);
  });
  markMapStatuses();
}

function onMapSelectChange(f, val) {
  if (val) {
    // one file column maps to one feature: release it from any other feature
    Object.keys(state.predictMap).forEach((g) => {
      if (g !== f && state.predictMap[g] === val) delete state.predictMap[g];
    });
    state.predictMap[f] = val;
  } else {
    delete state.predictMap[f];
  }
  renderMapTable((state.predictSummary || {}).features || [],
    (state.predictSummary || {}).feature_ranges || {}, state.predictColumns || []);
  updateMapStatus();
  schedulePredictValidate();
}

function markMapStatuses() {
  $("p-feature-map").querySelectorAll("tr[data-feature]").forEach((tr) => {
    const mapped = !!state.predictMap[tr.dataset.feature];
    tr.classList.toggle("matched", mapped);
    tr.classList.toggle("unmatched", !mapped);
    const st = tr.querySelector(".map-state");
    if (st) st.textContent = mapped ? "✓" : "!";
  });
}

function updateMapStatus() {
  const features = (state.predictSummary || {}).features || [];
  const n = features.filter((f) => state.predictMap[f]).length;
  $("p-map-status").textContent = features.length
    ? `${n} of ${features.length} features matched`
    : "— match each model feature to a column —";
}

function candidateColumns() {
  const cols = state.predictColumns || [];
  const excluded = new Set([$("p-id-col").value, $("p-time-col").value,
    $("p-event-col").value].filter(Boolean));
  return cols.filter((c) => !excluded.has(c));
}

function templateUrl(modelId) {
  return "/api/models/" + encodeURIComponent(modelId || "") + "/template.csv";
}

function updateTemplateAndPosition() {
  const features = (state.predictSummary || {}).features || [];
  const link = $("p-template-link");
  link.classList.toggle("hidden", !state.predictModelId);
  link.href = templateUrl(state.predictModelId);
  // Position matching needs at least as many candidate columns as features.
  $("p-match-position").disabled = !(features.length && candidateColumns().length >= features.length);
}

function onMatchByPosition() {
  const features = (state.predictSummary || {}).features || [];
  const cand = candidateColumns();
  if (!features.length || cand.length < features.length) return;
  const pairs = features.map((f, i) => `  ${f}  →  ${cand[i]}`).join("\n");
  const ok = confirm(
    "Match by position pairs the model's features with the file's columns in order:\n\n" +
    pairs + "\n\nIf the column order is wrong, predictions will be wrong with no error. Continue?");
  if (!ok) return;
  const map = {};
  features.forEach((f, i) => { map[f] = cand[i]; });
  state.predictMap = map;
  renderMapTable(features, (state.predictSummary || {}).feature_ranges || {}, state.predictColumns || []);
  updateMapStatus();
  schedulePredictValidate();
}

function currentFeatureMap() {
  return { ...state.predictMap };
}

function schedulePredictValidate() {
  clearTimeout(state.vpTimer);
  state.vpTimer = setTimeout(runPredictValidation, 250);
}

// ID / time / event choice changes affect the position-match candidates too.
function onPredictColChange() {
  updateTemplateAndPosition();
  schedulePredictValidate();
}

async function runPredictValidation() {
  if (!state.predictCsvPath || !state.predictModelId) return;
  const body = {
    model_id: state.predictModelId,
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
    const sug = res.suggestions || {};
    if (JSON.stringify(sug) !== JSON.stringify(state.predictSuggestions)) {
      state.predictSuggestions = sug;          // re-render the table to mark "(suggested)"
      buildPredictMapping();
    }
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
  const m = state.predictSummary || {};
  if (m.needs_build || m.needs_rebuild) return false;   // demo not usable until built
  return !!(v && (v.errors || []).length === 0 && state.predictCsvPath && state.predictModelId);
}

function updatePredictGating() {
  $("p-run-btn").disabled = !canPredict();
}

async function onPredictRun() {
  if (!canPredict()) { updatePredictGating(); return; }
  $("p-run-error").textContent = "";
  $("p-step-results").classList.add("hidden");
  const body = {
    model_id: state.predictModelId,
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
  const isDemo = state.predictModelId === DEMO_MODEL_ID;
  if (isDemo && /bundle|weights|could not read|load|architecture|corrupt/i.test(String(err))) {
    $("p-demo-rebuild").classList.remove("hidden");
    $("p-demo-rebuild-btn").textContent = "Rebuild demo model";
    $("p-demo-rebuild-msg").textContent = "The demo model needs rebuilding (" + err + ").";
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

  const fm = r.feature_map || {};
  const entries = Object.entries(fm);
  $("p-result-mapping").textContent = entries.length
    ? "Mapping used — " + entries.map(([f, c]) => `${f} → ${c}`).join(", ")
    : "";

  fillTable($("p-pred-table"), r.columns, r.preview_rows);
  $("p-dl-predictions").href = "/api/predictions?id=" + state.predictJobId;
  $("p-dl-report").href = "/api/report?id=" + state.predictJobId;
  $("p-dl-zip").href = "/api/zip?id=" + state.predictJobId;
  $("p-report-frame").src = "/api/report?id=" + state.predictJobId;
  $("p-step-results").classList.remove("hidden");
  $("p-step-results").scrollIntoView({ behavior: "smooth" });
}

// ===========================================================================
// Project tab (population / cohort cumulative-event projection)
// ===========================================================================
function initProject() {
  $("pr-refresh-models").addEventListener("click", loadSavedModels);
  $("pr-model-select").addEventListener("change", onProjModelSelect);
  document.querySelectorAll('input[name=pr-mode]').forEach((r) =>
    r.addEventListener("change", onProjModeChange));
  $("pr-file-input").addEventListener("change", onProjDataUpload);
  $("pr-id-col").addEventListener("change", projUpdateTemplateAndPosition);
  $("pr-match-position").addEventListener("click", onProjMatchPosition);
  $("pr-run-btn").addEventListener("click", onProjRun);
  $("pr-cancel-btn").addEventListener("click", onProjCancel);
}

function onProjModelSelect() {
  const id = $("pr-model-select").value;
  const note = $("pr-model-note");
  note.classList.remove("is-error");
  if (!id) {
    state.projModelId = null;
    $("pr-model-summary").classList.add("hidden");
    $("pr-step-inputs").classList.add("hidden");
    $("pr-step-query").classList.add("hidden");
    return;
  }
  const m = state.predictModels.find((x) => x.id === id);
  state.projModelId = id;
  state.projSummary = m;
  state.projMap = {};
  state.projSuggestions = {};
  renderProjSummary(m);
  if (m.needs_build || m.needs_rebuild) {
    note.classList.add("is-error");
    note.textContent = "This model isn't built yet — build it in the Predict tab first.";
    $("pr-step-inputs").classList.add("hidden");
    $("pr-step-query").classList.add("hidden");
    return;
  }
  const popRadio = document.querySelector('input[name=pr-mode][value=population]');
  if (!m.has_population) {
    popRadio.disabled = true;
    document.querySelector('input[name=pr-mode][value=cohort]').checked = true;
    note.textContent = "This model has no stored training curve; population mode needs it "
      + "re-saved. Using cohort mode.";
  } else {
    popRadio.disabled = false;
    note.textContent = "";
  }
  onProjModeChange();
  $("pr-step-inputs").classList.remove("hidden");
  $("pr-step-query").classList.remove("hidden");
}

function renderProjSummary(m) {
  const box = $("pr-model-summary");
  if (!m) { box.classList.add("hidden"); return; }
  box.innerHTML = summaryHtml(m);
  attachTemplateButton(box, state.projModelId);
  box.classList.remove("hidden");
}

function onProjModeChange() {
  const mode = document.querySelector('input[name=pr-mode]:checked').value;
  $("pr-population-block").classList.toggle("hidden", mode !== "population");
  $("pr-cohort-block").classList.toggle("hidden", mode !== "cohort");
}

async function onProjDataUpload(ev) {
  const file = ev.target.files[0];
  if (!file) return;
  const status = $("pr-upload-status");
  const dz = $("pr-file-input").closest(".dropzone");
  status.classList.remove("is-error");
  if (!state.ready) { status.classList.add("is-error"); status.textContent = state.healthMsg; return; }
  status.textContent = `Uploading ${file.name}…`;
  try {
    const buf = await file.arrayBuffer();
    const info = await api("/api/predict/upload-data", {
      method: "POST", headers: { "Content-Type": "text/csv", "X-Filename": file.name }, body: buf,
    });
    state.projCsvPath = info.csv_path;
    state.projColumns = info.columns;
    state.projMap = {};
    state.projSuggestions = {};
    status.textContent = `Loaded ${info.n_rows.toLocaleString()} rows and ${info.columns.length} columns.`;
    dz.querySelector(".dz-title").textContent = file.name;
    dz.querySelector(".dz-hint").textContent = "Choose a different file";
    dz.classList.add("has-file");
    fillColSelect($("pr-id-col"), info.columns, true);
    autoSelectByName($("pr-id-col"), info.columns, /^(subject_?id|patient_?id|case_?id|id)$|_id$/i);
    projBuildMapping();
  } catch (e) {
    status.classList.add("is-error");
    status.textContent = "Upload failed: " + e.message;
  }
}

// Cohort mapping (shares the name-matching helpers with Predict; the server
// validates the final map identically, so the two can't diverge dangerously).
function projBuildMapping() {
  const features = (state.projSummary || {}).features || [];
  const ranges = (state.projSummary || {}).feature_ranges || {};
  const cols = state.projColumns || [];
  const map = {};
  const used = new Set();
  features.forEach((f) => {
    const c = state.projMap[f];
    if (c && cols.includes(c) && !used.has(c)) { map[f] = c; used.add(c); }
  });
  const auto = jsAutoMatch(features.filter((f) => !map[f]), cols.filter((c) => !used.has(c)), new Set());
  Object.entries(auto).forEach(([f, c]) => { if (!used.has(c)) { map[f] = c; used.add(c); } });
  state.projMap = map;
  projRenderTable(features, ranges, cols);
  projUpdateStatus();
  projUpdateTemplateAndPosition();
}

function projRenderTable(features, ranges, cols) {
  const tbl = $("pr-feature-map");
  tbl.innerHTML = "";
  const head = document.createElement("tr");
  ["Model feature", "Training range", "File column", ""].forEach((h) => {
    const th = document.createElement("th"); th.textContent = h; head.appendChild(th);
  });
  tbl.appendChild(head);
  features.forEach((f) => {
    const tr = document.createElement("tr");
    tr.dataset.feature = f;
    const tdF = document.createElement("td"); tdF.className = "map-feature"; tdF.textContent = f;
    const r = ranges[f];
    const tdR = document.createElement("td"); tdR.className = "map-range";
    tdR.textContent = r ? `${fmtG(r.min)} – ${fmtG(r.max)}` : "—";
    const tdSel = document.createElement("td");
    const sel = document.createElement("select"); sel.dataset.feature = f;
    sel.appendChild(new Option("not mapped", ""));
    cols.forEach((c) => {
      const sug = (state.projSuggestions || {})[f] === c;
      sel.appendChild(new Option(c + (sug ? " (suggested)" : ""), c));
    });
    sel.value = state.projMap[f] || "";
    sel.addEventListener("change", () => onProjMapChange(f, sel.value));
    tdSel.appendChild(sel);
    const tdS = document.createElement("td"); tdS.className = "map-state";
    tr.appendChild(tdF); tr.appendChild(tdR); tr.appendChild(tdSel); tr.appendChild(tdS);
    tbl.appendChild(tr);
  });
  projMarkStatuses();
}

function onProjMapChange(f, val) {
  if (val) {
    Object.keys(state.projMap).forEach((g) => {
      if (g !== f && state.projMap[g] === val) delete state.projMap[g];
    });
    state.projMap[f] = val;
  } else {
    delete state.projMap[f];
  }
  projRenderTable((state.projSummary || {}).features || [],
    (state.projSummary || {}).feature_ranges || {}, state.projColumns || []);
  projUpdateStatus();
}

function projMarkStatuses() {
  $("pr-feature-map").querySelectorAll("tr[data-feature]").forEach((tr) => {
    const mapped = !!state.projMap[tr.dataset.feature];
    tr.classList.toggle("matched", mapped);
    tr.classList.toggle("unmatched", !mapped);
    const st = tr.querySelector(".map-state");
    if (st) st.textContent = mapped ? "✓" : "!";
  });
}

function projUpdateStatus() {
  const features = (state.projSummary || {}).features || [];
  const n = features.filter((f) => state.projMap[f]).length;
  $("pr-map-status").textContent = features.length ? `${n} of ${features.length} features matched` : "";
}

function projCandidateColumns() {
  const cols = state.projColumns || [];
  const excluded = new Set([$("pr-id-col").value].filter(Boolean));
  return cols.filter((c) => !excluded.has(c));
}

function projUpdateTemplateAndPosition() {
  const features = (state.projSummary || {}).features || [];
  const link = $("pr-template-link");
  link.classList.toggle("hidden", !state.projModelId);
  link.href = templateUrl(state.projModelId);
  $("pr-match-position").disabled = !(features.length && projCandidateColumns().length >= features.length);
}

function onProjMatchPosition() {
  const features = (state.projSummary || {}).features || [];
  const cand = projCandidateColumns();
  if (!features.length || cand.length < features.length) return;
  const pairs = features.map((f, i) => `  ${f}  →  ${cand[i]}`).join("\n");
  if (!confirm("Match by position pairs the model's features with the file's columns in order:\n\n"
    + pairs + "\n\nWrong column order gives wrong projections with no error. Continue?")) return;
  const map = {};
  features.forEach((f, i) => { map[f] = cand[i]; });
  state.projMap = map;
  projRenderTable(features, (state.projSummary || {}).feature_ranges || {}, state.projColumns || []);
  projUpdateStatus();
}

async function onProjRun() {
  const mode = document.querySelector('input[name=pr-mode]:checked').value;
  $("pr-run-error").textContent = "";
  if (!state.projModelId) { $("pr-run-error").textContent = "Choose a model first."; return; }
  const timePoints = ($("pr-time-points").value || "").split(",")
    .map((s) => parseFloat(s.trim())).filter((x) => !Number.isNaN(x));
  const targetRaw = $("pr-target-events").value;
  const body = {
    model_id: state.projModelId, mode, time_points: timePoints,
    target_events: targetRaw === "" ? null : Number(targetRaw),
  };
  if (mode === "population") {
    body.n = Number($("pr-n").value);
  } else {
    body.csv_path = state.projCsvPath;
    body.feature_map = { ...state.projMap };
    body.id_col = $("pr-id-col").value || null;
  }
  $("pr-step-results").classList.add("hidden");
  try {
    const res = await api("/api/project/run", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    state.projJobId = res.job_id;
    $("pr-run-btn").disabled = true;
    $("pr-cancel-btn").classList.remove("hidden");
    $("pr-progress-wrap").classList.remove("hidden");
    $("pr-progress-wrap").dataset.state = "pending";
    $("pr-bar-fill").style.width = "0%";
    pollJob(res.job_id, {
      onProgress: (s) => renderProgress("pr-progress-wrap", "pr-progress-step",
        "pr-bar-fill", "pr-progress-detail", s),
      onDone: showProjResults,
      onError: (e) => {
        $("pr-run-btn").disabled = false;
        $("pr-cancel-btn").classList.add("hidden");
        $("pr-run-error").textContent = "Failed: " + e;
      },
      onCancelled: () => {
        $("pr-run-btn").disabled = false;
        $("pr-cancel-btn").classList.add("hidden");
      },
    });
  } catch (e) {
    $("pr-run-error").textContent = e.message;
  }
}

async function onProjCancel() {
  if (!state.projJobId) return;
  await api("/api/cancel?id=" + state.projJobId, { method: "POST" });
  $("pr-progress-step").textContent = "Cancelling…";
}

async function showProjResults() {
  $("pr-run-btn").disabled = false;
  $("pr-cancel-btn").classList.add("hidden");
  let r;
  try { r = await api("/api/results?id=" + state.projJobId); }
  catch (e) { $("pr-run-error").textContent = "Could not load results: " + e.message; return; }

  const unit = r.time_unit ? ` ${r.time_unit}` : "";
  let summary = `${r.source === "population" ? "Population" : "Cohort"} projection · ` +
    `N=${fmtNum(r.n)} · supported to t=${(+r.horizon).toPrecision(3)}${unit} · ` +
    `up to ${(+r.max_events).toFixed(0)} events within follow-up`;
  const tf = r.time_for;
  if (tf) {
    summary += tf.out_of_range
      ? " · target beyond follow-up (needs extrapolation)"
      : ` · ${tf.target} events by t=${(+tf.time).toPrecision(3)}${unit}`;
  }
  $("pr-result-summary").textContent = summary;

  $("pr-dl-csv").href = "/api/projection?id=" + state.projJobId;
  $("pr-dl-report").href = "/api/report?id=" + state.projJobId;
  $("pr-dl-zip").href = "/api/zip?id=" + state.projJobId;
  $("pr-report-frame").src = "/api/report?id=" + state.projJobId;
  $("pr-step-results").classList.remove("hidden");
  $("pr-step-results").scrollIntoView({ behavior: "smooth" });
}

init().catch((e) => alert("Init failed: " + e.message));
