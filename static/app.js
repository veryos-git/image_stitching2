"use strict";

/* ============================== state ============================== */
const state = {
  images: [],          // File[] sorted by detected row and column
  positions: [],
  gridValid: false,
  gridRevision: 0,
  incrTemplate: null,
  pairDiagnostics: new Map(),
  diagnosticNames: [],
  overlapGraph: null,
  stages: [],          // from "plan" event
  markerEls: {},       // marker -> DOM refs
  maxSeq: 0,           // WebSocket dedupe on reconnect
  ws: null,
  jobId: null,
  projectId: null,
  running: false,
  resultUrl: null,
  incrSession: null,   // incremental session id
};

const $ = (id) => document.getElementById(id);
const logEl = $("log");
const banner = $("banner");
$("alignment").addEventListener("change", () => {
  const translation = $("alignment").value === "translation";
  $("refine").disabled = translation;
  if (translation) $("refine").checked = false;
});

/* ============================ utilities ============================ */
function log(msg, level = "") {
  const line = document.createElement("div");
  line.className = "line " + level;
  line.textContent = msg;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

function showBanner(msg) {
  banner.textContent = msg;
  banner.classList.add("error");
}
function clearBanner() {
  banner.textContent = "";
  banner.classList.remove("error");
}

function setConn(stateName) {
  $("conn-badge").textContent = "● " + stateName;
}

function setProgress(frac, label) {
  frac = Math.max(0, Math.min(1, frac || 0));
  $("p-fill").style.width = (frac * 100).toFixed(1) + "%";
  $("p-pct").textContent = (frac * 100).toFixed(0) + "%";
  if (label) $("p-label").textContent = label;
}

/* ============================ presets ============================ */
const PRESETS = {
  fast:     { feature_max_dim: 512,  max_keypoints: 512,  sinkhorn_iterations: 20,  match_threshold: 0.2,
              hint: "SuperPoint on a 512px image + 512 keypoints → ~4× less compute." },
  balanced: { feature_max_dim: 1024, max_keypoints: 1024, sinkhorn_iterations: 50,  match_threshold: 0.2,
              hint: "Features detected at 1024px, 1024 keypoints." },
  best:     { feature_max_dim: 1600, max_keypoints: 2048, sinkhorn_iterations: 100, match_threshold: 0.2,
              hint: "Outdoor-grade detail — slowest on CPU." },
};

function applyPreset(name) {
  const p = PRESETS[name] || PRESETS.balanced;
  $("feature-max-dim").value = p.feature_max_dim;
  $("max-keypoints").value = p.max_keypoints;
  $("sinkhorn-iterations").value = p.sinkhorn_iterations;
  $("match-threshold").value = p.match_threshold;
  $("preset-hint").textContent = p.hint;
}

function resetUI() {
  clearMatchDiagnostics();
  $("timeline").innerHTML = "";
  state.markerEls = {};
  state.stages = [];
  showViewer(null);
  $("result-card").hidden = true;
  $("download").hidden = true;
  $("download-png").hidden = true;
  $("log").innerHTML = "";
  setProgress(0, "Ready");
  clearBanner();
}

/* ============================ timeline ============================ */
function buildTimeline(stages) {
  state.stages = stages;
  state.markerEls = {};
  const tl = $("timeline");
  tl.innerHTML = "";
  stages.forEach((s) => {
    const el = document.createElement("div");
    el.className = "marker";
    el.innerHTML = `
      <div class="node">·</div>
      <div class="body">
        <div class="label">${s.label}</div>
        <div class="msg">Waiting…</div>
        <div class="mini"><div class="fill"></div></div>
        <div class="shots"></div>
      </div>`;
    tl.appendChild(el);
    state.markerEls[s.marker] = {
      el,
      node: el.querySelector(".node"),
      msg: el.querySelector(".msg"),
      fill: el.querySelector(".mini .fill"),
      shots: el.querySelector(".shots"),
    };
  });
}

function setMarker(marker, status, message, fraction) {
  const m = state.markerEls[marker];
  if (!m) return;
  m.el.className = "marker " + status;
  const icons = { running: "…", done: "✓", error: "✗", pending: "·" };
  m.node.textContent = icons[status] || "·";
  if (message != null) m.msg.textContent = message;
  if (fraction != null) {
    m.fill.style.width = (Math.max(0, Math.min(1, fraction)) * 100).toFixed(1) + "%";
  }
}

/* ============================ upload ============================ */
function showNameWarning(message) {
  $("filename-warning-message").textContent = message;
  if (!$("filename-warning").open) $("filename-warning").showModal();
}
$("filename-warning-close").onclick = () => {
  $("filename-warning").close();
  $("filename-template").focus();
};

async function inspectFileNames(files, template) {
  const response = await fetch("/api/grid/positions", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({names: files.map(f => f.name), template}),
  });
  const result = await response.json();
  if (!response.ok) throw Error([result.error, result.help].filter(Boolean).join("\n\n"));
  return result;
}

async function validateBatchNames(popup = false) {
  const revision = ++state.gridRevision;
  state.gridValid = false;
  state.positions = [];
  $("stitch-btn").disabled = true;
  renderThumbs();
  $("grid-status").textContent = state.images.length ? "Detecting row and column…" : "";
  try {
    const result = await inspectFileNames(state.images, $("filename-template").value);
    if (revision !== state.gridRevision) return false;
    state.positions = result.positions;
    state.gridValid = result.valid;
    if (result.valid) {
      const sorted = state.images.map((file, i) => ({file, position: result.positions[i]}))
        .sort((a, b) => a.position[0] - b.position[0] || a.position[1] - b.position[1]);
      state.images = sorted.map(item => item.file);
      state.positions = sorted.map(item => item.position);
      $("grid-status").textContent = state.images.length ? `Detected positions for ${state.images.length} images. Matching uses horizontal and vertical grid neighbors.` : "";
    } else {
      $("grid-status").textContent = result.errors.slice(0, 8).join(" ");
      if (popup) showNameWarning(result.errors.slice(0, 8).join("\n") + "\n\n" + result.help);
    }
    renderThumbs();
    $("stitch-btn").disabled = state.running || !state.gridValid || state.images.length < 2;
    return state.gridValid;
  } catch (error) {
    if (revision !== state.gridRevision) return false;
    $("grid-status").textContent = error.message;
    if (popup) showNameWarning(error.message);
    return false;
  }
}

async function validateIncrementalName(file, template) {
  try {
    const result = await inspectFileNames([file], template);
    if (!result.valid) {
      showNameWarning(result.errors.join("\n") + "\n\n" + result.help);
      return false;
    }
    return true;
  } catch (error) {
    showNameWarning(error.message);
    return false;
  }
}

$("filename-template").addEventListener("input", () => {
  clearMatchDiagnostics();
  ++state.gridRevision;
  state.gridValid = false;
  $("stitch-btn").disabled = true;
});
$("filename-template").addEventListener("change", () => validateBatchNames(true));

function clearMatchDiagnostics() {
  state.pairDiagnostics.clear();
  state.diagnosticNames = [];
  state.overlapGraph = null;
  $("match-debug").close();
  updateMatchHighlights();
}

function pairCrossesGroups(pair) {
  const groups = state.overlapGraph?.components;
  return groups && groups.findIndex(group => group.includes(pair[0])) !== groups.findIndex(group => group.includes(pair[1]));
}

function updateMatchHighlights() {
  const pairs = [...state.pairDiagnostics.values()];
  const rejected = pairs.filter(p => !p.accepted);
  $("match-debug-summary").hidden = !pairs.length;
  $("match-debug-count").textContent = `${rejected.length}/${pairs.length} tested overlaps rejected. ${state.overlapGraph ? `${state.overlapGraph.components.length} connected group(s).` : "Matching in progress."} Click a highlighted tile to inspect its neighbors.`;
  document.querySelectorAll(".tile-grid .thumb").forEach(tile => {
    const file = state.images[Number(tile.dataset.index)];
    const index = state.diagnosticNames.indexOf(file?.name) + 1;
    const related = pairs.filter(p => p.pair.includes(index));
    const bad = related.filter(p => !p.accepted);
    tile.classList.toggle("has-matches", !!related.length);
    tile.classList.toggle("match-rejected", !!bad.length);
    tile.classList.toggle("match-disconnected", bad.some(p => pairCrossesGroups(p.pair)));
    tile.tabIndex = related.length ? 0 : -1;
    tile.setAttribute("role", "button");
    tile.setAttribute("aria-label", `${tile.title}. ${bad.length} rejected neighbor overlaps. Inspect matched features.`);
    tile.onclick = () => { if (related.length) openMatchInspector(index); };
    tile.onkeydown = event => {
      if (event.target !== tile) return;
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); tile.click(); }
    };
  });
}

function openMatchInspector(index = null) {
  const pairs = [...state.pairDiagnostics.entries()]
    .filter(([, p]) => index === null || p.pair.includes(index))
    .sort((a, b) => Number(a[1].accepted) - Number(b[1].accepted));
  const select = $("match-debug-pair");
  select.innerHTML = "";
  for (const [key, p] of pairs) {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = `${p.accepted ? "Accepted" : "Rejected"}: ${p.names.join(" ↔ ")}`;
    select.appendChild(option);
  }
  if (!pairs.length) return;
  showMatchPair();
  if (!$("match-debug").open) $("match-debug").showModal();
}

function showMatchPair() {
  const p = state.pairDiagnostics.get($("match-debug-pair").value);
  if (!p) return;
  $("match-debug-reason").textContent = `${p.accepted ? "Accepted overlap" : `Rejected: ${p.reason}`}\n${p.matches} candidate matches · ${p.inliers} geometric inliers${pairCrossesGroups(p.pair) ? " · These images belong to different overlap groups." : ""}`;
  $("match-debug-names").textContent = `Left: ${p.names[0]}  |  Right: ${p.names[1]}`;
  $("match-debug-image").src = p.image;
  $("match-debug-full").href = p.image;
  $("match-debug-legend").textContent = p.matches ? `Showing ${p.drawn} of ${p.matches} candidate matches.` : "No matched features were found for this pair.";
}
$("match-debug-close").onclick = () => $("match-debug").close();
$("match-debug-open").onclick = () => openMatchInspector();
$("match-debug-pair").onchange = showMatchPair;

function fitThumbnailGrid() {
  const content = $("grid-overlay-content");
  const grid = content.querySelector(".tile-grid");
  if (!grid || $("grid-overlay").hidden) return;
  const scale = Math.min(content.clientWidth / grid.offsetWidth, content.clientHeight / grid.offsetHeight);
  grid.style.transform = `translate(-50%, -50%) scale(${Math.max(0, scale)})`;
}

function positionGridOverlay(left, top, width, height) {
  const overlay = $("grid-overlay");
  const availableWidth = Math.max(1, innerWidth - 16);
  const availableHeight = Math.max(1, innerHeight - 16);
  width = Math.min(availableWidth, Math.max(Math.min(240, availableWidth), width));
  height = Math.min(availableHeight, Math.max(Math.min(200, availableHeight), height));
  overlay.style.width = `${width}px`;
  overlay.style.height = `${height}px`;
  overlay.style.left = `${Math.max(8, Math.min(innerWidth - width - 8, left))}px`;
  overlay.style.top = `${Math.max(8, Math.min(innerHeight - height - 8, top))}px`;
}

function keepGridOverlayVisible() {
  if ($("grid-overlay").hidden) return;
  const rect = $("grid-overlay").getBoundingClientRect();
  positionGridOverlay(rect.left, rect.top, rect.width, rect.height);
  fitThumbnailGrid();
}

for (const [id, resizing] of [["grid-overlay-drag", false], ["grid-overlay-resize", true]]) {
  const handle = $(id);
  let gesture = null;
  handle.addEventListener("pointerdown", event => {
    if (event.button !== 0) return;
    gesture = {x: event.clientX, y: event.clientY, rect: $("grid-overlay").getBoundingClientRect()};
    handle.setPointerCapture(event.pointerId);
    event.preventDefault();
  });
  handle.addEventListener("pointermove", event => {
    if (!gesture) return;
    const dx = event.clientX - gesture.x, dy = event.clientY - gesture.y;
    const r = gesture.rect;
    positionGridOverlay(r.left + (resizing ? 0 : dx), r.top + (resizing ? 0 : dy),
                        r.width + (resizing ? dx : 0), r.height + (resizing ? dy : 0));
  });
  const finish = () => { gesture = null; };
  handle.addEventListener("pointerup", finish);
  handle.addEventListener("pointercancel", finish);
  handle.addEventListener("lostpointercapture", finish);
  handle.addEventListener("keydown", event => {
    const delta = {ArrowLeft: [-10, 0], ArrowRight: [10, 0], ArrowUp: [0, -10], ArrowDown: [0, 10]}[event.key];
    if (!delta) return;
    event.preventDefault();
    const r = $("grid-overlay").getBoundingClientRect();
    positionGridOverlay(r.left + (resizing ? 0 : delta[0]), r.top + (resizing ? 0 : delta[1]),
                        r.width + (resizing ? delta[0] : 0), r.height + (resizing ? delta[1] : 0));
  });
}
new ResizeObserver(fitThumbnailGrid).observe($("grid-overlay-content"));
window.addEventListener("resize", keepGridOverlayVisible);

function renderThumbs() {
  const wrap = $("thumbs");
  wrap.innerHTML = "";
  const overlayContent = $("grid-overlay-content");
  overlayContent.innerHTML = "";
  const positions = state.positions.filter(Boolean);
  $("grid-overlay").hidden = !positions.length;
  let grid = null;
  let firstRow = 0;
  let firstCol = 0;
  if (positions.length) {
    firstRow = Math.min(...positions.map(p => p[0]));
    firstCol = Math.min(...positions.map(p => p[1]));
    const lastRow = Math.max(...positions.map(p => p[0]));
    const lastCol = Math.max(...positions.map(p => p[1]));
    const rows = lastRow - firstRow + 1;
    const cols = lastCol - firstCol + 1;
    $("grid-overlay-caption").textContent = `${rows} rows × ${cols} columns · rows ${firstRow}–${lastRow}, columns ${firstCol}–${lastCol}. Empty cells indicate missing tiles.`;
    grid = document.createElement("div");
    grid.className = "tile-grid";
    grid.style.gridTemplateColumns = `repeat(${cols}, 96px)`;
    grid.style.gridTemplateRows = `repeat(${rows}, 72px)`;
    overlayContent.appendChild(grid);
  }
  const unplaced = document.createElement("div");
  unplaced.className = "unplaced-thumbs";
  const occupied = new Set();
  state.images.forEach((f, i) => {
    const t = document.createElement("div");
    t.className = "thumb";
    t.draggable = false;
    const position = state.positions[i];
    t.title = `${f.name}${position ? ` · row ${position[0]}, column ${position[1]}` : " · position unknown"}`;
    t.dataset.index = i;
    const img = document.createElement("img");
    img.src = URL.createObjectURL(f);
    t.innerHTML = `<span class="idx">${i + 1}</span>
      <button class="del" title="remove">×</button>`;
    t.querySelector(".idx").textContent = position ? `r${position[0]} · c${position[1]}` : "?";
    img.alt = t.title;
    img.onload = () => URL.revokeObjectURL(img.src);
    t.prepend(img);
    t.querySelector(".del").onclick = (e) => {
      e.stopPropagation();
      if (state.running) return;
      clearMatchDiagnostics();
      state.images.splice(i, 1);
      validateBatchNames(false);
    };
    const cell = position?.join(",");
    if (grid && position && !occupied.has(cell)) {
      t.style.gridRow = position[0] - firstRow + 1;
      t.style.gridColumn = position[1] - firstCol + 1;
      grid.appendChild(t);
      occupied.add(cell);
    } else {
      unplaced.appendChild(t);
    }
  });
  if (unplaced.children.length) {
    if (grid) {
      const caption = document.createElement("p");
      caption.className = "hint";
      caption.textContent = "Images with missing or duplicate positions";
      wrap.appendChild(caption);
    }
    wrap.appendChild(unplaced);
  }
  requestAnimationFrame(keepGridOverlayVisible);
  updateMatchHighlights();
}

function addFiles(list) {
  if (state.running) return;
  clearMatchDiagnostics();
  for (const f of list) {
    if (f.type.startsWith("image/")) state.images.push(f);
  }
  validateBatchNames(true);
}

const dz = $("dropzone");
dz.addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", (e) => { addFiles(e.target.files); e.target.value = ""; });
["dragenter", "dragover"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));

/* ============================ viewer ============================ */
function showViewer(src, caption) {
  const img = $("viewer-img");
  const ph = $("viewer-ph");
  const cap = $("viewer-cap");
  if (!src) { img.hidden = true; ph.hidden = false; cap.hidden = true; return; }
  img.src = src;
  img.hidden = false;
  ph.hidden = true;
  if (caption) { cap.hidden = false; cap.textContent = caption; }
  else cap.hidden = true;
}

function addShot(marker, src, caption) {
  const m = state.markerEls[marker];
  if (!m) return;
  const s = document.createElement("div");
  s.className = "shot";
  s.innerHTML = `<img src="${src}" /><div class="cap">${caption || ""}</div>`;
  s.onclick = () => showViewer(src, caption || "");
  m.shots.appendChild(s);
  showViewer(src, caption || "");
}

/* ============================ result ============================ */
function dataUrlToPng(jpegDataUrl, cb) {
  const img = new Image();
  img.onload = () => {
    const c = document.createElement("canvas");
    c.width = img.naturalWidth; c.height = img.naturalHeight;
    c.getContext("2d").drawImage(img, 0, 0);
    cb(c.toDataURL("image/png"));
  };
  img.src = jpegDataUrl;
}

function showResult(ev) {
  state.resultUrl = ev.image;
  $("result-card").hidden = false;
  const rv = $("result-viewer");
  rv.innerHTML = `<img src="${ev.image}" />`;
  const dl = $("download");
  // Prefer a full-res download URL (incremental mode) over the preview data URL.
  dl.href = (ev.meta && ev.meta.download) || ev.image;
  dl.hidden = false;
  const stats = $("stats");
  stats.innerHTML = "";
  const m = ev.meta || {};
  const items = [
    ["Engine", m.engine || "—"],
    ["Images", m.num_images || "—"],
    ["Output", m.output ? `${m.output.w}×${m.output.h}` : "—"],
    ["Canvas", m.canvas ? `${m.canvas.w}×${m.canvas.h}` : "—"],
    ["Reference", m.reference ? `#${m.reference}` : "—"],
    ["Elapsed", m.elapsed != null ? `${m.elapsed}s` : "—"],
  ];
  items.forEach(([k, v]) => {
    const d = document.createElement("div");
    d.className = "stat";
    d.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div>`;
    stats.appendChild(d);
  });
  dataUrlToPng(ev.image, (png) => {
    const dp = $("download-png");
    dp.href = png;
    dp.hidden = false;
  });
  rv.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ============================ stitch ============================ */
function currentOptions() {
  return {
    filename_template: $("filename-template").value,
    engine: $("engine").value,
    weights: $("weights").value,
    max_keypoints: $("max-keypoints").value,
    match_threshold: $("match-threshold").value,
    feature_max_dim: $("feature-max-dim").value,
    sinkhorn_iterations: $("sinkhorn-iterations").value,
    alignment: $("alignment").value,
    ransac_thresh: $("ransac-thresh").value,
    reference: $("reference").value,
    refine: $("refine").checked,
    blend_levels: $("blend-levels").value,
    exposure: $("exposure").checked,
    crop: $("crop").checked,
  };
}

$("stitch-btn").addEventListener("click", async () => {
  if (state.running) return;
  if (state.images.length < 2) {
    showBanner("Add at least 2 images first.");
    return;
  }
  if (!await validateBatchNames(true)) return;
  clearBanner();
  resetUI();
  state.running = true;
  $("filename-template").disabled = true;
  state.maxSeq = 0;
  state.projectId = null;
  $("stitch-btn").disabled = true;
  $("stitch-btn").innerHTML = '<span class="spin"></span>Stitching…';
  setProgress(0, "Uploading…");

  const fd = new FormData();
  state.images.forEach((f) => fd.append("files", f, f.name));
  const opts = currentOptions();
  Object.entries(opts).forEach(([k, v]) => fd.append(k, v));

  try {
    const res = await fetch("/api/stitch", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok || data.error) { showBanner(data.error || data.detail || "Upload failed"); resetButton(); return; }
    state.jobId = data.job_id;
    connect(data.job_id);
  } catch (err) {
    showBanner("Upload failed: " + err.message);
    resetButton();
  }
});

function resetButton() {
  state.running = false;
  $("filename-template").disabled = !!state.incrSession;
  $("stitch-btn").disabled = !state.gridValid || state.images.length < 2;
  $("stitch-btn").innerHTML = "⚡ Stitch panorama";
}

/* ============================ websocket ============================ */
function connect(jobId) {
  setConn("connecting…");
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/${jobId}`);
  state.ws = ws;

  ws.onopen = () => { setConn("live"); log("connected to job " + jobId); };
  ws.onmessage = (ev) => handleEvent(JSON.parse(ev.data));
  ws.onclose = () => {
    setConn(state.running ? "reconnecting…" : "idle");
    if (state.running) setTimeout(() => connect(jobId), 1200);
  };
  ws.onerror = () => ws.close();
}

function handleEvent(e) {
  if (e.seq != null && e.seq <= state.maxSeq) return;  // dedupe on replay
  if (e.seq != null) state.maxSeq = e.seq;

  switch (e.type) {
    case "plan":
      state.diagnosticNames = e.meta.input_names || state.images.map(f => f.name);
      buildTimeline(e.stages);
      log(`engine=${e.meta.engine}, weights=${e.meta.weights}, images=${e.meta.num_images}`);
      break;
    case "step":
      if (e.detail?.graph) {
        state.overlapGraph = e.detail.graph;
        updateMatchHighlights();
        if ($("match-debug").open) showMatchPair();
      }
      setMarker(e.marker, e.status, e.message, e.fraction);
      setProgress(e.progress, e.message || $("p-label").textContent);
      if (e.status === "done") log(`✓ ${e.marker}: ${e.message || ""}`);
      else if (e.status === "error") log(`✗ ${e.marker}: ${e.message || ""}`, "error");
      else log(`${e.marker}: ${e.message || ""}`);
      break;
    case "image":
      if (e.meta?.pair_diagnostic) {
        const p = e.meta.pair_diagnostic;
        state.pairDiagnostics.set(p.pair.join("-"), {...p, names: e.meta.names, image: e.image, drawn: e.meta.drawn_matches});
        updateMatchHighlights();
        break;
      }
      addShot(e.marker, e.image, e.label);
      log(`🖼  ${e.marker}: ${e.label} (${e.caption || ""})`);
      break;
    case "log":
      log(e.message, e.level);
      break;
    case "result":
      showResult(e);
      if (e.meta && e.meta.mode === "incremental") {
        incrUpdateStats(e.meta.num_images, e.meta.output);
      }
      log(`✅ done: ${e.width}×${e.height} in ${e.meta.elapsed}s`);
      break;
    case "error":
      showBanner(e.message);
      log("ERROR: " + e.message, "error");
      break;
    case "end":
      setConn("idle");
      resetButton();
      setProgress(1, "Done");
      state.running = false;
      break;
  }
}

/* ============================ projects ============================ */
async function saveProject() {
  if (!state.jobId) {
    showBanner("Run a stitch first, then save it as a project.");
    return;
  }
  const name = $("project-name").value.trim();
  try {
    const res = await fetch("/api/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: state.jobId, name }),
    });
    const data = await res.json();
    if (data.error) { showBanner(data.error); return; }
    log(`💾 saved project "${data.name}"`);
    $("project-name").value = "";
    loadProjects();
  } catch (err) {
    showBanner("Save failed: " + err.message);
  }
}

async function loadProjects() {
  try {
    const res = await fetch("/api/projects");
    const data = await res.json();
    const wrap = $("projects-list");
    wrap.innerHTML = "";
    if (!data.projects || !data.projects.length) {
      wrap.innerHTML = '<div class="empty">No saved projects yet.</div>';
      return;
    }
    data.projects.forEach((p) => {
      const el = document.createElement("div");
      el.className = "project";
      el.innerHTML = `
        <div class="thumb">${p.thumbnail ? `<img src="${p.thumbnail}" />` : ""}</div>
        <div class="info">
          <div class="name">${p.name}</div>
          <div class="meta">${p.engine}${p.weights ? "/" + p.weights : ""} ·
            ${p.num_images} images · ${(p.created_at || "").slice(0, 16).replace("T", " ")}</div>
        </div>
        <div class="actions">
          <button title="Load">↺ Load</button>
          <button title="Rename">✎</button>
          <button class="danger" title="Delete">🗑</button>
        </div>`;
      el.querySelector(".actions button:nth-child(1)").onclick = () => openProject(p.id);
      el.querySelector(".actions button:nth-child(2)").onclick = () => renameProject(p.id, p.name);
      el.querySelector(".actions button:nth-child(3)").onclick = () => deleteProject(p.id);
      wrap.appendChild(el);
    });
  } catch (err) {
    // ignore list errors
  }
}

async function openProject(pid) {
  try {
    const res = await fetch(`/api/projects/${pid}`);
    const data = await res.json();
    if (data.error) { showBanner(data.error); return; }
    resetUI();
    state.maxSeq = 0;
    state.projectId = pid;
    state.jobId = null;
    setConn("idle");
    $("project-name").value = data.meta.name || "";
    log(`↺ loaded project "${data.meta.name}"`);
    data.events.forEach((e) => handleEvent(e));
  } catch (err) {
    showBanner("Load failed: " + err.message);
  }
}

async function renameProject(pid, current) {
  const name = (window.prompt("Rename project", current) || "").trim();
  if (!name) return;
  try {
    await fetch(`/api/projects/${pid}/rename`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    loadProjects();
  } catch (err) { showBanner("Rename failed: " + err.message); }
}

async function deleteProject(pid) {
  if (!window.confirm("Delete this project?")) return;
  try {
    await fetch(`/api/projects/${pid}`, { method: "DELETE" });
    if (state.projectId === pid) { resetUI(); state.projectId = null; }
    loadProjects();
  } catch (err) { showBanner("Delete failed: " + err.message); }
}

/* ============================ incremental mode ============================ */
function incrOptions() {
  return {
    filename_template: $("filename-template").value,
    engine: $("engine").value,
    weights: $("weights").value,
    max_keypoints: $("max-keypoints").value,
    match_threshold: $("match-threshold").value,
    feature_max_dim: $("feature-max-dim").value,
    sinkhorn_iterations: $("sinkhorn-iterations").value,
    alignment: $("alignment").value,
    ransac_thresh: $("ransac-thresh").value,
    refine: $("refine").checked,
    blend_levels: $("blend-levels").value,
    exposure: $("exposure").checked,
    min_extend_ratio: ($("incr-min-extend").value / 100),  // % -> ratio
    min_extend_px: $("incr-min-extend-px").value,
  };
}

function incrForm() {
  const fd = new FormData();
  Object.entries(incrOptions()).forEach(([k, v]) => fd.append(k, v));
  return fd;
}

function incrShowSession(sid) {
  state.incrSession = sid;
  $("incr-seed-wrap").hidden = true;
  $("incr-session").hidden = false;
  $("incr-download").onclick = () => {
    window.location.href = `/api/incremental/${sid}/map.jpg`;
  };
}

function incrUpdateStats(count, dims) {
  const items = [["Images", count || "—"], ["Map", dims ? `${dims.w}×${dims.h}` : "—"]];
  const s = $("incr-stats");
  s.innerHTML = "";
  items.forEach(([k, v]) => {
    const d = document.createElement("div");
    d.className = "stat";
    d.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div>`;
    s.appendChild(d);
  });
}

async function incrSeed(file) {
  const template = $("filename-template").value;
  if (!await validateIncrementalName(file, template)) return;
  resetUI();
  state.maxSeq = 0;
  $("incr-msg").textContent = "Creating map…";
  const fd = incrForm();
  fd.set("filename_template", template);
  fd.append("file", file, file.name);
  try {
    const res = await fetch("/api/incremental/start", { method: "POST", body: fd });
    const data = await res.json();
    if (!data.ok) { showBanner(data.message); $("incr-msg").textContent = ""; return; }
    data.events.forEach((e) => handleEvent(e));
    state.incrTemplate = template;
    $("filename-template").disabled = true;
    incrShowSession(data.session_id);
    $("incr-msg").textContent = data.message;
    log("🧩 started incremental map: " + data.session_id);
  } catch (err) {
    showBanner("Seed failed: " + err.message);
    $("incr-msg").textContent = "";
  }
}

async function incrAdd(file) {
  if (!state.incrSession) return;
  if (!await validateIncrementalName(file, state.incrTemplate)) return;
  const btn = $("incr-add-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spin"></span>Matching…';
  $("incr-msg").textContent = "Matching to map…";
  const fd = new FormData();
  fd.append("session_id", state.incrSession);
  fd.append("file", file, file.name);
  try {
    const res = await fetch("/api/incremental/add", { method: "POST", body: fd });
    const data = await res.json();
    if (data.ok) {
      data.events.forEach((e) => handleEvent(e));
      $("incr-msg").textContent = data.message;
    } else {
      data.events.forEach((e) => handleEvent(e));
      $("incr-msg").textContent = "✗ " + data.message;
      showBanner(data.message);
    }
  } catch (err) {
    showBanner("Add failed: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = "➕ Add next image";
  }
}

async function incrReset() {
  if (state.incrSession) {
    try { await fetch(`/api/incremental/${state.incrSession}`, { method: "DELETE" }); }
    catch (err) { /* ignore */ }
  }
  state.incrSession = null;
  state.incrTemplate = null;
  $("filename-template").disabled = false;
  $("incr-seed-wrap").hidden = false;
  $("incr-session").hidden = true;
  $("incr-msg").textContent = "";
  resetUI();
}

/* ============================ boot ============================ */
$("preset").addEventListener("change", (e) => applyPreset(e.target.value));
$("save-project").addEventListener("click", saveProject);

// incremental file inputs
$("incr-seed").addEventListener("click", () => $("incr-seed-input").click());
$("incr-seed-input").addEventListener("change", (e) => {
  if (e.target.files.length) incrSeed(e.target.files[0]);
  e.target.value = "";
});
$("incr-add-btn").addEventListener("click", () => $("incr-add-input").click());
$("incr-add-input").addEventListener("change", (e) => {
  if (e.target.files.length) incrAdd(e.target.files[0]);
  e.target.value = "";
});
$("incr-reset").addEventListener("click", incrReset);

applyPreset("balanced");
loadProjects();
