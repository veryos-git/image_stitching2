"use strict";

/* ============================== state ============================== */
const state = {
  images: [],          // File[] in stitching order
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
function renderThumbs() {
  const wrap = $("thumbs");
  wrap.innerHTML = "";
  state.images.forEach((f, i) => {
    const t = document.createElement("div");
    t.className = "thumb";
    t.draggable = true;
    t.dataset.index = i;
    const img = document.createElement("img");
    img.src = URL.createObjectURL(f);
    t.innerHTML = `<span class="idx">${i + 1}</span>
      <button class="del" title="remove">×</button>`;
    t.prepend(img);
    t.querySelector(".del").onclick = (e) => {
      e.stopPropagation();
      state.images.splice(i, 1);
      renderThumbs();
    };
    t.addEventListener("dragstart", (e) => {
      t.classList.add("dragging");
      e.dataTransfer.setData("text/plain", String(i));
    });
    t.addEventListener("dragend", () => t.classList.remove("dragging"));
    t.addEventListener("dragover", (e) => e.preventDefault());
    t.addEventListener("drop", (e) => {
      e.preventDefault();
      const from = parseInt(e.dataTransfer.getData("text/plain"), 10);
      const to = i;
      if (from !== to) {
        const [f] = state.images.splice(from, 1);
        state.images.splice(to, 0, f);
        renderThumbs();
      }
    });
    wrap.appendChild(t);
  });
}

function addFiles(list) {
  for (const f of list) {
    if (f.type.startsWith("image/")) state.images.push(f);
  }
  renderThumbs();
}

const dz = $("dropzone");
dz.addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", (e) => addFiles(e.target.files));
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
    engine: $("engine").value,
    weights: $("weights").value,
    max_keypoints: $("max-keypoints").value,
    match_threshold: $("match-threshold").value,
    feature_max_dim: $("feature-max-dim").value,
    sinkhorn_iterations: $("sinkhorn-iterations").value,
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
  clearBanner();
  resetUI();
  state.running = true;
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
    if (data.error) { showBanner(data.error); resetButton(); return; }
    state.jobId = data.job_id;
    connect(data.job_id);
  } catch (err) {
    showBanner("Upload failed: " + err.message);
    resetButton();
  }
});

function resetButton() {
  state.running = false;
  $("stitch-btn").disabled = false;
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
      buildTimeline(e.stages);
      log(`engine=${e.meta.engine}, weights=${e.meta.weights}, images=${e.meta.num_images}`);
      break;
    case "step":
      setMarker(e.marker, e.status, e.message, e.fraction);
      setProgress(e.progress, e.message || $("p-label").textContent);
      if (e.status === "done") log(`✓ ${e.marker}: ${e.message || ""}`);
      else if (e.status === "error") log(`✗ ${e.marker}: ${e.message || ""}`, "error");
      else log(`${e.marker}: ${e.message || ""}`);
      break;
    case "image":
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
    engine: $("engine").value,
    weights: $("weights").value,
    max_keypoints: $("max-keypoints").value,
    match_threshold: $("match-threshold").value,
    feature_max_dim: $("feature-max-dim").value,
    sinkhorn_iterations: $("sinkhorn-iterations").value,
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
  resetUI();
  state.maxSeq = 0;
  $("incr-msg").textContent = "Creating map…";
  const fd = incrForm();
  fd.append("file", file, file.name);
  try {
    const res = await fetch("/api/incremental/start", { method: "POST", body: fd });
    const data = await res.json();
    if (!data.ok) { showBanner(data.message); $("incr-msg").textContent = ""; return; }
    data.events.forEach((e) => handleEvent(e));
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
