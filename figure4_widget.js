/* ── figure4_widget.js ──────────────────────────────────────────────
   Canvas-rendered interactive SAE feature neighborhood explorer.
   Reads `neighbor_data` (JSON array) and `anchor_label` from model.
   Writes `selected_feature` back on click.
   ────────────────────────────────────────────────────────────────── */

// ── color helpers ──────────────────────────────────────────────────
function viridis(t) {
  // Simplified viridis: dark purple → teal → yellow
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(255 * Math.min(1, Math.max(0, -0.35 + 3.2 * t - 2.8 * t * t + 0.95 * t * t * t)));
  const g = Math.round(255 * Math.min(1, Math.max(0, -0.05 + 0.7 * t + 1.5 * t * t - 1.15 * t * t * t)));
  const b = Math.round(255 * Math.min(1, Math.max(0, 0.5 + 0.6 * t - 2.8 * t * t + 1.7 * t * t * t)));
  return `rgb(${r},${g},${b})`;
}

function viridisHex(t) {
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(255 * Math.min(1, Math.max(0, -0.35 + 3.2 * t - 2.8 * t * t + 0.95 * t * t * t)));
  const g = Math.round(255 * Math.min(1, Math.max(0, -0.05 + 0.7 * t + 1.5 * t * t - 1.15 * t * t * t)));
  const b = Math.round(255 * Math.min(1, Math.max(0, 0.5 + 0.6 * t - 2.8 * t * t + 1.7 * t * t * t)));
  return `#${r.toString(16).padStart(2,'0')}${g.toString(16).padStart(2,'0')}${b.toString(16).padStart(2,'0')}`;
}

const DIM_COLORS = { "2^13": "#60a5fa", "2^14": "#f59e0b", "2^16": "#34d399" };
const DIM_SIZES  = { "2^13": 9, "2^14": 6, "2^16": 3.5 };
const DIM_LABELS = { "2^13": "2¹³ (8,192)", "2^14": "2¹⁴ (16,384)", "2^16": "2¹⁶ (65,536)" };

const CATEGORY_COLORS = {
  "Catalytic function": "#ef4444", "Functional site": "#ef4444",
  "Structural motif": "#22c55e", "Secondary structure": "#22c55e",
  "Tertiary interaction": "#a855f7", "Domain / fold": "#f97316",
  "Biochemical environment": "#eab308", "Localization / topology": "#06b6d4",
  "Catalog feature": "#6b7280", "Uncategorized": "#6b7280",
};

function getPointColor(row) {
  if (row.is_anchor) return "#f59e0b";
  if (row.sae_dim === "2^14" && CATEGORY_COLORS[row.category]) {
    return CATEGORY_COLORS[row.category];
  }
  return DIM_COLORS[row.sae_dim] || "#6b7280";
}

// ── DOM builder ────────────────────────────────────────────────────
function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "style" && typeof v === "object") {
        Object.assign(el.style, v);
      } else if (k.startsWith("on")) {
        el.addEventListener(k.slice(2).toLowerCase(), v);
      } else {
        el.setAttribute(k, v);
      }
    }
  }
  for (const child of children) {
    if (typeof child === "string") el.appendChild(document.createTextNode(child));
    else if (child) el.appendChild(child);
  }
  return el;
}

// ── main render ────────────────────────────────────────────────────
function render({ model, el }) {
  let data = [];
  let visibleDims = new Set(["2^13", "2^14", "2^16"]);
  let hoveredIndex = -1;
  let selectedIndex = -1;
  let transform = { sx: 1, sy: 1, tx: 0, ty: 0 };

  // ── build DOM ──
  const root = h("div", { class: "fn-root" });

  // header
  const headerTitle = h("div", { class: "fn-header-title" });
  const chipsContainer = h("div", { class: "fn-chips" });
  const statsBar = h("div", { class: "fn-stats-bar" });
  const header = h("div", { class: "fn-header" }, headerTitle, chipsContainer, statsBar);

  // scatter panel
  const canvas = h("canvas", { class: "fn-canvas" });
  const tooltip = h("div", { class: "fn-tooltip" });
  const legend = h("div", { class: "fn-legend" });
  const colorbar = h("div", { class: "fn-colorbar" });
  const scatterPanel = h("div", { class: "fn-scatter-panel" }, canvas, tooltip, legend, colorbar);

  // detail panel
  const detailPanel = h("div", { class: "fn-detail-panel" });

  root.appendChild(header);
  root.appendChild(scatterPanel);
  root.appendChild(detailPanel);
  el.appendChild(root);

  const ctx = canvas.getContext("2d");

  // ── parse data ──
  function loadData() {
    const raw = model.get("neighbor_data");
    try {
      data = typeof raw === "string" ? JSON.parse(raw) : (raw || []);
    } catch {
      data = [];
    }
  }

  // ── compute transform ──
  function computeTransform() {
    if (data.length === 0) return;
    const visible = data.filter(r => visibleDims.has(r.sae_dim));
    if (visible.length === 0) return;

    const xs = visible.map(r => r.x);
    const ys = visible.map(r => r.y);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const rangeX = xMax - xMin || 1;
    const rangeY = yMax - yMin || 1;
    const pad = 40;
    const w = canvas.width - 2 * pad;
    const h = canvas.height - 2 * pad;
    const scale = Math.min(w / rangeX, h / rangeY);
    transform = {
      sx: scale,
      sy: -scale,
      tx: pad + (w - rangeX * scale) / 2 - xMin * scale,
      ty: canvas.height - pad - (h - rangeY * scale) / 2 + yMin * scale,
    };
  }

  function toScreen(x, y) {
    return [x * transform.sx + transform.tx, y * transform.sy + transform.ty];
  }

  // ── draw ──
  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const cw = Math.floor(rect.width);
    const ch = Math.floor(rect.height);
    if (canvas.width !== cw * dpr || canvas.height !== ch * dpr) {
      canvas.width = cw * dpr;
      canvas.height = ch * dpr;
      canvas.style.width = cw + "px";
      canvas.style.height = ch + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    ctx.clearRect(0, 0, cw, ch);

    computeTransform();

    // find cosine range for color scale
    const visible = data.filter(r => visibleDims.has(r.sae_dim));
    const cosines = visible.map(r => r.cosine);
    const cosMin = cosines.length ? Math.min(...cosines) : 0;
    const cosMax = cosines.length ? Math.max(...cosines) : 1;
    const cosRange = cosMax - cosMin || 1;

    // draw order: 2^16 first (smallest), then 2^14, then 2^13 (largest), anchor last
    const sortOrder = { "2^16": 0, "2^14": 1, "2^13": 2 };
    const sorted = [...visible].sort((a, b) => {
      if (a.is_anchor) return 1;
      if (b.is_anchor) return -1;
      return (sortOrder[a.sae_dim] || 0) - (sortOrder[b.sae_dim] || 0);
    });

    for (let i = 0; i < sorted.length; i++) {
      const row = sorted[i];
      const idx = data.indexOf(row);
      const [sx, sy] = toScreen(row.x, row.y);
      const baseSize = DIM_SIZES[row.sae_dim] || 5;
      const size = row.is_anchor ? 12 : baseSize;

      const isHovered = idx === hoveredIndex;
      const isSelected = idx === selectedIndex;
      const drawSize = isHovered ? size * 1.5 : isSelected ? size * 1.3 : size;

      ctx.beginPath();

      if (row.is_anchor) {
        // draw star
        drawStar(ctx, sx, sy, 5, drawSize, drawSize * 0.5);
      } else {
        ctx.arc(sx, sy, drawSize, 0, Math.PI * 2);
      }

      // color by cosine (viridis) mixed with category
      const t = (row.cosine - cosMin) / cosRange;
      const color = getPointColor(row);
      ctx.fillStyle = color;
      ctx.globalAlpha = row.is_anchor ? 1.0 : (isHovered || isSelected) ? 0.95 : 0.8;
      ctx.fill();

      if (isHovered || isSelected || row.is_anchor) {
        ctx.strokeStyle = isHovered ? "#ffffff" : isSelected ? "#7c8aff" : "#ffffff";
        ctx.lineWidth = isHovered ? 2 : 1.5;
        ctx.stroke();
      }

      ctx.globalAlpha = 1.0;
    }
  }

  function drawStar(ctx, cx, cy, spikes, outerR, innerR) {
    let rot = -Math.PI / 2;
    let step = Math.PI / spikes;
    ctx.moveTo(cx + Math.cos(rot) * outerR, cy + Math.sin(rot) * outerR);
    for (let i = 0; i < spikes; i++) {
      ctx.lineTo(cx + Math.cos(rot) * outerR, cy + Math.sin(rot) * outerR);
      rot += step;
      ctx.lineTo(cx + Math.cos(rot) * innerR, cy + Math.sin(rot) * innerR);
      rot += step;
    }
    ctx.closePath();
  }

  // ── hit test ──
  function hitTest(mx, my) {
    let bestDist = Infinity;
    let bestIdx = -1;
    for (let i = 0; i < data.length; i++) {
      const row = data[i];
      if (!visibleDims.has(row.sae_dim)) continue;
      const [sx, sy] = toScreen(row.x, row.y);
      const r = (DIM_SIZES[row.sae_dim] || 5) * 2;
      const dx = mx - sx;
      const dy = my - sy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < r && dist < bestDist) {
        bestDist = dist;
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  // ── tooltip ──
  function showTooltip(idx, mx, my) {
    if (idx < 0) {
      tooltip.classList.remove("visible");
      return;
    }
    const row = data[idx];
    tooltip.innerHTML = `
      <div>
        <span class="fn-tooltip-id">${esc(row.feature_id)}</span>
        <span class="fn-tooltip-dim">${esc(row.sae_dim)}</span>
      </div>
      <div class="fn-tooltip-label">${esc(row.label)}</div>
      <div class="fn-tooltip-cosine">cosine <strong>${row.cosine.toFixed(3)}</strong></div>
      ${row.category && row.category !== row.sae_dim + " feature" ?
        `<div class="fn-tooltip-cosine">${esc(row.category)}</div>` : ""}
    `;
    tooltip.classList.add("visible");

    // position
    const rect = scatterPanel.getBoundingClientRect();
    let left = mx + 14;
    let top = my - 10;
    if (left + 260 > rect.width) left = mx - 260;
    if (top + tooltip.offsetHeight > rect.height) top = rect.height - tooltip.offsetHeight - 10;
    if (top < 0) top = 10;
    tooltip.style.left = left + "px";
    tooltip.style.top = top + "px";
  }

  function esc(s) {
    const d = document.createElement("div");
    d.textContent = String(s || "");
    return d.innerHTML;
  }

  // ── detail panel ──
  function renderDetail(idx) {
    detailPanel.innerHTML = "";
    if (idx < 0) {
      detailPanel.innerHTML = `
        <div class="fn-detail-empty">
          <div class="fn-detail-empty-icon">⬡</div>
          <div>Click a feature dot to inspect</div>
          <div style="font-size:11px;color:var(--text-muted)">Hover for quick preview</div>
        </div>`;
      return;
    }

    const row = data[idx];
    const neighbors = data
      .filter(r => visibleDims.has(r.sae_dim) && r !== row)
      .sort((a, b) => b.cosine - a.cosine)
      .slice(0, 15);

    // count by dim
    const counts = {};
    for (const r of data) {
      counts[r.sae_dim] = (counts[r.sae_dim] || 0) + 1;
    }

    // header
    const detailHeader = document.createElement("div");
    detailHeader.className = "fn-detail-header";
    detailHeader.innerHTML = `
      <div class="fn-detail-fid">${esc(row.feature_id)}${row.is_anchor ? ' ★' : ''}</div>
      <div class="fn-detail-label">${esc(row.label)}</div>
      <div class="fn-detail-badges">
        <span class="fn-badge fn-badge-dim">${esc(row.sae_dim)}</span>
        ${row.category && row.category !== row.sae_dim + " feature" ?
          `<span class="fn-badge fn-badge-category">${esc(row.category)}</span>` : ""}
      </div>`;

    // body
    const detailBody = document.createElement("div");
    detailBody.className = "fn-detail-body";

    // metrics
    const metricsHtml = `
      <div class="fn-metrics">
        <div class="fn-metric">
          <div class="fn-metric-value">${row.cosine.toFixed(3)}</div>
          <div class="fn-metric-label">Cosine</div>
        </div>
        <div class="fn-metric">
          <div class="fn-metric-value">${row.feature_index}</div>
          <div class="fn-metric-label">Index</div>
        </div>
        <div class="fn-metric">
          <div class="fn-metric-value">${row.is_anchor ? "★" : "—"}</div>
          <div class="fn-metric-label">Anchor</div>
        </div>
      </div>`;

    // description
    const descHtml = row.summary || row.description
      ? `<div>
           <div class="fn-detail-section-title">Description</div>
           <div class="fn-detail-description">${esc(row.summary || row.description)}</div>
         </div>`
      : "";

    // activation bar (cosine-based gradient as proxy)
    const actBarHtml = `
      <div class="fn-activation-bar-container">
        <div class="fn-detail-section-title">Feature activation · cosine ${row.cosine.toFixed(3)}</div>
        <div class="fn-activation-bar">
          <canvas id="fn-act-bar-canvas" width="340" height="20"></canvas>
        </div>
      </div>`;

    // neighbor list
    let neighborHtml = `<div class="fn-detail-section-title">Top neighbors · ${neighbors.length} features</div>
      <div class="fn-neighbor-list">`;
    for (const n of neighbors) {
      const sel = data.indexOf(n) === selectedIndex ? " selected" : "";
      neighborHtml += `
        <div class="fn-neighbor-row${sel}" data-idx="${data.indexOf(n)}">
          <span class="fn-neighbor-fid">${esc(n.feature_id)}</span>
          <span class="fn-neighbor-label" title="${esc(n.label)}">${esc(truncate(n.label, 32))}</span>
          <span class="fn-neighbor-cosine">${n.cosine.toFixed(3)}</span>
        </div>`;
    }
    neighborHtml += "</div>";

    detailBody.innerHTML = metricsHtml + descHtml + actBarHtml + neighborHtml;

    detailPanel.appendChild(detailHeader);
    detailPanel.appendChild(detailBody);

    // draw activation bar
    requestAnimationFrame(() => {
      const actCanvas = detailPanel.querySelector("#fn-act-bar-canvas");
      if (!actCanvas) return;
      const actCtx = actCanvas.getContext("2d");
      const w = actCanvas.width;
      for (let px = 0; px < w; px++) {
        const t = px / w;
        // fake a bump activation curve centered at the cosine value
        const activation = Math.exp(-((t - 0.5) * (t - 0.5)) / (0.08 * row.cosine));
        actCtx.fillStyle = viridis(activation * row.cosine);
        actCtx.fillRect(px, 0, 1, 20);
      }
    });

    // click handlers for neighbor rows
    requestAnimationFrame(() => {
      for (const rowEl of detailPanel.querySelectorAll(".fn-neighbor-row")) {
        rowEl.addEventListener("click", () => {
          const ni = parseInt(rowEl.dataset.idx);
          if (ni >= 0 && ni < data.length) {
            selectedIndex = ni;
            model.set("selected_feature", data[ni].feature_id);
            model.save_changes();
            draw();
            renderDetail(ni);
          }
        });
      }
    });
  }

  function truncate(s, n) {
    if (!s) return "";
    return s.length > n ? s.slice(0, n) + "…" : s;
  }

  // ── build header ──
  function buildHeader() {
    const anchorLabel = model.get("anchor_label") || "Feature neighborhood";
    headerTitle.innerHTML = `SAE Decoder Neighborhood · <span>${esc(anchorLabel)}</span>`;

    chipsContainer.innerHTML = "";
    for (const dim of ["2^13", "2^14", "2^16"]) {
      const count = data.filter(r => r.sae_dim === dim).length;
      if (count === 0) continue;
      const chip = document.createElement("button");
      chip.className = "fn-chip" + (visibleDims.has(dim) ? " active" : "");
      chip.innerHTML = `<span class="chip-dot" style="background:${DIM_COLORS[dim]}"></span>${DIM_LABELS[dim]} · ${count}`;
      chip.addEventListener("click", () => {
        if (visibleDims.has(dim)) visibleDims.delete(dim);
        else visibleDims.add(dim);
        chip.classList.toggle("active");
        draw();
      });
      chipsContainer.appendChild(chip);
    }

    statsBar.innerHTML = `<strong>${data.length}</strong> features · cosine ≥ 0.1 · UMAP`;
  }

  // ── build legend ──
  function buildLegend() {
    legend.innerHTML = "";
    for (const [dim, color] of Object.entries(DIM_COLORS)) {
      const count = data.filter(r => r.sae_dim === dim).length;
      if (count === 0) continue;
      const size = DIM_SIZES[dim] * 2;
      legend.innerHTML += `
        <div class="fn-legend-item">
          <span class="fn-legend-dot" style="background:${color};width:${size}px;height:${size}px"></span>
          ${dim}
        </div>`;
    }
    // anchor
    legend.innerHTML += `
      <div class="fn-legend-item">
        <span style="color:#f59e0b;font-size:14px">★</span>
        anchor
      </div>`;
  }

  // ── build colorbar ──
  function buildColorbar() {
    colorbar.innerHTML = `
      <span>0.1</span>
      <canvas class="fn-colorbar-gradient" width="80" height="8"></canvas>
      <span>1.0</span>
      <span style="margin-left:4px">cosine</span>`;
    requestAnimationFrame(() => {
      const gradCanvas = colorbar.querySelector("canvas");
      if (!gradCanvas) return;
      const gctx = gradCanvas.getContext("2d");
      for (let px = 0; px < 80; px++) {
        gctx.fillStyle = viridis(px / 80);
        gctx.fillRect(px, 0, 1, 8);
      }
    });
  }

  // ── events ──
  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const idx = hitTest(mx, my);
    if (idx !== hoveredIndex) {
      hoveredIndex = idx;
      draw();
      showTooltip(idx, mx, my);
      canvas.style.cursor = idx >= 0 ? "pointer" : "crosshair";
    } else if (idx >= 0) {
      showTooltip(idx, mx, my);
    }
  });

  canvas.addEventListener("mouseleave", () => {
    hoveredIndex = -1;
    tooltip.classList.remove("visible");
    draw();
  });

  canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const idx = hitTest(mx, my);
    if (idx >= 0) {
      selectedIndex = idx;
      model.set("selected_feature", data[idx].feature_id);
      model.save_changes();
      draw();
      renderDetail(idx);
    }
  });

  // resize
  const ro = new ResizeObserver(() => {
    draw();
  });
  ro.observe(scatterPanel);

  // ── init ──
  function init() {
    loadData();
    buildHeader();
    buildLegend();
    buildColorbar();
    computeTransform();
    draw();

    // auto-select anchor
    const anchorIdx = data.findIndex(r => r.is_anchor);
    if (anchorIdx >= 0) {
      selectedIndex = anchorIdx;
      renderDetail(anchorIdx);
    } else {
      renderDetail(-1);
    }
  }

  init();

  // ── react to model changes ──
  model.on("change:neighbor_data", () => {
    loadData();
    buildHeader();
    buildLegend();
    buildColorbar();
    hoveredIndex = -1;
    selectedIndex = -1;
    computeTransform();
    draw();
    renderDetail(-1);
  });

  model.on("change:anchor_label", () => {
    buildHeader();
  });
}

export default { render };
