// vecsvc dashboard — vanilla JS, no external dependencies.

const $ = (id) => document.getElementById(id);
const fmt = (n, d = 1) => Number(n).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d });
const humanBytes = (b) => {
  if (!b) return "–";
  const u = ["B", "KiB", "MiB", "GiB", "TiB"];
  let i = 0;
  while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
  return fmt(b, b < 10 ? 2 : 1) + " " + u[i];
};
const strategyTag = (s) => {
  if (!s) return "";
  const cls = s.startsWith("graph") ? "graph" : s.startsWith("hybrid") ? "hybrid" : "brute";
  return `<span class="tag ${cls}">${s}</span>`;
};

// ---------- generic canvas line chart ----------
function drawChart(canvasId, series, opts = {}) {
  const cv = $(canvasId);
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  const pad = { l: 52, r: opts.rightAxis ? 52 : 16, t: 14, b: 42 };
  const cw = w - pad.l - pad.r, ch = h - pad.t - pad.b;

  const xs = series[0].points.map((_, i) => i);
  const allY = series.flatMap(s => s.points.map(p => p.y));
  let yMin = opts.yMin ?? Math.min(...allY), yMax = opts.yMax ?? Math.max(...allY);
  if (opts.zeroOne) { yMin = 0; yMax = 1; }
  if (yMin === yMax) { yMax = yMin + 1; }
  const x = (i) => pad.l + (xs.length <= 1 ? cw / 2 : (i / (xs.length - 1)) * cw);
  const y = (v) => pad.t + ch - ((v - yMin) / (yMax - yMin)) * ch;

  // grid + y labels
  ctx.strokeStyle = "#2d3748"; ctx.fillStyle = "#8b949e"; ctx.lineWidth = 1;
  ctx.font = "11px sans-serif"; ctx.textAlign = "right";
  for (let g = 0; g <= 4; g++) {
    const v = yMin + (yMax - yMin) * g / 4;
    const yy = y(v);
    ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(w - pad.r, yy); ctx.stroke();
    ctx.fillText(opts.pct ? fmt(v * 100, 0) + "%" : fmt(v, opts.yDigits ?? 1), pad.l - 6, yy + 3);
  }
  // x labels
  ctx.textAlign = "center";
  series[0].points.forEach((p, i) => {
    ctx.fillText(p.label, x(i), h - pad.b + 16);
  });

  // series lines
  series.forEach((s) => {
    ctx.strokeStyle = s.color; ctx.fillStyle = s.color; ctx.lineWidth = 2;
    ctx.beginPath();
    s.points.forEach((p, i) => { i ? ctx.lineTo(x(i), y(p.y)) : ctx.moveTo(x(i), y(p.y)); });
    ctx.stroke();
    s.points.forEach((p, i) => {
      ctx.beginPath(); ctx.arc(x(i), y(p.y), 3.2, 0, Math.PI * 2); ctx.fill();
    });
  });

  // legend values on hover (title attr)
  cv.title = series.map(s => s.name + ": " + s.points.map(p => `${p.label}=${fmt(p.y, 2)}`).join("  ")).join("\n");
}

// ---------- stats ----------
async function loadStats() {
  const s = await (await fetch("/api/stats")).json();
  $("statsPill").textContent = `节点 ${s.index.live_nodes}/${s.index.total_nodes} · 顶层 L${s.index.top_layer} · 修复 ${s.index.repair_runs} 次`;
  $("mNodes").innerHTML = `${s.index.live_nodes.toLocaleString()} <small>/ ${s.index.total_nodes.toLocaleString()}</small>`;
  $("mNodesSub").textContent = `顶层 L${s.index.top_layer} · 底层平均度 ${fmt(s.index.mean_degree0, 2)} · 墓碑 ${fmt(s.index.deleted_fraction * 100, 1)}%`;
  $("mMem").textContent = humanBytes(s.mem_heap_bytes);
  $("mMemSub").textContent = `原始 float32 向量 ${humanBytes(s.raw_vector_bytes)} · 堆含图连边与属性`;
  if (s.has_pq) {
    const ratio = s.raw_vector_bytes / s.pq_code_bytes;
    $("mPQ").innerHTML = `${fmt(ratio, 1)}<small> x</small>`;
    $("mPQSub").textContent = `PQ 编码 ${humanBytes(s.pq_code_bytes)} vs 原始 ${humanBytes(s.raw_vector_bytes)}`;
  } else {
    $("mPQ").textContent = "未启用";
    $("mPQSub").textContent = "启动 serve 时指定 -pqm 开启";
  }
}

// ---------- filter cases ----------
async function loadFilters() {
  const cases = await (await fetch("/api/filters/cases")).json();
  $("filterSel").innerHTML = cases.map(c =>
    `<option value="${c.name}">${c.name} · 选择率 ${fmt(c.actual_selectivity, 5)}</option>`).join("");
  window._filterCases = cases;
}

// ---------- benchmark ----------
async function runBench() {
  const body = {
    n: parseInt($("benchN").value),
    queries_per_case: parseInt($("benchQ").value),
    pq_m: parseInt($("benchPQ").value),
    delete_fraction: parseFloat($("benchDel").value),
  };
  $("benchStatus").innerHTML = '<span class="warn">基准运行中，请稍候（构建索引+暴力真值+删除实验）…</span>';
  const t0 = performance.now();
  try {
    const rep = await (await fetch("/api/bench/run", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    })).json();
    if (rep.error) throw new Error(rep.error);
    $("benchStatus").innerHTML = `<span class="ok">完成，耗时 ${fmt((performance.now() - t0) / 1000, 1)}s</span>`;
    renderReport(rep);
  } catch (e) {
    $("benchStatus").innerHTML = `<span class="bad">失败: ${e.message}</span>`;
  }
}

async function loadReport() {
  const r = await fetch("/api/report");
  if (!r.ok) { $("benchStatus").textContent = "服务端暂无报告"; return; }
  renderReport(await r.json());
}

function renderReport(rep) {
  window._report = rep;
  $("reportBlocks").style.display = "block";
  const labels = rep.cases.map(c => c.name.replace(/_/g, "_"));
  const shortLabels = rep.cases.map((c, i) => `S${i}`);

  drawChart("recallChart", [
    { name: "精确", color: "#4493f8", points: rep.cases.map((c, i) => ({ label: shortLabels[i], y: c.recall })) },
    ...(rep.pq ? [{ name: "PQ", color: "#bc8cff", points: rep.cases.map((c, i) => ({ label: shortLabels[i], y: c.recall_pq || 0 })) }] : []),
  ], { zeroOne: true, pct: true });

  drawChart("latencyChart", [
    { name: "精确P50", color: "#3fb950", points: rep.cases.map((c, i) => ({ label: shortLabels[i], y: c.p50_latency_us })) },
    ...(rep.pq ? [{ name: "PQ-P50", color: "#d29922", points: rep.cases.map((c, i) => ({ label: shortLabels[i], y: c.p50_latency_pq_us || 0 })) }] : []),
  ], { yDigits: 0 });

  drawChart("efChart", [
    { name: "召回", color: "#4493f8", points: rep.ef_sweep.map(e => ({ label: "ef" + e.ef, y: e.recall })) },
    { name: "P50µs", color: "#f85149", points: rep.ef_sweep.map(e => ({ label: "ef" + e.ef, y: e.p50_latency_us })) },
  ]);

  // case table
  $("caseTable").querySelector("tbody").innerHTML = rep.cases.map((c, i) => `
    <tr><td title="${labels[i]}">S${i} ${c.name}</td>
    <td>${fmt(c.selectivity, 5)}</td><td>${strategyTag(c.strategy)}</td>
    <td class="${c.recall > 0.9 ? "ok" : c.recall > 0.75 ? "warn" : "bad"}">${fmt(c.recall, 4)}</td>
    <td>${rep.pq ? fmt(c.recall_pq, 4) : "–"}</td>
    <td>${fmt(c.p50_latency_us, 0)}</td><td>${fmt(c.p99_latency_us, 0)}</td>
    <td>${fmt(c.mean_visited, 0)}</td></tr>`).join("");

  // build info
  const b = rep.build, pq = rep.pq;
  $("buildInfo").innerHTML = `
    <table><tbody>
    <tr><td>数据生成</td><td>${b.dataset_gen_ms} ms</td></tr>
    <tr><td>索引构建</td><td>${b.index_build_ms} ms（${fmt(b.insert_per_sec, 0)} inserts/s）</td></tr>
    <tr><td>堆内存</td><td>${humanBytes(b.heap_bytes)}</td></tr>
    <tr><td>原始向量</td><td>${humanBytes(b.raw_vector_bytes)}（${rep.config.n} × ${rep.config.dim} × float32）</td></tr>
    ${pq ? `<tr><td>PQ 训练 / 编码</td><td>${pq.train_ms} ms / ${pq.encode_ms} ms</td></tr>
    <tr><td>PQ 编码体积</td><td>${humanBytes(pq.code_bytes)}（压缩 ${fmt(pq.compression_x, 1)}x）</td></tr>` : ""}
    </tbody></table>`;

  // deletion info
  if (rep.deletion) {
    const d = rep.deletion;
    $("deleteInfo").innerHTML = `
    <table><tbody>
    <tr><td>删除比例</td><td>${fmt(d.deleted_fraction * 100, 1)}%</td></tr>
    <tr><td>删除耗时</td><td>${d.delete_ms} ms</td></tr>
    <tr><td>图修复（重建）次数</td><td>${d.repair_runs}</td></tr>
    <tr><td>删除前召回</td><td class="ok">${fmt(d.recall_before, 4)}</td></tr>
    <tr><td>修复后召回</td><td class="${d.recall_after > 0.9 ? "ok" : "warn"}">${fmt(d.recall_after, 4)}</td></tr>
    <tr><td>修复后 P50</td><td>${fmt(d.p50_latency_after_us, 0)} µs</td></tr>
    </tbody></table>
    <p class="muted" style="margin-bottom:0">删除先打墓碑标记，搜索直接跳过；墓碑比例超过 10% 触发全图局部重建，
    存活节点按原层级重新执行插入连边与启发式裁剪。</p>`;
  }
}

// ---------- single query + path ----------
async function runSearch() {
  const cases = window._filterCases || [];
  const sel = $("filterSel").selectedIndex;
  const fcase = cases[sel];
  const body = {
    k: parseInt($("qK").value),
    ef: parseInt($("qEf").value),
    use_pq: $("qPQ").checked,
    record_path: $("qPath").checked,
    query_seed: parseInt($("qSeed").value),
    filter: fcase && fcase.name.startsWith("unfiltered") ? null : (fcase ? fcase.filter : null),
  };
  const r = await fetch("/api/search", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await r.json();
  if (data.error) { alert(data.error); return; }

  $("strategyBox").innerHTML = `<span class="pill">选择率 ${fmt(data.selectivity, 5)}</span>
    ${strategyTag(data.result.strategy)}
    ${data.result.reranked_raw ? '<span class="tag hybrid">PQ粗排→原始精排</span>' : ""}
    <div class="muted" style="margin-top:6px">${data.strategy_reason}</div>
    <div class="muted">距离评估 ${data.result.visited} 个节点${data.result.brute_scanned ? ` · 暴力扫描 ${data.result.brute_scanned}` : ""}</div>`;

  const attrMap = {};
  if (data.result.ids.length) {
    const qs = data.result.ids.map(id => "id=" + id).join("&");
    const attrs = await (await fetch("/api/attrs?" + qs)).json();
    attrs.forEach(a => { attrMap[a.id] = a; });
  }
  $("resultTable").querySelector("tbody").innerHTML = data.result.ids.map((id, i) =>
    `<tr><td>${i + 1}</td><td>${id}</td><td>${fmt(data.result.dists[i], 3)}</td>
     <td>${attrMap[id] ? attrMap[id].tenant : "–"}</td><td>${attrMap[id] ? attrMap[id].category : "–"}</td></tr>`
  ).join("");
  drawPath(data.result.path || []);
  $("pathJSON").textContent = JSON.stringify((data.result.path || []).slice(0, 60), null, 1);
}

function drawPath(path) {
  const svg = $("pathSvg");
  const w = svg.clientWidth, h = svg.clientHeight;
  svg.innerHTML = "";
  const NS = "http://www.w3.org/2000/svg";
  if (!path.length) {
    const t = document.createElementNS(NS, "text");
    t.setAttribute("x", 12); t.setAttribute("y", 24); t.setAttribute("fill", "#8b949e");
    t.textContent = "无路径数据（暴力策略不经过图；勾选记录搜索路径再试）";
    svg.appendChild(t); return;
  }
  const layers = Math.max(...path.map(p => p.layer)) + 1;
  const padX = 40, padY = 26;
  const n = path.length;
  const X = (i) => padX + (n <= 1 ? (w - 2 * padX) / 2 : (i / (n - 1)) * (w - 2 * padX));
  const Y = (layer) => padY + (layers <= 1 ? (h - 2 * padY) / 2 : (layer / (layers - 1)) * (h - 2 * padY));

  // layer bands
  for (let l = 0; l < layers; l++) {
    const rect = document.createElementNS(NS, "rect");
    rect.setAttribute("x", 0); rect.setAttribute("width", w);
    rect.setAttribute("y", Y(l) - 11); rect.setAttribute("height", 22);
    rect.setAttribute("fill", l % 2 ? "rgba(255,255,255,0.02)" : "transparent");
    svg.appendChild(rect);
    const t = document.createElementNS(NS, "text");
    t.setAttribute("x", 6); t.setAttribute("y", Y(l) + 4); t.setAttribute("fill", "#8b949e");
    t.setAttribute("font-size", 10); t.textContent = "L" + l;
    svg.appendChild(t);
  }
  // edges (visit order)
  const line = document.createElementNS(NS, "polyline");
  line.setAttribute("points", path.map((p, i) => `${X(i)},${Y(p.layer)}`).join(" "));
  line.setAttribute("fill", "none"); line.setAttribute("stroke", "#30363d"); line.setAttribute("stroke-width", 1);
  svg.appendChild(line);
  // nodes
  path.forEach((p, i) => {
    const c = document.createElementNS(NS, "circle");
    c.setAttribute("cx", X(i)); c.setAttribute("cy", Y(p.layer)); c.setAttribute("r", p.from_entry ? 6 : 4);
    c.setAttribute("fill", p.from_entry ? "#4493f8" : p.match ? "#3fb950" : "#f85149");
    const title = document.createElementNS(NS, "title");
    title.textContent = `#${i} node=${p.id} layer=L${p.layer} dist=${fmt(p.dist, 2)} match=${p.match}${p.from_entry ? " entry" : ""}`;
    c.appendChild(title);
    svg.appendChild(c);
  });
}

window.addEventListener("resize", () => { if (window._report) renderReport(window._report); });
loadStats();
loadFilters();
setInterval(loadStats, 15000);
