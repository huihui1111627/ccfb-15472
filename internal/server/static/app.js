// 向量检索可视化前端: 纯原生 JS + SVG, 无外部依赖。
const $ = (s) => document.querySelector(s);
const COLORS = { accent:'#4ea1ff', green:'#59d29b', warn:'#ffb454', red:'#ff6b6b', purple:'#b48ef6' };

let REPORT = null;

// ---------- 轻量 SVG 折线图 ----------
function lineChart(container, series, opts={}) {
  const W = 640, H = 300, M = {l:56,r:20,t:20,b:44};
  const allX = series.flatMap(s => s.points.map(p => p.x));
  const allY = series.flatMap(s => s.points.map(p => p.y));
  const xmin = Math.min(...allX), xmax = Math.max(...allX);
  const ymin = opts.yMin ?? Math.min(...allY) * 0.95;
  const ymax = opts.yMax ?? Math.max(...allY) * 1.03;
  const X = (x) => M.l + (x-xmin)/Math.max(xmax-xmin,1e-9) * (W-M.l-M.r);
  const Y = (y) => H-M.b - (y-ymin)/Math.max(ymax-ymin,1e-9) * (H-M.t-M.b);
  const ticks = 5;
  let svg = `<svg viewBox="0 0 ${W} ${H}">`;
  for (let i=0;i<=ticks;i++) {
    const yv = ymin + (ymax-ymin)*i/ticks, yp = Y(yv);
    svg += `<line x1="${M.l}" y1="${yp}" x2="${W-M.r}" y2="${yp}" stroke="#2e3b4e" stroke-dasharray="3,3"/>
            <text x="${M.l-8}" y="${yp+4}" text-anchor="end" fill="#8b98a9" font-size="11">${yv.toFixed(opts.yDigits??3)}</text>`;
  }
  series[0].points.forEach(p => {
    svg += `<text x="${X(p.x)}" y="${H-M.b+18}" text-anchor="middle" fill="#8b98a9" font-size="11">${p.label}</text>`;
  });
  svg += `<line x1="${M.l}" y1="${H-M.b}" x2="${W-M.r}" y2="${H-M.b}" stroke="#2e3b4e"/>
          <line x1="${M.l}" y1="${M.t}" x2="${M.l}" y2="${H-M.b}" stroke="#2e3b4e"/>`;
  series.forEach(s => {
    const pts = s.points.map(p => `${X(p.x)},${Y(p.y)}`).join(' ');
    svg += `<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="2.5"/>`;
    s.points.forEach(p => {
      svg += `<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="4" fill="${s.color}">
        <title>${s.name}: ${p.label} = ${p.y.toFixed(4)}</title></circle>`;
    });
  });
  svg += `</svg>`;
  container.innerHTML = svg;
}

function legend(container, series) {
  container.innerHTML = series.map(s =>
    `<span><i style="background:${s.color}"></i>${s.name}</span>`).join('');
}

// ---------- 条形图 ----------
function bars(container, rows, opts={}) {
  const max = Math.max(...rows.map(r => r.value), 1e-9);
  container.innerHTML = rows.map(r => {
    const pct = (r.value/max*100).toFixed(1);
    const shown = opts.fmt ? opts.fmt(r.value) : r.value.toFixed(4);
    return `<div class="bar-row"><div class="name">${r.name}</div>
      <div class="bar-track"><div class="bar-fill ${r.cls||''}" style="width:${Math.max(pct,6)}%">${shown}</div></div></div>`;
  }).join('');
}

// ---------- 报告渲染 ----------
async function loadReport() {
  try {
    const res = await fetch('/api/report');
    if (!res.ok) return false;
    REPORT = await res.json();
  } catch (e) { return false; }
  renderDashboard(); renderCurves(); renderStrategies();
  return true;
}

function fmtBytes(b) {
  if (b > 1e9) return (b/1e9).toFixed(2)+' GB';
  if (b > 1e6) return (b/1e6).toFixed(1)+' MB';
  return (b/1e3).toFixed(1)+' KB';
}

function renderDashboard() {
  const r = REPORT, s = r.stats;
  $('#h-n').textContent = r.n.toLocaleString();
  $('#h-dim').textContent = r.dim;
  $('#h-build').textContent = (r.build_ms/1000).toFixed(1)+' s';
  $('#h-heap').textContent = fmtBytes(s.heap_alloc_bytes);
  $('#h-pq').textContent = s.pq_codes_bytes ? fmtBytes(s.pq_codes_bytes) : '未启用';

  const noFilterRecall = r.by_selectivity['无过滤']?.exact?.recall_at_k;
  const kpis = [
    { name:'无过滤召回率@'+r.k, val:noFilterRecall, fmt:v=>v==null?'—':v.toFixed(4) },
    { name:'索引构建时间', val:r.build_ms/1000, fmt:v=>v.toFixed(1)+' s' },
    { name:'估计总内存', val:s.total_est_bytes, fmt:fmtBytes },
  ];
  $('#kpi-row').innerHTML = kpis.map(k =>
    `<div class="card"><h3>${k.name}</h3><div class="metric">${k.fmt(k.val)}</div></div>`).join('');

  // 各档位召回 / 时延
  const labels = r.bands.map(b => b.label);
  const recallRows = labels.map(l => ({
    name:l, value:r.by_selectivity[l]?.exact?.recall_at_k ?? 0,
    cls:(r.by_selectivity[l]?.exact?.recall_at_k ?? 0) > 0.9 ? 'green' : ''
  }));
  bars($('#sel-recall'), recallRows);
  const latRows = labels.map(l => ({
    name:l, value:r.by_selectivity[l]?.exact?.latency_us?.p95 ?? 0,
    fmt:v=>v.toFixed(0)+' µs'
  }));
  bars($('#sel-lat'), latRows);

  // PQ 对比
  if (r.pq_compare) {
    bars($('#pq-compare'), Object.entries(r.pq_compare).map(([k,v]) => ({
      name:k, value:v, cls:v>0.9?'green':v>0.8?'warn':'red'
    })));
  }
  // 删除 sweep
  if (r.deletion_sweep?.length) {
    bars($('#del-sweep'), r.deletion_sweep.map(d => ({
      name:(d.deleted_frac*100).toFixed(0)+'% 已删除',
      value:d.recall_no_filter,
      cls:d.recall_no_filter>0.85?'green':'warn'
    })));
    const repairInfo = r.deletion_sweep.map(d =>
      `${(d.deleted_frac*100).toFixed(0)}%: 修复${(d.repair_ms/1000).toFixed(1)}s`).join(' · ');
    $('#del-sweep').insertAdjacentHTML('beforeend',
      `<p class="muted" style="font-size:12px;margin-top:10px">各批局部图修复耗时 — ${repairInfo}</p>`);
  }
}

function renderCurves() {
  const r = REPORT;
  const efSeries = [{
    name:'召回率', color:COLORS.accent,
    points:r.ef_curve.map((p,i) => ({x:i, label:'ef='+p.ef, y:p.recall}))
  }];
  lineChart($('#ef-chart'), efSeries, {yMin:0, yMax:1.02, yDigits:2});
  legend($('#ef-legend'), efSeries);

  if (r.deletion_sweep?.length) {
    const delSeries = [
      { name:'无过滤召回', color:COLORS.green,
        points:r.deletion_sweep.map((d,i)=>({x:i,label:(d.deleted_frac*100)+'%',y:d.recall_no_filter})) },
      { name:'过滤召回', color:COLORS.purple,
        points:r.deletion_sweep.map((d,i)=>({x:i,label:(d.deleted_frac*100)+'%',y:d.recall_filtered})) },
    ];
    lineChart($('#del-chart'), delSeries, {yMin:0, yMax:1.02, yDigits:2});
    legend($('#del-legend'), delSeries);
  }
}

function renderStrategies() {
  const tb = $('#strat-table tbody');
  const rows = [];
  const add = (label, sel, m) => {
    if (!m) return;
    const dist = Object.entries(m.strategy_dist).map(([k,v]) =>
      `<span class="tag ${k}">${k} ×${v}</span>`).join(' ');
    rows.push(`<tr><td>${label}</td><td>${sel?sel.toFixed(4):'1.0000'}</td><td style="text-align:left">${dist}</td>
      <td>${m.recall_at_k.toFixed(4)}</td><td>${m.latency_us.p50.toFixed(0)} µs</td>
      <td>${m.latency_us.p95.toFixed(0)} µs</td><td>${m.avg_visited.toFixed(0)}</td></tr>`);
  };
  const r = REPORT;
  add('无过滤', 1, r.by_selectivity['无过滤']?.exact);
  r.bands.forEach(b => add(b.label, b.cases[0].selectivity, r.by_selectivity[b.label]?.exact));
  tb.innerHTML = rows.join('');
}

// ---------- Tab 切换 ----------
document.querySelectorAll('.tabs button').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('.tabs button').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    btn.classList.add('active');
    $('#view-'+btn.dataset.view).classList.add('active');
  };
});

// ---------- 交互查询 ----------
async function initExplorer() {
  const fs = await (await fetch('/api/filters')).json();
  const sel = $('#q-filter');
  // 去重后填充
  const seen = new Set();
  fs.forEach(f => {
    const key = f.key+':'+f.value;
    if (seen.has(key)) return;
    seen.add(key);
    const o = document.createElement('option');
    o.value = JSON.stringify({[f.key]: f.value});
    o.textContent = `${f.label}=${f.value} (选择率 ${(f.selectivity*100).toFixed(2)}%, ${f.matches}条)`;
    sel.appendChild(o);
  });

  $('#q-run').onclick = runQuery;
}

async function runQuery() {
  const filterRaw = $('#q-filter').value;
  let filter = null;
  if (filterRaw) {
    const kv = JSON.parse(filterRaw);
    filter = {};
    for (const [k, v] of Object.entries(kv)) {
      if (k === 'tenant') filter.tenant = v;
      if (k === 'category') filter.category = v;
      if (k === 'region') filter.region = v;
      if (k === 'mintime') filter.min_time = v;
      if (k === 'label') filter.label = v;
      if (k === 'minprice') filter.min_price = v;
    }
  }
  const body = {
    query_id: parseInt($('#q-id').value) || 0,
    k: parseInt($('#q-k').value), ef: parseInt($('#q-ef').value),
    mode: $('#q-mode').value, track_path: $('#q-track').checked,
    filter,
  };
  const res = await (await fetch('/api/search', {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)
  })).json();

  const r = res.result;
  $('#q-meta').innerHTML =
    `策略 <span class="tag ${r.strategy}">${r.strategy}</span> ·
     估计选择率 <code>${(r.selectivity*100).toFixed(3)}%</code> ·
     访问节点 <b>${r.visited}</b> ·
     精排后距离为精确 L2²`;

  $('#q-hits tbody').innerHTML = res.hits.map(h =>
    `<tr><td>${h.id}</td><td>${h.dist.toFixed(4)}</td><td>${h.tenant}</td><td>${h.category}</td></tr>`).join('');
  renderPath(res.path, new Set(res.hits.map(h=>h.id)));
}

function renderPath(path, resultSet) {
  if (!path?.length) { $('#q-path').innerHTML = '<span class="muted">未开启路径追踪</span>'; return; }
  // 按层分组
  const byLayer = {};
  let entry = path[0]?.node;
  path.forEach(p => { (byLayer[p.layer] ??= []).push(p); });
  const layers = Object.keys(byLayer).map(Number).sort((a,b)=>b-a);
  // 每层最多渲染的节点数, 避免上万节点撑爆 DOM
  const MAX_PER_LAYER = 120;
  $('#q-path').innerHTML = layers.map(L => {
    const all = byLayer[L];
    const nodes = all.slice(0, MAX_PER_LAYER);
    const dots = nodes.map((p, i) => {
      const cls = ['node-dot'];
      if (p.filtered_out) cls.push('filtered');
      if (p.node === entry && L === layers[0]) cls.push('entry');
      if (resultSet.has(p.node)) cls.push('result');
      const title = `节点 ${p.node}, 距离 ${p.dist.toFixed(4)}${p.filtered_out?' (不满足过滤, 仅经过)':''}`;
      return `${i>0?'<span class="arrow">→</span>':''}
        <div class="${cls.join(' ')}" title="${title}">${p.node}</div>`;
    }).join('');
    const omitted = all.length - nodes.length;
    const more = omitted > 0
      ? `<span class="muted" style="font-size:12px">… 其余 ${omitted} 个节点已折叠(本层共 ${all.length} 个)</span>` : '';
    return `<div class="layer-row"><div class="layer-label">第 ${L} 层</div>${dots}${more}</div>`;
  }).join('') + `<p class="muted" style="font-size:12px">
    图例: <span class="pill" style="border-color:var(--warn)">顶层入口</span>
    <span class="pill" style="border-color:var(--accent)">经过节点</span>
    <span class="pill" style="border-color:var(--danger)">过滤不满足(仍经过)</span>
    <span class="pill" style="background:var(--accent2);color:#06121f">最终命中</span></p>`;
}

// ---------- 启动 ----------
(async () => {
  const ok = await loadReport();
  if (!ok) {
    document.querySelector('main').insertAdjacentHTML('afterbegin',
      '<div class="card" style="margin-bottom:16px;border-color:var(--warn)">暂无 bench 报告, ' +
      '总览/曲线页为空; 可先运行 <code>vecsearch bench</code>。交互查询仍可使用。</div>');
  }
  await initExplorer();
})();
