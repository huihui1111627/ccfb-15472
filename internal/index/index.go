// Package index 把图索引、存储、乘积量化组装成完整检索服务:
//
//   - 构建: 顺序插入 HNSW, 可选训练 PQ
//   - 查询: 按过滤选择率在三种策略间选择
//     宽 -> 图搜索中跳过不满足点但继续经过它们走
//     中 -> 图搜索 + 放宽 ef + 命中不足时在候选子集补搜
//     严 -> 在满足条件的子集上暴力(可用 PQ 加速)
//   - PQ: 图/暴力内部用压缩距离粗排, 再用原始向量精排
//   - 删除/更新: 软删除位图, 累积到阈值触发局部图修复
package index

import (
	"fmt"
	"sort"
	"sync"
	"time"

	"vecsearch/internal/hnsw"
	"vecsearch/internal/pq"
	"vecsearch/internal/store"
)

// Mode 距离模式。
type Mode int

const (
	ModeExact    Mode = iota // 全程原始向量精确距离
	ModeHybridPQ             // 图导航用精确距离, 候选粗排用 PQ, 再原始向量精排(工程默认)
	ModePQ                   // 全程 PQ 距离导航 + 原始向量精排(极限省内存场景)
)

// Strategy 过滤检索策略。
type Strategy string

const (
	StratGraphFiltered Strategy = "graph_filtered" // 宽: 图内过滤
	StratGraphRelaxed  Strategy = "graph_relaxed"  // 中: 放宽 ef + 子集补搜
	StratBruteForce    Strategy = "brute_force"    // 严: 子集暴力
	StratNone          Strategy = "none"           // 无过滤
)

const (
	defaultRelaxedSel = 0.05  // 低于 5% 进入中间策略
	defaultBruteSel   = 0.005 // 低于 0.5% 直接暴力
)

// Options 构建参数。
type Options struct {
	HNSW hnsw.Config
	PQ   *pq.TrainConfig // nil 表示不启用压缩
	// 策略阈值
	RelaxedSelectivity float64
	BruteSelectivity   float64
	// 删除比例达到该值触发图修复
	RepairThreshold float64
}

func DefaultOptions() Options {
	return Options{
		HNSW:               hnsw.DefaultConfig(),
		RelaxedSelectivity: defaultRelaxedSel,
		BruteSelectivity:   defaultBruteSel,
		RepairThreshold:    0.1,
	}
}

// Index 完整检索索引。
type Index struct {
	opts    Options
	st      *store.Store
	g       *hnsw.Graph
	quant   *pq.PQ
	vectors []float32 // N*Dim 原始向量(行主序)
	n, dim  int

	// 外部 id(文档 id) -> 内部 id; 更新复用内部 id
	extID map[string]int32
	idSeq int32

	buildTime time.Duration

	repairCount int
	lastRepair  time.Duration
	repaired    map[int32]struct{} // 已做过图修复的删除节点

	mu sync.Mutex // 保护删除/更新/修复
}

// ---- 距离源 ----

type exactDist struct{ idx *Index }

func (e exactDist) Distance(q, id int32) float32 {
	return l2sq(e.idx.vectors[int(q)*e.idx.dim:(int(q)+1)*e.idx.dim], e.idx.vectors[int(id)*e.idx.dim:(int(id)+1)*e.idx.dim])
}

type queryExactDist struct {
	idx *Index
	q   []float32
}

func (d queryExactDist) Distance(_ int32, id int32) float32 {
	return l2sq(d.q, d.idx.vectors[int(id)*d.idx.dim:(int(id)+1)*d.idx.dim])
}

type queryPQDist struct {
	idx *Index
	lut []float32
}

func (d queryPQDist) Distance(_ int32, id int32) float32 {
	return d.idx.quant.LUTDist(id, d.lut)
}

type distPair struct {
	id int32
	d  float32
}

func sortPairs(ps []distPair) {
	sort.Slice(ps, func(i, j int) bool { return ps[i].d < ps[j].d })
}

// Build 用已填充的 store 与原始向量构建索引。
func Build(st *store.Store, vectors []float32, dim int, opts Options) (*Index, time.Duration, error) {
	n := st.N
	idx := &Index{
		opts: opts, st: st, vectors: vectors, n: n, dim: dim,
		g:     hnsw.New(opts.HNSW, n),
		extID: make(map[string]int32, n),
	}
	for i := 0; i < n; i++ {
		idx.extID[fmt.Sprintf("doc-%d", i)] = int32(i)
	}
	idx.idSeq = int32(n)
	idx.g.Preallocate(n)

	t0 := time.Now()
	ed := exactDist{idx: idx}
	pair := func(a, b int32) float32 {
		return l2sqOff(vectors, dim, int(a)*dim, int(b)*dim)
	}
	for i := int32(0); i < int32(n); i++ {
		idx.g.Insert(i, ed, pair)
		if (int(i)+1)%200_000 == 0 {
			fmt.Printf("  hnsw: %d/%d 节点已插入 (%v)\n", i+1, n, time.Since(t0).Round(time.Millisecond))
		}
	}
	idx.buildTime = time.Since(t0)

	if opts.PQ != nil {
		tp0 := time.Now()
		idx.quant = pq.Train(vectors, n, dim, *opts.PQ)
		fmt.Printf("  pq: 训练+编码完成 (%v)\n", time.Since(tp0).Round(time.Millisecond))
	}
	return idx, idx.buildTime, nil
}

// Store / Graph / Quant 暴露给评估与服务层。
func (idx *Index) Store() *store.Store      { return idx.st }
func (idx *Index) Graph() *hnsw.Graph       { return idx.g }
func (idx *Index) Quant() *pq.PQ            { return idx.quant }
func (idx *Index) Dim() int                 { return idx.dim }
func (idx *Index) N() int                   { return idx.n }
func (idx *Index) BuildTime() time.Duration { return idx.buildTime }
func (idx *Index) Vector(id int32) []float32 {
	return idx.vectors[int(id)*idx.dim : (int(id)+1)*idx.dim]
}

// SearchOptions 单次查询参数。
type SearchOptions struct {
	K         int
	Ef        int
	Mode      Mode
	TrackPath bool
	// RerankMul: PQ 粗排保留 K*RerankMul 个候选再精排
	RerankMul int
}

// SearchResult 对外查询结果。
type SearchResult struct {
	IDs         []int32                  `json:"ids"`
	Dists       []float32                `json:"dist_exact"`
	Strategy    Strategy                 `json:"strategy"`
	Selectivity float64                  `json:"selectivity"`
	Visited     int                      `json:"visited"`
	Path        []hnsw.PathEntry         `json:"path,omitempty"`
	Mode        Mode                     `json:"mode"`
	Timings     map[string]time.Duration `json:"-"`
}

// decideStrategy 根据选择率决定策略。
func (idx *Index) decideStrategy(sel float64, hasFilter bool) Strategy {
	if !hasFilter {
		return StratNone
	}
	if sel <= idx.opts.BruteSelectivity {
		return StratBruteForce
	}
	if sel <= idx.opts.RelaxedSelectivity {
		return StratGraphRelaxed
	}
	return StratGraphFiltered
}

// Search 主查询入口。
func (idx *Index) Search(q []float32, f *store.Filter, so SearchOptions) SearchResult {
	if so.K == 0 {
		so.K = 10
	}
	if so.Ef == 0 {
		so.Ef = 64
	}
	if so.RerankMul == 0 {
		so.RerankMul = 8
	}
	hasFilter := f != nil
	sel := 1.0
	if hasFilter {
		sel = idx.st.Selectivity(f)
	}
	strat := idx.decideStrategy(sel, hasFilter)

	timings := map[string]time.Duration{}
	var coarse []distPair
	var path []hnsw.PathEntry
	visited := 0

	switch strat {
	case StratBruteForce:
		t := time.Now()
		coarse, visited = idx.bruteCandidates(q, f, so)
		timings["candidates"] = time.Since(t)
	default: // none / graph_filtered / graph_relaxed
		t := time.Now()
		ef := so.Ef
		if strat == StratGraphRelaxed {
			// 中间选择率: 放大 ef 对抗过滤造成的命中稀释
			ef = min(so.Ef*4, idx.n)
		}
		var dd hnsw.Dist
		if so.Mode == ModePQ && idx.quant != nil {
			// 全程 PQ 导航(极限压缩场景); HybridPQ 仍用精确距离导航
			dd = queryPQDist{idx: idx, lut: idx.quant.BuildLUT(q)}
		} else {
			dd = queryExactDist{idx: idx, q: q}
		}
		allow := func(id int32) bool { return !idx.st.Deleted(id) && idx.st.Match(id, f) }
		res := idx.g.Search(-1, max(ef, so.K), ef, dd, allow, so.TrackPath)
		coarse = make([]distPair, len(res.Nodes))
		for i := range res.Nodes {
			coarse[i] = distPair{res.Nodes[i], res.Dists[i]}
		}
		path, visited = res.Path, res.Visited

		// 中间策略: 图命中不足 K 时, 用倒排枚举子集补搜
		if strat == StratGraphRelaxed && len(coarse) < so.K {
			coarse = idx.supplement(q, f, coarse, so)
		}
		timings["graph"] = time.Since(t)
		// HybridPQ: 图用精确距离取回 ef 个候选后, 用 PQ 距离重排截断,
		// 再在保留的 K*RerankMul 个上用原始向量精排 —— 模拟真实流水线
		if so.Mode == ModeHybridPQ && idx.quant != nil {
			tp := time.Now()
			lut := idx.quant.BuildLUT(q)
			for i := range coarse {
				coarse[i].d = idx.quant.LUTDist(coarse[i].id, lut)
			}
			sortPairs(coarse)
			keep := min(len(coarse), max(so.K*so.RerankMul, so.K))
			coarse = coarse[:keep]
			timings["pq_rerank"] = time.Since(tp)
		}
	}

	// 精排
	t := time.Now()
	usePQ := (so.Mode == ModePQ || so.Mode == ModeHybridPQ) && idx.quant != nil
	if so.Mode == ModePQ && idx.quant != nil {
		// 全程 PQ: 粗排不保证严格有序, 保留 K*RerankMul 个候选, 用原始向量精排
		keep := min(len(coarse), max(so.K*so.RerankMul, so.K))
		for i := 0; i < keep; i++ {
			coarse[i].d = l2sq(q, idx.Vector(coarse[i].id))
		}
		coarse = coarse[:keep]
	}
	if so.Mode == ModeHybridPQ && idx.quant != nil {
		// PQ 截断后的候选统一用原始向量精排
		for i := range coarse {
			coarse[i].d = l2sq(q, idx.Vector(coarse[i].id))
		}
	}
	// 输出前统一按精确距离排序(补搜/精排后有序化)
	if strat == StratGraphRelaxed || usePQ {
		sortPairs(coarse)
	}
	timings["rerank"] = time.Since(t)

	k := min(so.K, len(coarse))
	out := SearchResult{
		IDs:         make([]int32, k),
		Dists:       make([]float32, k),
		Strategy:    strat,
		Selectivity: sel,
		Visited:     visited,
		Path:        path,
		Mode:        so.Mode,
		Timings:     timings,
	}
	for i := 0; i < k; i++ {
		out.IDs[i] = coarse[i].id
		out.Dists[i] = coarse[i].d
	}
	return out
}
