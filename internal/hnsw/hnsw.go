// Package hnsw 实现分层近邻图索引(Hierarchical Navigable Small World)。
//
// 结构与算法遵循 HNSW 论文(Malkov & Yashunin, 2018):
//   - 节点随机分层, 层数服从指数衰减; 高层稀疏、低层稠密
//   - 插入时从顶层入口贪心下降定位, 在每层用启发式选边保证
//     连通性与邻居多样性, 并做双向连接
//   - 查询从顶层入口逐层贪心下降, 底层在有界候选队列(ef)内
//     做贪心最佳优先搜索, ef 在召回率与时延间取舍
//
// 图本身不绑定距离实现, 通过 Dist 接口注入(精确 L2 / PQ 查表),
// 也不感知过滤语义, 由上层把谓词包成 allow 回调。
package hnsw

import (
	"math"
	"math/rand/v2"
	"sort"
	"sync"
)

// Dist 距离源: 返回查询 q 与节点 id 的平方 L2 距离。
type Dist interface {
	Distance(q int32, id int32) float32
}

// PathEntry 搜索路径上的一次访问(可视化与诊断用)。
type PathEntry struct {
	Node        int32   `json:"node"`
	Layer       int     `json:"layer"`
	Dist        float32 `json:"dist"`
	Expanded    bool    `json:"expanded"`
	FilteredOut bool    `json:"filtered_out"`
}

// Graph 分层近邻图。
type Graph struct {
	mu sync.RWMutex

	M              int // 每层目标出度(层0为 2M)
	MaxM           int // 层 >0 的最大度
	MaxM0          int // 层 0 的最大度
	EfConstruction int // 建图时候选队列大小
	simpleSelect   bool

	levels []int          // 每个节点的最高层
	links  [][][]int32    // links[id][layer] = 该节点在该层的邻居
	nodeMu []sync.RWMutex // 每节点锁, 保护其邻接表

	entryPoint int32
	maxLayer   int
	size       int32
	capacity   int

	rng   *rand.Rand
	rngMu sync.Mutex

	// per-insert 节点间距离缓存, 消除启发式选边/剪枝的重复距离计算。
	// 仅在单次 Insert 期间有效, Insert 结束清空。
	pairCache map[uint64]float32
	cacheOn   bool
}

// pairKey 缓存键: 有序节点对。
func pairKey(a, b int32) uint64 {
	if a > b {
		a, b = b, a
	}
	return uint64(uint32(a))<<32 | uint64(uint32(b))
}

// cachedPair 包装 pairDist, 按有序节点对缓存结果。
func (g *Graph) cachedPair(pairDist func(a, b int32) float32) func(a, b int32) float32 {
	if !g.cacheOn {
		return pairDist
	}
	return func(a, b int32) float32 {
		k := pairKey(a, b)
		if d, ok := g.pairCache[k]; ok {
			return d
		}
		d := pairDist(a, b)
		g.pairCache[k] = d
		return d
	}
}

// Config 图参数。
type Config struct {
	M              int
	EfConstruction int
	Seed           uint64
	// SimpleSelect: 用最近邻选边代替启发式(仅实验对比用)。
	SimpleSelect bool
}

func DefaultConfig() Config {
	return Config{M: 16, EfConstruction: 200, Seed: 1}
}

// New 创建空图, capacity 用于预分配。
func New(cfg Config, capacity int) *Graph {
	if cfg.M == 0 {
		cfg = DefaultConfig()
	}
	return &Graph{
		M: cfg.M, MaxM: cfg.M, MaxM0: 2 * cfg.M,
		EfConstruction: cfg.EfConstruction,
		simpleSelect:   cfg.SimpleSelect,
		levels:         make([]int, 0, capacity),
		links:          make([][][]int32, 0, capacity),
		nodeMu:         make([]sync.RWMutex, 0, capacity),
		entryPoint:     -1,
		rng:            rand.New(rand.NewPCG(cfg.Seed, cfg.Seed^0xdeadbeef)),
		capacity:       capacity,
	}
}

// NewVisited 为一次搜索创建私有的访问集合(底层经 Pool 复用)。
func (g *Graph) NewVisited() *VisitedSet {
	return acquireVisited(max(g.capacity, int(g.size)+1))
}

// ReleaseVisited 归还访问集合。
func (g *Graph) ReleaseVisited(vs *VisitedSet) { vs.release() }

// Size / EntryPoint / Level 基本访问器。
func (g *Graph) Size() int32 {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.size
}
func (g *Graph) EntryPoint() int32 {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.entryPoint
}
func (g *Graph) Level(id int32) int { return g.levels[id] }
func (g *Graph) MaxLayer() int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.maxLayer
}

// Neighbors 返回节点在某层的邻居副本(修复/评估用)。
func (g *Graph) Neighbors(id int32, layer int) []int32 {
	g.nodeMu[id].RLock()
	defer g.nodeMu[id].RUnlock()
	if layer >= len(g.links[id]) {
		return nil
	}
	out := make([]int32, len(g.links[id][layer]))
	copy(out, g.links[id][layer])
	return out
}

// SetLinks 用给定邻接表覆盖节点在某层的连边(局部修复用)。
func (g *Graph) SetLinks(id int32, layer int, nbrs []int32) {
	g.nodeMu[id].Lock()
	defer g.nodeMu[id].Unlock()
	// Repair 可能已清空外层切片, 按 level 重建骨架
	if layer >= len(g.links[id]) {
		links := make([][]int32, g.levels[id]+1)
		copy(links, g.links[id])
		g.links[id] = links
	}
	cp := make([]int32, len(nbrs))
	copy(cp, nbrs)
	g.links[id][layer] = cp
}

// ReinitNodeLinks 为被 Repair 清空的节点重建空的分层邻接骨架(更新场景)。
func (g *Graph) ReinitNodeLinks(id int32) {
	g.nodeMu[id].Lock()
	defer g.nodeMu[id].Unlock()
	if len(g.links[id]) < g.levels[id]+1 {
		g.links[id] = make([][]int32, g.levels[id]+1)
	}
}

// randomLevel 按 HNSW 几何分布采样: l = floor(-ln(uniform)/ln(M))。
func (g *Graph) randomLevel() int {
	g.rngMu.Lock()
	r := g.rng.Float64()
	g.rngMu.Unlock()
	l := int(-math.Log(math.Max(r, 1e-12)) / math.Log(float64(g.M)))
	if l < 0 {
		l = 0
	}
	return l
}

// ---- 堆: 候选最小堆 + 结果最大堆, 基于切片避免 interface 装箱 ----

type cand struct {
	id int32
	d  float32
}

type minCand []cand

func (h minCand) Len() int           { return len(h) }
func (h minCand) Less(i, j int) bool { return h[i].d < h[j].d }
func (h minCand) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minCand) Push(x cand)       { *h = append(*h, x); siftUp(h, len(*h)-1, true) }
func (h *minCand) Pop() cand {
	v := (*h)[0]
	n := len(*h) - 1
	(*h)[0] = (*h)[n]
	*h = (*h)[:n]
	if n > 0 {
		siftDown(h, 0, true)
	}
	return v
}

type maxCand []cand

func (h maxCand) Len() int           { return len(h) }
func (h maxCand) Less(i, j int) bool { return h[i].d > h[j].d }
func (h maxCand) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *maxCand) Push(x cand)       { *h = append(*h, x); siftUp(h, len(*h)-1, false) }
func (h *maxCand) Pop() cand {
	v := (*h)[0]
	n := len(*h) - 1
	(*h)[0] = (*h)[n]
	*h = (*h)[:n]
	if n > 0 {
		siftDown(h, 0, false)
	}
	return v
}
func (h *maxCand) Top() cand { return (*h)[0] }

func siftUp(h sort.Interface, i int, _ bool) {
	for i > 0 {
		p := (i - 1) / 2
		if h.Less(i, p) {
			h.Swap(i, p)
			i = p
		} else {
			return
		}
	}
}

func siftDown(h sort.Interface, i int, _ bool) {
	n := h.Len()
	for {
		l, r, best := 2*i+1, 2*i+2, i
		if l < n && h.Less(l, best) {
			best = l
		}
		if r < n && h.Less(r, best) {
			best = r
		}
		if best == i {
			return
		}
		h.Swap(i, best)
		i = best
	}
}

// searchLayer 在单层做贪心最佳优先搜索。
//
// allow 为 nil 表示不过滤; 否则不满足谓词的节点仍会被展开
// (搜索可以"经过"它们以保住图的连通性), 但不进入结果队列。
// path != nil 时记录完整访问轨迹。vs 为本次搜索私有的访问集合,
// 每次调用前由调用方 Reset, 使搜索可并行且无共享状态。
func (g *Graph) searchLayer(q int32, ep []cand, ef int, layer int,
	d Dist, allow func(int32) bool, path *[]PathEntry, vs *VisitedSet) []cand {

	vs.Reset()
	candidates := make(minCand, 0, ef*2)
	result := make(maxCand, 0, ef+1)

	for _, e := range ep {
		if vs.Seen(e.id) {
			continue
		}
		allowed := allow == nil || allow(e.id)
		if allowed {
			result.Push(e)
		}
		candidates.Push(e)
		if path != nil {
			*path = append(*path, PathEntry{Node: e.id, Layer: layer, Dist: e.d, FilteredOut: !allowed})
		}
	}

	for candidates.Len() > 0 {
		c := candidates.Pop()
		worst := float32(1e30)
		if result.Len() > 0 {
			worst = result.Top().d
		}
		// 结果队列已满且最近候选都比最差结果远: 收敛
		if c.d > worst && result.Len() >= ef {
			break
		}
		g.nodeMu[c.id].RLock()
		var nbrs []int32
		if layer < len(g.links[c.id]) {
			nbrs = g.links[c.id][layer]
		}
		g.nodeMu[c.id].RUnlock()
		for _, nb := range nbrs {
			if vs.Seen(nb) {
				continue
			}
			dd := d.Distance(q, nb)
			allowed := allow == nil || allow(nb)
			candidates.Push(cand{nb, dd})
			if allowed {
				result.Push(cand{nb, dd})
				if result.Len() > ef {
					result.Pop()
				}
			}
			if path != nil {
				*path = append(*path, PathEntry{Node: nb, Layer: layer, Dist: dd, Expanded: true, FilteredOut: !allowed})
			}
		}
	}

	out := make([]cand, result.Len())
	copy(out, result)
	sort.Slice(out, func(i, j int) bool { return out[i].d < out[j].d })
	return out
}

// selectHeuristic HNSW 启发式邻居选择。
//
// 输入按到 queryNode 的距离升序; 对每个候选 c, 若它到任一已选
// 邻居的距离比它到 queryNode 更近, 说明该方向已被覆盖, 跳过。
// 由此抑制团簇内冗余的短边, 保留方向多样的连接; 选不满时用
// 最近候选回填, 保证度与连通性。
func (g *Graph) selectHeuristic(candidates []cand, maxM int,
	pairDist func(a, b int32) float32) []int32 {

	// 实验对照: 纯最近邻选边
	if g.simpleSelect {
		n := min(maxM, len(candidates))
		out := make([]int32, n)
		for i := 0; i < n; i++ {
			out[i] = candidates[i].id
		}
		return out
	}

	work := make([]cand, len(candidates))
	copy(work, candidates)
	selected := make([]int32, 0, maxM)
	for len(work) > 0 && len(selected) < maxM {
		c := work[0]
		work = work[1:]
		good := true
		for _, s := range selected {
			if pairDist(c.id, s) < c.d {
				good = false
				break
			}
		}
		if good {
			selected = append(selected, c.id)
		}
	}
	if len(selected) < maxM {
		have := make(map[int32]struct{}, len(selected))
		for _, s := range selected {
			have[s] = struct{}{}
		}
		for _, c := range candidates {
			if len(selected) >= maxM {
				break
			}
			if _, ok := have[c.id]; !ok {
				selected = append(selected, c.id)
				have[c.id] = struct{}{}
			}
		}
	}
	return selected
}

// Insert 按序插入节点 q(内部 id 等于当前 size)。
// dist: q 到任意节点的距离; pairDist: 两个已存在节点间距离。
func (g *Graph) Insert(q int32, dist Dist, pairDist func(a, b int32) float32) int {
	level := g.randomLevel()

	g.mu.Lock()
	if int(q) >= len(g.levels) {
		// 预分配不足时扩容
		need := int(q) + 1
		g.levels = append(g.levels, make([]int, need-len(g.levels))...)
		g.links = append(g.links, make([][][]int32, need-len(g.links))...)
		g.nodeMu = append(g.nodeMu, make([]sync.RWMutex, need-len(g.nodeMu))...)
	}
	// 已预分配(定长切片)时不得截断, 否则后续批量节点越界
	if int(q+1) > len(g.levels) {
		g.levels = g.levels[:q+1]
		g.links = g.links[:q+1]
		g.nodeMu = g.nodeMu[:q+1]
	}
	g.levels[q] = level
	g.links[q] = make([][]int32, level+1)
	g.size = q + 1

	if g.entryPoint < 0 {
		g.entryPoint = q
		g.maxLayer = level
		g.mu.Unlock()
		return level
	}
	ep := g.entryPoint
	topLayer := g.maxLayer
	g.mu.Unlock()

	// 开启本次插入的节点间距离缓存
	g.pairCache = make(map[uint64]float32, 256)
	g.cacheOn = true
	pair := g.cachedPair(pairDist)
	defer func() {
		g.pairCache = nil
		g.cacheOn = false
	}()

	// 阶段 1: 顶层下降到 level+1, 每层只保留最近 1 个。
	// 每次 searchLayer 前 Reset, 等价于原 generation stamp 每次自增。
	vs := g.NewVisited()
	defer g.ReleaseVisited(vs)
	cur := []cand{{ep, dist.Distance(q, ep)}}
	for lc := topLayer; lc > level; lc-- {
		cur = g.searchLayer(q, cur, 1, lc, dist, nil, nil, vs)
	}

	// 阶段 2: 各目标层找邻居 + 双向连接 + 度超限剪枝
	startL := min(level, topLayer)
	for lc := startL; lc >= 0; lc-- {
		nearest := g.searchLayer(q, cur, g.EfConstruction, lc, dist, nil, nil, vs)
		maxM := g.MaxM
		if lc == 0 {
			maxM = g.MaxM0
		}
		chosen := g.selectHeuristic(nearest, maxM, pair)
		g.links[q][lc] = append(g.links[q][lc], chosen...)
		for _, nb := range chosen {
			g.nodeMu[nb].Lock()
			g.links[nb][lc] = append(g.links[nb][lc], q)
			conns := g.links[nb][lc]
			if len(conns) > maxM {
				cs := make([]cand, len(conns))
				for i, x := range conns {
					cs[i] = cand{x, pair(nb, x)}
				}
				sort.Slice(cs, func(i, j int) bool { return cs[i].d < cs[j].d })
				g.links[nb][lc] = g.selectHeuristic(cs, maxM, pair)
			}
			g.nodeMu[nb].Unlock()
		}
		cur = nearest
	}

	if level > topLayer {
		g.mu.Lock()
		g.entryPoint = q
		g.maxLayer = level
		g.mu.Unlock()
	}
	return level
}

// SearchResult 一次图搜索的输出。
type SearchResult struct {
	Nodes   []int32
	Dists   []float32
	Path    []PathEntry
	Visited int
}

// Search 从顶层入口逐层贪心下降, 底层用 ef 有界队列搜索。
// 上层下降忽略谓词(保证路由不被过滤切断), 底层启用谓词但
// 允许经过不满足的节点。
func (g *Graph) Search(q int32, k, ef int, d Dist, allow func(int32) bool, trackPath bool) SearchResult {
	g.mu.RLock()
	ep := g.entryPoint
	top := g.maxLayer
	g.mu.RUnlock()
	if ep < 0 {
		return SearchResult{}
	}
	var path []PathEntry
	if trackPath {
		path = make([]PathEntry, 0, 256)
	}
	cur := []cand{{ep, d.Distance(q, ep)}}
	vs := g.NewVisited()
	defer g.ReleaseVisited(vs)
	totalVisited := 1
	if trackPath {
		path = append(path, PathEntry{Node: ep, Layer: top, Dist: cur[0].d})
	}
	for lc := top; lc >= 1; lc-- {
		cur = g.searchLayer(q, cur, 1, lc, d, nil, &path, vs)
		totalVisited += vs.Count()
	}
	bottom := g.searchLayer(q, cur, max(ef, k), 0, d, allow, &path, vs)
	totalVisited += vs.Count()
	n := min(k, len(bottom))
	res := SearchResult{
		Nodes: make([]int32, n), Dists: make([]float32, n),
		Path: path, Visited: totalVisited,
	}
	for i := 0; i < n; i++ {
		res.Nodes[i] = bottom[i].id
		res.Dists[i] = bottom[i].d
	}
	return res
}
