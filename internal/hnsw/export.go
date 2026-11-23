package hnsw

// 供 index 包做节点重连(更新)的导出包装, 复用图内部的
// searchLayer / selectHeuristic, 不重复实现算法。

// Cand 候选节点(导出形式)。
type Cand struct {
	ID int32
	D  float32
}

// SearchLayerExternal 单层贪心搜索的导出接口。
func (g *Graph) SearchLayerExternal(q int32, ep []Cand, ef, layer int,
	d Dist, allow func(int32) bool) []Cand {
	internal := make([]cand, len(ep))
	for i, e := range ep {
		internal[i] = cand{e.ID, e.D}
	}
	vs := g.NewVisited()
	defer g.ReleaseVisited(vs)
	out := g.searchLayer(q, internal, ef, layer, d, allow, nil, vs)
	res := make([]Cand, len(out))
	for i, c := range out {
		res[i] = Cand{c.id, c.d}
	}
	return res
}

// SelectHeuristicExternal 启发式选边的导出接口。
func (g *Graph) SelectHeuristicExternal(cs []Cand, maxM int,
	pairDist func(a, b int32) float32) []int32 {
	internal := make([]cand, len(cs))
	for i, c := range cs {
		internal[i] = cand{c.ID, c.D}
	}
	return g.selectHeuristic(internal, maxM, pairDist)
}
