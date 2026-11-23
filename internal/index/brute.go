package index

import (
	"vecsearch/internal/store"
)

// bruteCandidates 在满足过滤条件的子集上暴力检索。
// PQ 模式用距离表做近似距离, exact 模式用原始向量。
// 返回按距离升序的候选(最多 K 个, 中间补搜场景由调用方放宽)。
func (idx *Index) bruteCandidates(q []float32, f *store.Filter, so SearchOptions) ([]distPair, int) {
	limit := so.K
	// 严策略下暴力天然精确, 直接取 K; PQ 模式多取供精排
	if (so.Mode == ModePQ || so.Mode == ModeHybridPQ) && idx.quant != nil {
		limit = max(so.K*so.RerankMul, so.K)
	}
	buf := make([]int32, 0, 1024)
	cands := idx.st.Enumerate(f, buf)

	var lut []float32
	usePQ := (so.Mode == ModePQ || so.Mode == ModeHybridPQ) && idx.quant != nil
	if usePQ {
		lut = idx.quant.BuildLUT(q)
	}

	// 有界最大堆, 避免对全部候选排序
	heap := make([]distPair, 0, limit+1)
	for _, id := range cands {
		var d float32
		if usePQ {
			d = idx.quant.LUTDist(id, lut)
		} else {
			d = l2sq(q, idx.Vector(id))
		}
		if len(heap) < limit {
			heap = append(heap, distPair{id, d})
			if len(heap) == limit {
				heapInitMax(heap)
			}
			continue
		}
		if d < heap[0].d {
			heap[0] = distPair{id, d}
			heapDown(heap, 0)
		}
	}
	sortPairs(heap)
	return heap, len(cands)
}

// supplement 中间策略的补搜: 图结果不足 K 时, 枚举过滤子集,
// 对图未覆盖的候选直接计算精确距离并合并。
func (idx *Index) supplement(q []float32, f *store.Filter, have []distPair, so SearchOptions) []distPair {
	seen := make(map[int32]struct{}, len(have)*2)
	for _, p := range have {
		seen[p.id] = struct{}{}
	}
	buf := make([]int32, 0, 1024)
	for _, id := range idx.st.Enumerate(f, buf) {
		if _, ok := seen[id]; ok {
			continue
		}
		have = append(have, distPair{id, l2sq(q, idx.Vector(id))})
	}
	sortPairs(have)
	if len(have) > so.Ef {
		have = have[:so.Ef]
	}
	return have
}

// BruteForceGroundTruth 暴力计算真实 top-K(始终用原始精确向量,
// 忽略图/PQ, 含删除判断), 作为召回评估的基准。
func (idx *Index) BruteForceGroundTruth(q []float32, f *store.Filter, k int) []int32 {
	so := SearchOptions{K: k, Ef: k, Mode: ModeExact}
	ps, _ := idx.bruteCandidates(q, f, so)
	out := make([]int32, min(k, len(ps)))
	for i := range out {
		out[i] = ps[i].id
	}
	return out
}

// ---- 有界最大堆(小切片手工实现) ----

func heapInitMax(h []distPair) {
	for i := len(h)/2 - 1; i >= 0; i-- {
		heapDown(h, i)
	}
}

func heapDown(h []distPair, i int) {
	n := len(h)
	for {
		l, r, best := 2*i+1, 2*i+2, i
		if l < n && h[l].d > h[best].d {
			best = l
		}
		if r < n && h[r].d > h[best].d {
			best = r
		}
		if best == i {
			return
		}
		h[i], h[best] = h[best], h[i]
		i = best
	}
}

// EstimateEnumerate 仅估计过滤子集大小(统计/策略展示用)。
func (idx *Index) EstimateEnumerate(f *store.Filter) int {
	buf := make([]int32, 0, 256)
	return len(idx.st.Enumerate(f, buf))
}
