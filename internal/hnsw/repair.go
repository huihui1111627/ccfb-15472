package hnsw

import "sort"

// Repair 局部修复: 对一批已删除节点, 把它们从存活邻居的邻接表中
// 摘除, 并为每个受影响节点在其"本地二度邻域"内用启发式重新选边补连,
// 保持节点度与图的连通性。
//
// isDeleted 由上层提供(软删除位图)。pairDist 用于启发式选边。
// 补连候选只取每个节点自己的邻居与二度邻居, 复杂度与局部邻域成正比,
// 不随删除总量或图规模二次方增长, 不需要全图重建。
func (g *Graph) Repair(deleted []int32, isDeleted func(int32) bool,
	pairDist func(a, b int32) float32) int {

	g.mu.RLock()
	top := g.maxLayer
	g.mu.RUnlock()

	delSet := make(map[int32]struct{}, len(deleted))
	for _, d := range deleted {
		delSet[d] = struct{}{}
	}

	// affected: 所有与删除节点直接相连的存活节点
	affected := make(map[int32]struct{}, len(deleted)*g.M)
	for _, d := range deleted {
		// 二次修复时, 该节点可能已在上一轮被清空邻接表, 跳过
		if len(g.links[d]) == 0 {
			continue
		}
		for layer := 0; layer <= g.levels[d]; layer++ {
			g.nodeMu[d].RLock()
			var nbrs []int32
			if layer < len(g.links[d]) {
				nbrs = append([]int32(nil), g.links[d][layer]...)
			}
			g.nodeMu[d].RUnlock()
			for _, nb := range nbrs {
				if !isDeleted(nb) {
					affected[nb] = struct{}{}
				}
			}
		}
	}

	// 全图实际最高层, 用于高层孤岛的边界保护
	maxLevel := top
	for id := int32(0); id < g.size; id++ {
		if g.levels[id] > maxLevel {
			maxLevel = g.levels[id]
		}
	}

	// 阶段 1: 摘除受影响节点连向删除点的边, 记录每层摘除后保留的邻居
	keptNbrs := make(map[int32][][]int32)
	for a := range affected {
		layers := make([][]int32, g.levels[a]+1)
		for layer := 0; layer <= g.levels[a] && layer < len(g.links[a]); layer++ {
			g.nodeMu[a].Lock()
			old := g.links[a][layer]
			kept := make([]int32, 0, len(old))
			for _, nb := range old {
				if _, gone := delSet[nb]; gone || isDeleted(nb) {
					continue
				}
				kept = append(kept, nb)
			}
			g.links[a][layer] = kept
			g.nodeMu[a].Unlock()
			layers[layer] = append([]int32(nil), kept...)
		}
		keptNbrs[a] = layers
	}

	// 阶段 2: 对每个度不足的受影响节点, 用本地二度邻域补连。
	// 候选仅来自该节点保留邻居(一度)及其邻居的邻居(二度), 规模有界。
	for a := range affected {
		for layer := 0; layer <= g.levels[a] && layer <= maxLevel; layer++ {
			maxM := g.MaxM
			if layer == 0 {
				maxM = g.MaxM0
			}
			cur := keptNbrs[a][layer]
			if len(cur) >= g.M { // 度仍充足, 不必补
				continue
			}

			// 收集本地候选: 二度邻居, 排除已删除、自身与现有邻居
			candSet := make(map[int32]struct{}, len(cur)*(g.M+1))
			for _, nb := range cur {
				candSet[nb] = struct{}{}
				g.nodeMu[nb].RLock()
				if layer < len(g.links[nb]) {
					for _, nn := range g.links[nb][layer] {
						if nn != a && !isDeleted(nn) {
							candSet[nn] = struct{}{}
						}
					}
				}
				g.nodeMu[nb].RUnlock()
			}
			curHave := make(map[int32]struct{}, len(cur))
			for _, x := range cur {
				curHave[x] = struct{}{}
			}
			cs := make([]cand, 0, len(candSet))
			for cid := range candSet {
				if cid == a {
					continue
				}
				if _, ok := curHave[cid]; ok {
					continue
				}
				cs = append(cs, cand{cid, pairDist(a, cid)})
			}
			sort.Slice(cs, func(i, j int) bool { return cs[i].d < cs[j].d })
			newLinks := g.selectHeuristic(cs, maxM-len(cur), pairDist)
			if len(newLinks) == 0 {
				continue
			}
			// 双向补连
			g.nodeMu[a].Lock()
			if layer < len(g.links[a]) {
				g.links[a][layer] = append(g.links[a][layer], newLinks...)
			}
			g.nodeMu[a].Unlock()
			for _, nb := range newLinks {
				g.nodeMu[nb].Lock()
				if layer < len(g.links[nb]) {
					g.links[nb][layer] = append(g.links[nb][layer], a)
					conns := g.links[nb][layer]
					if len(conns) > maxM {
						cc := make([]cand, len(conns))
						for i, x := range conns {
							cc[i] = cand{x, pairDist(nb, x)}
						}
						sort.Slice(cc, func(i, j int) bool { return cc[i].d < cc[j].d })
						g.links[nb][layer] = g.selectHeuristic(cc, maxM, pairDist)
					}
				}
				g.nodeMu[nb].Unlock()
			}
		}
	}

	// 清空已删除节点的邻接表并在必要时更换入口
	for _, d := range deleted {
		g.nodeMu[d].Lock()
		g.links[d] = nil
		g.nodeMu[d].Unlock()
	}
	g.mu.Lock()
	if isDeleted(g.entryPoint) {
		g.entryPoint = -1
		for id := int32(0); id < g.size; id++ {
			if !isDeleted(id) {
				g.entryPoint = id
				g.maxLayer = g.levels[id]
				break
			}
		}
	}
	g.mu.Unlock()
	return len(affected)
}

// Stats 返回图的统计信息: 总边数、各层节点数、平均度。
func (g *Graph) Stats(isDeleted func(int32) bool) map[string]any {
	g.mu.RLock()
	defer g.mu.RUnlock()
	// 数组按全图实际最高 level 分配, 兼容历史高层节点
	hi := g.maxLayer
	for id := int32(0); id < g.size; id++ {
		if g.levels[id] > hi {
			hi = g.levels[id]
		}
	}
	layerNodes := make([]int, hi+1)
	edges := make([]int64, hi+1)
	degreeSum := make([]int64, hi+1)
	alive := int32(0)
	for id := int32(0); id < g.size; id++ {
		if isDeleted != nil && isDeleted(id) {
			continue
		}
		alive++
		for l := 0; l <= g.levels[id] && l < len(g.links[id]) && l <= hi; l++ {
			layerNodes[l]++
			edges[l] += int64(len(g.links[id][l]))
			degreeSum[l] += int64(len(g.links[id][l]))
		}
	}
	avgDeg := make([]float64, hi+1)
	for l := range avgDeg {
		if layerNodes[l] > 0 {
			avgDeg[l] = float64(degreeSum[l]) / float64(layerNodes[l])
		}
	}
	return map[string]any{
		"nodes": alive, "max_layer": g.maxLayer,
		"layer_nodes": layerNodes, "edges_per_layer": edges,
		"avg_degree_per_layer": avgDeg,
	}
}
