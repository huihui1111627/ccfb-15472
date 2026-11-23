package index

import (
	"time"

	"vecsearch/internal/hnsw"
	"vecsearch/internal/store"
)

// Delete 按内部 id 软删除。当累计删除比例超过阈值时触发局部图修复。
// 返回是否触发了修复以及修复影响的节点数。
func (idx *Index) Delete(id int32) (repaired bool, affected int) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	if idx.st.Deleted(id) {
		return false, 0
	}
	idx.st.SetDeleted(id, true)
	ratio := float64(idx.st.DeletedCount()) / float64(idx.n)
	if ratio < idx.opts.RepairThreshold {
		return false, 0
	}
	affected = idx.repairLocked()
	return true, affected
}

// DeleteBatch 批量软删除, 完成后统一修复一次(删除 sweep 评估用)。
func (idx *Index) DeleteBatch(ids []int32) int {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	for _, id := range ids {
		idx.st.SetDeleted(id, true)
	}
	return idx.repairLocked()
}

// repairLocked 只对"新增"的删除节点做局部修复。已修复过的删除节点
// 其连边早已摘除, 重复处理只会让耗时随累计删除量线性恶化。
// 删除位图保留 true: 节点仍然不可见。
func (idx *Index) repairLocked() int {
	t0 := time.Now()
	if idx.repaired == nil {
		idx.repaired = make(map[int32]struct{})
	}
	fresh := make([]int32, 0, 64)
	for i := 0; i < idx.n; i++ {
		id := int32(i)
		if idx.st.Deleted(id) {
			if _, done := idx.repaired[id]; !done {
				fresh = append(fresh, id)
				idx.repaired[id] = struct{}{}
			}
		}
	}
	if len(fresh) == 0 {
		return 0
	}
	pair := func(a, b int32) float32 {
		return l2sqOff(idx.vectors, idx.dim, int(a)*idx.dim, int(b)*idx.dim)
	}
	affected := idx.g.Repair(fresh, idx.st.Deleted, pair)
	idx.lastRepair = time.Since(t0)
	idx.repairCount++
	return affected
}

// Update 更新一条记录: 内部 id 复用, 原地替换向量与属性,
// 然后摘除旧边并按新向量重新接入图(局部重连, 不重建全图)。
func (idx *Index) Update(id int32, newVec []float32, attrs store.Attrs) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	copy(idx.vectors[int(id)*idx.dim:], newVec)
	idx.st.Tenant[id] = attrs.Tenant
	idx.st.Category[id] = attrs.Category
	idx.st.Timestamp[id] = attrs.Timestamp
	idx.st.Region[id] = attrs.Region
	idx.st.Price[id] = attrs.Price
	idx.st.Labels[id] = attrs.Labels
	if idx.quant != nil {
		idx.quant.Encode(newVec, idx.quant.Codes[int(id)*idx.quant.M:])
	}

	pair := func(a, b int32) float32 {
		return l2sqOff(idx.vectors, idx.dim, int(a)*idx.dim, int(b)*idx.dim)
	}
	// 临时置删除位让 Repair 摘除旧边, 再恢复可见性重新接入
	idx.st.SetDeleted(id, true)
	idx.g.Repair([]int32{id}, idx.st.Deleted, pair)
	idx.st.SetDeleted(id, false)
	idx.g.ReinitNodeLinks(id) // Repair 清空了 links, 先重建分层骨架
	idx.rewireNode(id, pair)
}

// rewireNode 更新后的节点在其原有各层重新搜索邻居并接入。
func (idx *Index) rewireNode(id int32, pair func(a, b int32) float32) {
	ed := exactDist{idx: idx}
	level := idx.g.Level(id)
	ep := idx.g.EntryPoint()
	if ep < 0 {
		return
	}
	cur := []hnsw.Cand{{ID: ep, D: ed.Distance(id, ep)}}
	for lc := idx.g.MaxLayer(); lc > level; lc-- {
		cur = idx.g.SearchLayerExternal(id, cur, 1, lc, ed, nil)
	}
	for lc := min(level, idx.g.MaxLayer()); lc >= 0; lc-- {
		nearest := idx.g.SearchLayerExternal(id, cur, idx.opts.HNSW.EfConstruction, lc, ed, nil)
		maxM := idx.opts.HNSW.M
		if lc == 0 {
			maxM = 2 * idx.opts.HNSW.M
		}
		chosen := idx.g.SelectHeuristicExternal(nearest, maxM, pair)
		idx.g.SetLinks(id, lc, chosen)
		for _, nb := range chosen {
			nbrs := idx.g.Neighbors(nb, lc)
			nbrs = append(nbrs, id)
			if len(nbrs) > maxM {
				cc := make([]hnsw.Cand, len(nbrs))
				for i, x := range nbrs {
					cc[i] = hnsw.Cand{ID: x, D: pair(nb, x)}
				}
				nbrs = idx.g.SelectHeuristicExternal(cc, maxM, pair)
			}
			idx.g.SetLinks(nb, lc, nbrs)
		}
		cur = nearest
	}
}
