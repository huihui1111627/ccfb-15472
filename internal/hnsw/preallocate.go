package hnsw

import "sync"

// Preallocate 一次性分配容纳 n 个节点的全部底层切片,
// 避免插入过程中反复扩容(大批量构建前调用)。切片长度直接置为 n,
// 未插入的位置由 size/entryPoint 保证不可见。
func (g *Graph) Preallocate(n int) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if len(g.levels) >= n {
		return
	}
	oldLevels, oldLinks, oldMu := g.levels, g.links, g.nodeMu
	g.levels = make([]int, n)
	copy(g.levels, oldLevels)
	g.links = make([][][]int32, n)
	copy(g.links, oldLinks)
	g.nodeMu = make([]sync.RWMutex, n)
	copy(g.nodeMu, oldMu)
	g.capacity = n
}
