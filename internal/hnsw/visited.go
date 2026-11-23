package hnsw

import "sync"

// VisitedSet 单次搜索私有的已访问集合, 基于 generation stamp 数组,
// 比 map[int32]struct{} 快一个数量级且零哈希开销。
//
// 每个并行搜索持有自己的实例(通过 Pool 复用), 因此无共享状态、无需锁。
// stamp 单调递增, 跨层调用通过 Reset 复用同一块数组。
type VisitedSet struct {
	marks []uint32
	gen   uint32
	count int
}

// NewVisitedSet 创建容量为图节点上限的访问集合。
func NewVisitedSet(capacity int) *VisitedSet {
	return &VisitedSet{marks: make([]uint32, capacity)}
}

// visitedPool 按容量复用 VisitedSet, 消除并行搜索时的大数组重复分配。
var visitedPool = sync.Pool{
	New: func() any { return &VisitedSet{} },
}

// acquireVisited 取一个容量至少为 capacity 的 VisitedSet。
// 新分配的 marks 全为 0, 因此 gen 从 1 起即可保证不与初始值冲突。
func acquireVisited(capacity int) *VisitedSet {
	vs := visitedPool.Get().(*VisitedSet)
	if cap(vs.marks) < capacity {
		vs.marks = make([]uint32, capacity)
		vs.gen = 0
	}
	vs.marks = vs.marks[:capacity]
	vs.count = 0
	// 若复用的是全新 marks(全 0), 从 gen=1 开始; 否则延续旧 gen。
	// Reset 会再 +1, 这里只需保证本轮 gen 不与任何残留 marks 相等。
	if vs.gen == 0 || vs.gen == ^uint32(0) {
		vs.gen = 0
	}
	return vs
}

// release 归还 VisitedSet 供复用。
func (v *VisitedSet) release() { visitedPool.Put(v) }

// Reset 开始一次新的 searchLayer 调用, 逻辑上清空但不释放内存。
func (v *VisitedSet) Reset() { v.gen++; v.count = 0 }

// Count 返回本次 Reset 后访问的节点数。
func (v *VisitedSet) Count() int { return v.count }

// Seen 标记 id 已访问; 返回 true 表示此前已访问(调用方应跳过)。
func (v *VisitedSet) Seen(id int32) bool {
	if v.marks[id] == v.gen {
		return true
	}
	v.marks[id] = v.gen
	v.count++
	return false
}
