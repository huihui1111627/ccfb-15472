package index

import (
	"runtime"
)

// Stats 索引运行时统计。
type Stats struct {
	N                int     `json:"n"`
	Dim              int     `json:"dim"`
	Alive            int     `json:"alive"`
	Deleted          int     `json:"deleted"`
	DeletedRatio     float64 `json:"deleted_ratio"`
	RepairCount      int     `json:"repair_count"`
	LastRepairMillis int64   `json:"last_repair_ms"`
	Graph            any     `json:"graph"`

	// 内存估算(字节)
	RawVectorBytes  int64 `json:"raw_vector_bytes"`
	PQCodesBytes    int64 `json:"pq_codes_bytes"`
	PQCentroidBytes int64 `json:"pq_centroid_bytes"`
	GraphBytes      int64 `json:"graph_bytes"`
	AttributeBytes  int64 `json:"attribute_bytes"`
	TotalEstBytes   int64 `json:"total_est_bytes"`

	// Go 运行时实测
	HeapAllocBytes uint64 `json:"heap_alloc_bytes"`
	HeapSysBytes   uint64 `json:"heap_sys_bytes"`

	BuildTimeMillis int64 `json:"build_time_ms"`
}

// Stats 采集统计与内存占用。
func (idx *Index) Stats() Stats {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)

	st := idx.st
	s := Stats{
		N: idx.n, Dim: idx.dim,
		Alive: st.AliveCount(), Deleted: st.DeletedCount(),
		DeletedRatio:     float64(st.DeletedCount()) / float64(idx.n),
		RepairCount:      idx.repairCount,
		LastRepairMillis: idx.lastRepair.Milliseconds(),
		Graph:            idx.g.Stats(st.Deleted),
		BuildTimeMillis:  idx.buildTime.Milliseconds(),
	}

	// 原始向量
	s.RawVectorBytes = int64(idx.n) * int64(idx.dim) * 4
	// 属性列: tenant/cat(2) + ts(4) + region(1) + price(4) + 删除位图(1)
	s.AttributeBytes = int64(idx.n) * (2 + 2 + 4 + 1 + 4 + 1)
	// 标签切片额外开销按均值 2.5 个/条估算
	s.AttributeBytes += int64(float64(idx.n) * 2.5 * 2)

	// 图: 每条边 int32 + 各层切片头, 由图统计给出边总数
	if gs, ok := s.Graph.(map[string]any); ok {
		if edges, ok := gs["edges_per_layer"].([]int64); ok {
			var totalEdges int64
			for _, e := range edges {
				totalEdges += e
			}
			s.GraphBytes = totalEdges * 4
		}
	}
	// 每节点 levels(int) + links 外层切片
	s.GraphBytes += int64(idx.n) * (8 + 24)

	if idx.quant != nil {
		s.PQCodesBytes = int64(idx.quant.Bytes())
		s.PQCentroidBytes = int64(len(idx.quant.Centroids)) * 4
	}
	s.TotalEstBytes = s.RawVectorBytes + s.PQCodesBytes + s.PQCentroidBytes +
		s.GraphBytes + s.AttributeBytes

	s.HeapAllocBytes = ms.HeapAlloc
	s.HeapSysBytes = ms.HeapSys
	return s
}
