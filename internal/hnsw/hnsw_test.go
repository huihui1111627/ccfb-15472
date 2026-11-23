package hnsw

import (
	"math"
	"testing"
)

// 测试用距离: 一维坐标上的欧氏距离, 手工可验证。
type lineDist struct{ coords []float32 }

func (d lineDist) Distance(q, id int32) float32 {
	diff := d.coords[q] - d.coords[id]
	return diff * diff
}

func buildLineGraph(t *testing.T, n int) *Graph {
	g := New(Config{M: 8, EfConstruction: 64, Seed: 1}, n)
	g.Preallocate(n)
	coords := make([]float32, n)
	for i := range coords {
		coords[i] = float32(i)
	}
	d := lineDist{coords}
	pair := func(a, b int32) float32 {
		diff := coords[a] - coords[b]
		return diff * diff
	}
	for i := int32(0); i < int32(n); i++ {
		g.Insert(i, d, pair)
	}
	return g
}

func TestGraphConnectivityAndDegree(t *testing.T) {
	n := 1000
	g := buildLineGraph(t, n)
	if g.Size() != int32(n) {
		t.Fatalf("size=%d want %d", g.Size(), n)
	}
	// 从入口 BFS, 层 0 必须连通所有节点
	seen := map[int32]bool{g.EntryPoint(): true}
	queue := []int32{g.EntryPoint()}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		for _, nb := range g.Neighbors(cur, 0) {
			if !seen[nb] {
				seen[nb] = true
				queue = append(queue, nb)
			}
		}
	}
	if len(seen) != n {
		t.Fatalf("层0不连通: 可达 %d/%d", len(seen), n)
	}
	// 度不超过上限
	for id := int32(0); id < int32(n); id++ {
		if got := len(g.Neighbors(id, 0)); got > 2*g.M {
			t.Fatalf("节点 %d 层0度 %d 超过 2M=%d", id, got, 2*g.M)
		}
	}
}

func TestSearchFindsNearestOnLine(t *testing.T) {
	n := 1000
	g := buildLineGraph(t, n)
	coords := make([]float32, n)
	for i := range coords {
		coords[i] = float32(i)
	}
	d := lineDist{coords}
	// 查询点放在 500.5 附近(用已有节点 500 作查询), 最近邻应包含 499-501
	res := g.Search(500, 5, 64, d, nil, false)
	if len(res.Nodes) != 5 {
		t.Fatalf("返回 %d 个结果, want 5", len(res.Nodes))
	}
	// 距离必须升序
	for i := 1; i < len(res.Dists); i++ {
		if res.Dists[i] < res.Dists[i-1] {
			t.Fatal("结果未按距离升序")
		}
	}
	// 节点 499/501 是 500 的最近邻, 必须命中
	hit := map[int32]bool{}
	for _, id := range res.Nodes {
		hit[id] = true
	}
	if !hit[499] || !hit[501] {
		t.Fatalf("未找到直线上的最近邻, got %v", res.Nodes)
	}
}

func TestRandomLevelDistribution(t *testing.T) {
	g := New(Config{M: 16, EfConstruction: 64, Seed: 1}, 100)
	const trials = 50000
	counts := map[int]int{}
	for i := 0; i < trials; i++ {
		counts[g.randomLevel()]++
	}
	// P(level=0) 理论 = 1 - 1/M ≈ 0.9375
	p0 := float64(counts[0]) / trials
	if math.Abs(p0-0.9375) > 0.01 {
		t.Fatalf("level0 概率 %.4f 偏离理论 0.9375", p0)
	}
}
