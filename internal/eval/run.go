package eval

import (
	"fmt"
	"sort"
	"time"

	"vecsearch/internal/hnsw"
	"vecsearch/internal/index"
	"vecsearch/internal/store"
)

// Config 评估参数。
type Config struct {
	Queries  int // 查询数(每档)
	K        int
	Ef       int
	Seed     uint64
	EfSweep  []int
	DelFracs []float64 // 删除 sweep 的比例点
	PathEf   int
}

func DefaultEvalConfig() Config {
	return Config{
		Queries: 100, K: 10, Ef: 128, Seed: 2024,
		EfSweep:  []int{16, 32, 64, 128, 256, 512},
		DelFracs: []float64{0.05, 0.10, 0.20, 0.35},
		PathEf:   128,
	}
}

func recall(got, truth []int32) float64 {
	if len(truth) == 0 {
		return 1
	}
	have := make(map[int32]struct{}, len(got))
	for _, id := range got {
		have[id] = struct{}{}
	}
	hit := 0
	for _, id := range truth {
		if _, ok := have[id]; ok {
			hit++
		}
	}
	return float64(hit) / float64(len(truth))
}

func latencyStats(lat []time.Duration) LatencyStats {
	if len(lat) == 0 {
		return LatencyStats{}
	}
	us := make([]float64, len(lat))
	var sum float64
	for i, d := range lat {
		us[i] = float64(d.Microseconds())
		sum += us[i]
	}
	sort.Float64s(us)
	pct := func(p float64) float64 { return us[min(int(p*float64(len(us))), len(us)-1)] }
	return LatencyStats{
		Mean: sum / float64(len(us)),
		P50:  pct(0.50), P95: pct(0.95), P99: pct(0.99), Max: us[len(us)-1],
	}
}

// RunBand 在一个选择率档位上评估三种模式。
func RunBand(idx *index.Index, qs [][]float32, b Band, mode index.Mode, ec Config) *Metrics {
	m := &Metrics{StrategyDist: map[string]int{}}
	lats := make([]time.Duration, 0, ec.Queries)
	var recSum, visSum float64
	for qi := 0; qi < ec.Queries; qi++ {
		fc := pickFilter(&b, qi)
		q := qs[qi]
		// ground truth: 原始向量暴力
		gt := idx.BruteForceGroundTruth(q, fc.Filter, ec.K)
		t0 := time.Now()
		res := idx.Search(q, fc.Filter, index.SearchOptions{
			K: ec.K, Ef: ec.Ef, Mode: mode, RerankMul: 8,
		})
		lats = append(lats, time.Since(t0))
		recSum += recall(res.IDs, gt)
		visSum += float64(res.Visited)
		m.StrategyDist[string(res.Strategy)]++
	}
	m.N = ec.Queries
	m.RecallAtK = recSum / float64(ec.Queries)
	m.Visited = visSum / float64(ec.Queries)
	m.LatencyUs = latencyStats(lats)
	return m
}

// RunNoFilter 无过滤基线。
func RunNoFilter(idx *index.Index, qs [][]float32, mode index.Mode, ec Config) *Metrics {
	m := &Metrics{StrategyDist: map[string]int{string(index.StratNone): ec.Queries}}
	lats := make([]time.Duration, 0, ec.Queries)
	var recSum, visSum float64
	for qi := 0; qi < ec.Queries; qi++ {
		q := qs[qi]
		gt := idx.BruteForceGroundTruth(q, nil, ec.K)
		t0 := time.Now()
		res := idx.Search(q, nil, index.SearchOptions{K: ec.K, Ef: ec.Ef, Mode: mode})
		lats = append(lats, time.Since(t0))
		recSum += recall(res.IDs, gt)
		visSum += float64(res.Visited)
	}
	m.N = ec.Queries
	m.RecallAtK = recSum / float64(ec.Queries)
	m.Visited = visSum / float64(ec.Queries)
	m.LatencyUs = latencyStats(lats)
	return m
}

// EfCurve 无过滤下扫 ef, 画召回-时延权衡曲线。
func EfCurve(idx *index.Index, qs [][]float32, ec Config) []EfPoint {
	out := make([]EfPoint, 0, len(ec.EfSweep))
	for _, ef := range ec.EfSweep {
		var recSum, visSum float64
		lats := make([]time.Duration, 0, ec.Queries)
		for qi := 0; qi < ec.Queries; qi++ {
			q := qs[qi]
			gt := idx.BruteForceGroundTruth(q, nil, ec.K)
			t0 := time.Now()
			res := idx.Search(q, nil, index.SearchOptions{K: ec.K, Ef: ef, Mode: index.ModeExact})
			lats = append(lats, time.Since(t0))
			recSum += recall(res.IDs, gt)
			visSum += float64(res.Visited)
		}
		ls := latencyStats(lats)
		out = append(out, EfPoint{
			Ef: ef, Recall: recSum / float64(ec.Queries),
			LatUs: ls.Mean, P95Us: ls.P95, Visited: visSum / float64(ec.Queries),
		})
	}
	return out
}

// PQCompare 三种距离模式的召回对比(无过滤 + 固定 ef)。
func PQCompare(idx *index.Index, qs [][]float32, ec Config) map[string]float64 {
	modes := map[string]index.Mode{
		"exact":     index.ModeExact,
		"hybrid_pq": index.ModeHybridPQ,
		"full_pq":   index.ModePQ,
	}
	out := map[string]float64{}
	for name, mode := range modes {
		var recSum float64
		for qi := 0; qi < ec.Queries; qi++ {
			q := qs[qi]
			gt := idx.BruteForceGroundTruth(q, nil, ec.K)
			res := idx.Search(q, nil, index.SearchOptions{
				K: ec.K, Ef: ec.Ef * 2, Mode: mode, RerankMul: 8,
			})
			recSum += recall(res.IDs, gt)
		}
		out[name] = recSum / float64(ec.Queries)
	}
	return out
}

// DeletionSweep 逐步删除并修复, 观察召回变化。
// 每个比例点从上一个状态继续删除(累计)。idx 会被修改, 评估后不可复用。
func DeletionSweep(idx *index.Index, qs [][]float32, bands []Band, ec Config, rngSeed uint64) []DeletionPoint {
	n := idx.N()
	out := make([]DeletionPoint, 0, len(ec.DelFracs))
	var band *Band
	for i := range bands {
		if len(bands[i].Cases) > 0 {
			band = &bands[i]
			break
		}
	}

	already := 0
	for _, frac := range ec.DelFracs {
		target := int(float64(n) * frac)
		// 均匀抽取尚未删除的 id
		victims := pickVictims(idx, target-already, rngSeed+uint64(frac*1000))
		t0 := time.Now()
		idx.DeleteBatch(victims)
		repairMs := time.Since(t0).Milliseconds()
		already = target

		var recNoF, recF float64
		for qi := 0; qi < ec.Queries; qi++ {
			q := qs[qi]
			gt := idx.BruteForceGroundTruth(q, nil, ec.K)
			res := idx.Search(q, nil, index.SearchOptions{K: ec.K, Ef: ec.Ef, Mode: index.ModeExact})
			recNoF += recall(res.IDs, gt)
			if band != nil {
				fc := pickFilter(band, qi)
				gtf := idx.BruteForceGroundTruth(q, fc.Filter, ec.K)
				rf := idx.Search(q, fc.Filter, index.SearchOptions{K: ec.K, Ef: ec.Ef, Mode: index.ModeExact})
				recF += recall(rf.IDs, gtf)
			}
		}
		out = append(out, DeletionPoint{
			DeletedFrac:    frac,
			RecallNoFilter: recNoF / float64(ec.Queries),
			RecallFiltered: recF / float64(ec.Queries),
			RepairMs:       repairMs,
			Alive:          idx.Store().AliveCount(),
		})
		fmt.Printf("  删除 %.0f%%: 无过滤召回=%.4f 过滤召回=%.4f 修复=%dms\n",
			frac*100, recNoF/float64(ec.Queries), recF/float64(ec.Queries), repairMs)
	}
	return out
}

// pickVictims 选择 n 个当前存活的 id(确定性伪随机)。
func pickVictims(idx *index.Index, n int, seed uint64) []int32 {
	st := idx.Store()
	victims := make([]int32, 0, n)
	step := 997 // 大步长
	cand := int(seed % uint64(st.N))
	attempts, maxAttempts := 0, st.N*4
	for len(victims) < n && attempts < maxAttempts {
		attempts++
		id := int32(cand)
		if !st.Deleted(id) {
			victims = append(victims, id)
		}
		cand = (cand + step) % st.N
	}
	return victims
}

// CapturePath 跑一次带路径追踪的查询, 供页面展示搜索路径。
func CapturePath(idx *index.Index, q []float32, f *store.Filter, ec Config) *PathSample {
	res := idx.Search(q, f, index.SearchOptions{
		K: ec.K, Ef: ec.PathEf, Mode: index.ModeExact, TrackPath: true,
	})
	nodes := make([]PathNode, len(res.Path))
	for i, p := range res.Path {
		nodes[i] = PathNode{Node: p.Node, Layer: p.Layer, Dist: float64(p.Dist), FilteredOut: p.FilteredOut}
	}
	fname := "none"
	if f != nil {
		fname = "filtered"
	}
	return &PathSample{
		Filter: fname, Selectivity: res.Selectivity,
		Strategy: res.Strategy, Path: nodes, ResultIDs: res.IDs,
	}
}

// 确保 hnsw 导入在扩展路径功能时可用(当前仅占位)。
var _ = hnsw.PathEntry{}
