package index

import (
	"testing"

	"vecsearch/internal/hnsw"
	"vecsearch/internal/sim"
	"vecsearch/internal/store"
)

func buildSmokeIndex(t *testing.T, n int, pqOn bool) *Index {
	t.Helper()
	cfg := sim.DefaultConfig(n)
	cfg.Seed = 1
	ds := sim.Generate(cfg)
	st := store.New(n, cfg.Dim, cfg.Tenants, cfg.Categories, cfg.Regions, cfg.NumLabels)
	st.Tenant, st.Category = ds.Tenant, ds.Category
	st.Timestamp, st.Region = ds.Timestamp, ds.Region
	st.Price, st.Labels = ds.Price, ds.Labels
	st.BuildPostings()
	opts := DefaultOptions()
	opts.HNSW = hnsw.Config{M: 16, EfConstruction: 128, Seed: 2}
	if pqOn {
		pqc := sim.PQConfigForTest(cfg.Dim)
		opts.PQ = &pqc
	}
	idx, _, err := Build(st, ds.Vectors, cfg.Dim, opts)
	if err != nil {
		t.Fatal(err)
	}
	return idx
}

func recallAt(got, truth []int32) float64 {
	want := len(truth)
	if want == 0 {
		return 1
	}
	hit := 0
	have := map[int32]struct{}{}
	for _, id := range got {
		have[id] = struct{}{}
	}
	for _, id := range truth {
		if _, ok := have[id]; ok {
			hit++
		}
	}
	return float64(hit) / float64(want)
}

func TestNoFilterRecall(t *testing.T) {
	idx := buildSmokeIndex(t, 5000, false)
	qs := idx.testQueries(50, 99)
	var sum float64
	for _, q := range qs {
		gt := idx.BruteForceGroundTruth(q, nil, 10)
		res := idx.Search(q, nil, SearchOptions{K: 10, Ef: 64, Mode: ModeExact})
		sum += recallAt(res.IDs, gt)
	}
	r := sum / float64(len(qs))
	t.Logf("无过滤 recall@10 = %.4f", r)
	if r < 0.85 {
		t.Fatalf("recall too low: %.4f", r)
	}
}

func TestFilteredStrategies(t *testing.T) {
	idx := buildSmokeIndex(t, 8000, false)
	qs := idx.testQueries(30, 123)
	st := idx.Store()

	cases := []struct {
		name   string
		filter func(i int) *store.Filter
	}{
		{"宽: 单区域(8%)", func(i int) *store.Filter {
			r := st.Region[i%st.N]
			return &store.Filter{Region: &r}
		}},
		{"中: 单类目", func(i int) *store.Filter {
			c := st.Category[i%st.N]
			return &store.Filter{Category: &c}
		}},
		{"中: 类目+区域", func(i int) *store.Filter {
			c := st.Category[i%st.N]
			r := st.Region[i%st.N]
			return &store.Filter{Category: &c, Region: &r}
		}},
		{"严: 租户+类目+区域", func(i int) *store.Filter {
			tn := st.Tenant[i%st.N]
			c := st.Category[i%st.N]
			r := st.Region[i%st.N]
			return &store.Filter{Tenant: &tn, Category: &c, Region: &r}
		}},
	}
	for _, tc := range cases {
		var sum float64
		strats := map[Strategy]int{}
		for i, q := range qs {
			f := tc.filter(i)
			gt := idx.BruteForceGroundTruth(q, f, 10)
			res := idx.Search(q, f, SearchOptions{K: 10, Ef: 64, Mode: ModeExact})
			strats[res.Strategy]++
			sum += recallAt(res.IDs, gt)
		}
		t.Logf("%s: recall@10=%.4f 策略分布=%v", tc.name, sum/float64(len(qs)), strats)
		if sum/float64(len(qs)) < 0.85 {
			t.Fatalf("%s recall too low", tc.name)
		}
	}
}

func TestPQRecall(t *testing.T) {
	idx := buildSmokeIndex(t, 8000, true)
	qs := idx.testQueries(40, 7)
	var exactR, hybridR, fullPqR float64
	for _, q := range qs {
		gt := idx.BruteForceGroundTruth(q, nil, 10)
		exactR += recallAt(idx.Search(q, nil, SearchOptions{K: 10, Ef: 64, Mode: ModeExact}).IDs, gt)
		hybridR += recallAt(idx.Search(q, nil, SearchOptions{K: 10, Ef: 128, Mode: ModeHybridPQ, RerankMul: 8}).IDs, gt)
		fullPqR += recallAt(idx.Search(q, nil, SearchOptions{K: 10, Ef: 128, Mode: ModePQ, RerankMul: 8}).IDs, gt)
	}
	t.Logf("exact recall=%.4f  HybridPQ recall=%.4f  全程PQ recall=%.4f",
		exactR/40, hybridR/40, fullPqR/40)
	if hybridR/40 < 0.85 {
		t.Fatal("HybridPQ recall too low")
	}
}

func TestDeleteAndRepair(t *testing.T) {
	idx := buildSmokeIndex(t, 6000, false)
	qs := idx.testQueries(30, 55)
	before := 0.0
	for _, q := range qs {
		before += recallAt(idx.Search(q, nil, SearchOptions{K: 10, Ef: 64}).IDs,
			idx.BruteForceGroundTruth(q, nil, 10))
	}
	before /= float64(len(qs))

	// 删掉 15% 并强制修复
	victims := make([]int32, 0, idx.N()/6)
	for i := 0; i < idx.N()/6; i++ {
		victims = append(victims, int32(i*6))
	}
	idx.DeleteBatch(victims)

	after := 0.0
	for _, q := range qs {
		got := idx.Search(q, nil, SearchOptions{K: 10, Ef: 64}).IDs
		// ground truth 也需排除已删除点
		gt := idx.BruteForceGroundTruth(q, nil, 10)
		after += recallAt(got, gt)
	}
	after /= float64(len(qs))
	t.Logf("删除前 recall=%.4f, 删除15%%并修复后 recall=%.4f", before, after)
	if after < before-0.08 {
		t.Fatalf("repair failed to preserve recall: before=%.4f after=%.4f", before, after)
	}
}
