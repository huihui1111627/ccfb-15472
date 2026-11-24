package hnsw_test

import (
	"testing"

	"vecsvc/internal/hnsw"
	"vecsvc/internal/synth"
)

// TestGraphRecall builds a small clustered dataset and checks that graph
// search recovers most of the brute-force top-10 without a filter.
func TestGraphRecall(t *testing.T) {
	ds, err := synth.Generate(synth.Config{
		N: 5000, Dim: 32, Clusters: 16,
		NTenants: 8, NCategories: 8, TSSpan: 100000, Seed: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	idx := hnsw.New(hnsw.DefaultConfig(), ds.N)
	rng := synth.NewPCG64(1, 2)
	dim := ds.Dim
	for i := 0; i < ds.N; i++ {
		q := ds.Vectors[i*dim : (i+1)*dim]
		distOf := func(cand int32) float32 {
			off := int(cand) * dim
			return hnsw.L2Squared(q, ds.Vectors[off:off+dim])
		}
		idx.SetEdgeDist(func(a, b int32) float32 {
			da, db := int(a)*dim, int(b)*dim
			return hnsw.L2Squared(ds.Vectors[da:da+dim], ds.Vectors[db:db+dim])
		})
		idx.Insert(int32(i), distOf, rng.Uint64)
	}

	st := idx.Stats()
	if st.LiveNodes != ds.N {
		t.Fatalf("live nodes = %d, want %d", st.LiveNodes, ds.N)
	}
	if st.MeanDegree0 < 5 {
		t.Fatalf("mean layer-0 degree = %.2f, graph looks disconnected", st.MeanDegree0)
	}

	// Brute-force ground truth for 30 queries.
	const k = 10
	const nq = 30
	totalHit, total := 0, 0
	for qi := 0; qi < nq; qi++ {
		base := qi * 137 % ds.N
		query := ds.Vectors[base*dim : (base+1)*dim]
		src := &vecSource{ds: ds, q: query}

		// Ground truth.
		gt := bruteTopK(ds, query, k)

		res := idx.Search(src, hnsw.SearchOptions{K: k, Ef: 64})
		if len(res.IDs) != k {
			t.Fatalf("query %d returned %d ids, want %d", qi, len(res.IDs), k)
		}
		got := map[int32]bool{}
		for _, id := range res.IDs {
			got[id] = true
		}
		for _, id := range gt {
			total++
			if got[id] {
				totalHit++
			}
		}
	}
	recall := float64(totalHit) / float64(total)
	if recall < 0.85 {
		t.Fatalf("unfiltered recall@10 = %.3f, want >= 0.85", recall)
	}
	t.Logf("recall@10=%.3f meanDegree0=%.1f topLayer=%d", recall, st.MeanDegree0, st.TopLayer)
}

// TestDeleteAndRepair checks tombstones hide nodes and a rebuild restores recall.
func TestDeleteAndRepair(t *testing.T) {
	ds, _ := synth.Generate(synth.Config{
		N: 4000, Dim: 32, Clusters: 16,
		NTenants: 8, NCategories: 8, TSSpan: 100000, Seed: 3,
	})
	idx := hnsw.New(hnsw.DefaultConfig(), ds.N)
	rng := synth.NewPCG64(3, 4)
	dim := ds.Dim
	for i := 0; i < ds.N; i++ {
		q := ds.Vectors[i*dim : (i+1)*dim]
		distOf := func(cand int32) float32 {
			off := int(cand) * dim
			return hnsw.L2Squared(q, ds.Vectors[off:off+dim])
		}
		idx.SetEdgeDist(func(a, b int32) float32 {
			da, db := int(a)*dim, int(b)*dim
			return hnsw.L2Squared(ds.Vectors[da:da+dim], ds.Vectors[db:db+dim])
		})
		idx.Insert(int32(i), distOf, rng.Uint64)
	}

	// Delete 25% of the nodes.
	for i := 0; i < ds.N/4; i++ {
		if !idx.Delete(int32(i * 4)) {
			t.Fatalf("delete %d failed", i*4)
		}
	}
	if frac := idx.DeletedFraction(); frac < 0.24 || frac > 0.26 {
		t.Fatalf("deleted fraction = %.3f", frac)
	}

	// Search must never return a tombstoned id.
	query := ds.Vectors[0:dim]
	res := idx.Search(&vecSource{ds: ds, q: query}, hnsw.SearchOptions{K: 10, Ef: 64})
	for _, id := range res.IDs {
		if idx.IsDeleted(id) {
			t.Fatalf("search returned tombstoned node %d", id)
		}
	}

	// Force a repair.
	distFactory := func(node int32) func(int32) float32 {
		va := ds.Vectors[int(node)*dim : int(node)*dim+dim]
		return func(cand int32) float32 {
			off := int(cand) * dim
			return hnsw.L2Squared(va, ds.Vectors[off:off+dim])
		}
	}
	idx.SetEdgeDist(func(a, b int32) float32 {
		da, db := int(a)*dim, int(b)*dim
		return hnsw.L2Squared(ds.Vectors[da:da+dim], ds.Vectors[db:db+dim])
	})
	if !idx.RepairIfNeeded(0.10, distFactory, rng.Uint64) {
		t.Fatal("expected repair to run")
	}
	if idx.RepairRuns() != 1 || idx.DeletedCount() != 0 {
		t.Fatalf("repair state wrong: runs=%d deleted=%d", idx.RepairRuns(), idx.DeletedCount())
	}
	res = idx.Search(&vecSource{ds: ds, q: query}, hnsw.SearchOptions{K: 10, Ef: 64})
	if len(res.IDs) != 10 {
		t.Fatalf("post-repair search returned %d ids", len(res.IDs))
	}
	for _, id := range res.IDs {
		if id%4 == 0 && int(id) < ds.N {
			t.Fatalf("post-repair result contains deleted id %d", id)
		}
	}
}

type vecSource struct {
	ds *synth.Dataset
	q  []float32
}

func (v *vecSource) Dist(id int32) float32 {
	off := int(id) * v.ds.Dim
	return hnsw.L2Squared(v.q, v.ds.Vectors[off:off+v.ds.Dim])
}

func bruteTopK(ds *synth.Dataset, q []float32, k int) []int32 {
	type pair struct {
		id int32
		d  float32
	}
	all := make([]pair, ds.N)
	for i := 0; i < ds.N; i++ {
		off := i * ds.Dim
		all[i] = pair{int32(i), hnsw.L2Squared(q, ds.Vectors[off:off+ds.Dim])}
	}
	// Partial selection: simple sort is fine at test scale.
	for i := 0; i < k; i++ {
		best := i
		for j := i + 1; j < len(all); j++ {
			if all[j].d < all[best].d {
				best = j
			}
		}
		all[i], all[best] = all[best], all[i]
	}
	out := make([]int32, k)
	for i := 0; i < k; i++ {
		out[i] = all[i].id
	}
	return out
}
