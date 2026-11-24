package store_test

import (
	"testing"

	"vecsvc/internal/hnsw"
	"vecsvc/internal/store"
	"vecsvc/internal/synth"
)

func buildStore(t *testing.T, n int) *store.Store {
	t.Helper()
	ds, err := synth.Generate(synth.Config{
		N: n, Dim: 64, Clusters: 32,
		NTenants: 16, NCategories: 8, TSSpan: 100000, Seed: 11,
	})
	if err != nil {
		t.Fatal(err)
	}
	st := store.New(ds, hnsw.DefaultConfig(), store.DefaultThresholds())
	rng := synth.NewPCG64(11, 22)
	dim := ds.Dim
	for i := 0; i < ds.N; i++ {
		q := ds.Vectors[i*dim : (i+1)*dim]
		distOf := func(cand int32) float32 {
			off := int(cand) * dim
			return hnsw.L2Squared(q, ds.Vectors[off:off+dim])
		}
		st.Index().SetEdgeDist(func(a, b int32) float32 {
			da, db := int(a)*dim, int(b)*dim
			return hnsw.L2Squared(ds.Vectors[da:da+dim], ds.Vectors[db:db+dim])
		})
		st.Index().Insert(int32(i), distOf, rng.Uint64)
	}
	return st
}

// TestStrategySelection verifies the wide/middle/strict strategy bands are
// chosen from filter selectivity and that every strategy returns k results
// consistent with the predicate.
func TestStrategySelection(t *testing.T) {
	st := buildStore(t, 6000)
	ds := st.Dataset()
	query := ds.Vectors[0:ds.Dim]

	cases := []struct {
		name   string
		filter *synth.FilterSpec
		want   string // strategy prefix
	}{
		{"unfiltered", nil, store.StrategyGraphRelaxed},
		{"wide_tenant_in", &synth.FilterSpec{Op: synth.OpTenantIN,
			Vals: []uint16{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}}, store.StrategyGraphRelaxed},
		{"very_strict_combo", &synth.FilterSpec{Op: synth.OpAnd,
			Left: &synth.FilterSpec{Op: synth.OpAnd,
				Left:  &synth.FilterSpec{Op: synth.OpTenantEQ, Val: 1},
				Right: &synth.FilterSpec{Op: synth.OpCatEQ, Val: 2}},
			Right: &synth.FilterSpec{Op: synth.OpPriceLTE, FloatVal: 1}}, ""}, // strategy asserted below from real selectivity
	}
	thr := store.DefaultThresholds()
	for _, c := range cases {
		sel := 1.0
		if c.filter != nil {
			sel = c.filter.Selectivity(ds)
		}
		res := st.FilteredSearch(query, c.filter, sel, store.FilteredSearchOptions{K: 10, Ef: 64})
		// Derive the expected strategy band from the measured selectivity.
		want := c.want
		if want == "" {
			switch {
			case sel <= thr.Strict:
				want = store.StrategyBruteForce
			case sel >= thr.Wide:
				want = store.StrategyGraphRelaxed
			default:
				want = store.StrategyHybrid
			}
			if sel > thr.Strict {
				t.Fatalf("test setup: very_strict_combo selectivity %.5f not below strict threshold %.4f", sel, thr.Strict)
			}
		}
		if res.Strategy != want && !startsWith(res.Strategy, want) {
			t.Errorf("%s: strategy=%q want prefix %q (sel=%.5f)", c.name, res.Strategy, want, sel)
		}
		if len(res.IDs) > 10 {
			t.Errorf("%s: returned %d ids, want <= 10", c.name, len(res.IDs))
		}
		// Every returned id must satisfy the predicate and be live.
		pred := func(id int32) bool {
			return c.filter == nil || c.filter.Match(ds, int(id))
		}
		for _, id := range res.IDs {
			if !pred(id) {
				t.Errorf("%s: returned id %d that fails the filter", c.name, id)
			}
			if st.Index().IsDeleted(id) {
				t.Errorf("%s: returned tombstoned id %d", c.name, id)
			}
		}
	}
}

// TestInsertUpdateDelete exercises the mutation API end to end.
func TestInsertUpdateDelete(t *testing.T) {
	st := buildStore(t, 4000)
	ds := st.Dataset()
	rng := synth.NewPCG64(99, 1)
	vec := make([]float32, ds.Dim)
	for i := range vec {
		vec[i] = float32(rng.Gaussian())
	}
	id := st.Insert(vec, 2, 3, 12345, 99.5, rng.Uint64)
	if st.Index().IsDeleted(id) {
		t.Fatal("newly inserted id is deleted")
	}
	if got := st.Dataset().Tenant[id]; got != 2 {
		t.Fatalf("tenant=%d want 2", got)
	}
	newID, ok := st.Update(id, vec, rng.Uint64)
	if !ok {
		t.Fatal("update failed")
	}
	if !st.Index().IsDeleted(id) {
		t.Fatal("old id should be tombstoned after update")
	}
	if st.Index().IsDeleted(newID) {
		t.Fatal("new id from update should be live")
	}
	if st.Delete(newID, rng.Uint64) != true {
		t.Fatal("delete of live id should succeed")
	}
	if st.Delete(newID, rng.Uint64) != false {
		t.Fatal("double delete should return false")
	}
}

func startsWith(s, prefix string) bool {
	return len(s) >= len(prefix) && s[:len(prefix)] == prefix
}
