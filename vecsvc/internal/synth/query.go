package synth

// Op enumerates the supported filter predicates.
type Op int

const (
	OpTenantEQ Op = iota // one tenant
	OpTenantIN           // multiple tenants (union)
	OpCatEQ
	OpCatIN
	OpTSGTE
	OpPriceLTE
	OpAnd // conjunction of two predicates
)

// FilterSpec is a serializable filter definition. Filtering itself lives in
// the store package; this type only describes what was generated so that
// benchmarks and the web UI share the same vocabulary.
type FilterSpec struct {
	Op       Op
	Val      uint16 // equality / IN single operand
	Vals     []uint16
	NumVal   int64   // threshold for TS >=
	FloatVal float32 // threshold for price <=
	Left     *FilterSpec
	Right    *FilterSpec
}

// Match applies the predicate to row i of the dataset (used by brute-force
// ground truth and by the strict/hybrid posting-list strategies).
func (f *FilterSpec) Match(ds *Dataset, i int) bool {
	if f == nil {
		return true
	}
	switch f.Op {
	case OpTenantEQ:
		return ds.Tenant[i] == f.Val
	case OpTenantIN:
		for _, v := range f.Vals {
			if ds.Tenant[i] == v {
				return true
			}
		}
		return false
	case OpCatEQ:
		return ds.Category[i] == f.Val
	case OpCatIN:
		for _, v := range f.Vals {
			if ds.Category[i] == v {
				return true
			}
		}
		return false
	case OpTSGTE:
		return ds.TS[i] >= f.NumVal
	case OpPriceLTE:
		return ds.Price[i] <= f.FloatVal
	case OpAnd:
		return f.Left.Match(ds, i) && f.Right.Match(ds, i)
	default:
		return true
	}
}

// Selectivity evaluates the exact fraction of dataset rows matching f.
func (f *FilterSpec) Selectivity(ds *Dataset) float64 {
	if f == nil {
		return 1
	}
	c := 0
	for i := 0; i < ds.N; i++ {
		if f.Match(ds, i) {
			c++
		}
	}
	return float64(c) / float64(ds.N)
}

// Query is a benchmark query: a vector plus an optional filter.
type Query struct {
	Vector    []float32
	Filter    *FilterSpec
	TargetSel float64 // intended selectivity when the query was generated
	CaseName  string
}

// QuerySet is a named collection of queries sharing the same filter shape.
type QuerySet struct {
	Name      string
	Target    float64
	Filter    *FilterSpec // shared filter; nil for unfiltered
	Queries   []*Query
	ActualSel float64
}

// GenerateQuerySets builds the fixed ladder of filter selectivities used in
// the acceptance benchmark: from unfiltered (1.0) down to very strict
// (~0.0005), covering wide / middle / strict strategy regions.
func GenerateQuerySets(ds *Dataset, perCase int, seed uint64) ([]*QuerySet, error) {
	rng := NewPCG64(seed, 0xabcddcba)
	mkVec := func() []float32 {
		// Jitter an existing vector: guarantees a non-trivial local
		// neighborhood and mimics a real embedding near a stored point.
		base := rng.Intn(ds.N)
		v := make([]float32, ds.Dim)
		for j := range v {
			v[j] = ds.Vectors[base*ds.Dim+j] + float32(rng.Gaussian()*0.15)
		}
		return v
	}

	cases := []struct {
		name string
		sel  float64
		spec *FilterSpec
	}{
		{"unfiltered", 1.0, nil},
		{"tenant_in_8_wide", 0.25, &FilterSpec{Op: OpTenantIN, Vals: pickLabels(rng, ds.NTenants, 8)}},
		{"ts_recent_wide", 0.15, &FilterSpec{Op: OpTSGTE, NumVal: int64(float64(ds.TSSpan) * 0.85)}},
		{"tenant_eq_mid", 1.0 / float64(ds.NTenants), &FilterSpec{Op: OpTenantEQ, Val: uint16(rng.Intn(ds.NTenants))}},
		{"cat_and_ts_mid", 0.02, &FilterSpec{
			Op:    OpAnd,
			Left:  &FilterSpec{Op: OpCatEQ, Val: uint16(rng.Intn(ds.NCategories))},
			Right: &FilterSpec{Op: OpTSGTE, NumVal: int64(float64(ds.TSSpan) * 0.70)},
		}},
		{"tenant_cat_strict", 0.002, &FilterSpec{
			Op:    OpAnd,
			Left:  &FilterSpec{Op: OpTenantEQ, Val: uint16(rng.Intn(ds.NTenants))},
			Right: &FilterSpec{Op: OpCatEQ, Val: uint16(rng.Intn(ds.NCategories))},
		}},
		{"very_strict", 0.0005, &FilterSpec{
			Op:    OpAnd,
			Left:  &FilterSpec{Op: OpTenantEQ, Val: uint16(rng.Intn(ds.NTenants))},
			Right: &FilterSpec{Op: OpPriceLTE, FloatVal: 5}, // ~0.5% of price range
		}},
	}

	sets := make([]*QuerySet, 0, len(cases))
	for _, c := range cases {
		qs := &QuerySet{Name: c.name, Target: c.sel, Filter: c.spec}
		if c.spec != nil {
			qs.ActualSel = c.spec.Selectivity(ds)
		} else {
			qs.ActualSel = 1
		}
		for i := 0; i < perCase; i++ {
			qs.Queries = append(qs.Queries, &Query{
				Vector:    mkVec(),
				Filter:    c.spec,
				TargetSel: c.sel,
				CaseName:  c.name,
			})
		}
		sets = append(sets, qs)
	}
	return sets, nil
}

func pickLabels(rng *PCG64, max, k int) []uint16 {
	if k > max {
		k = max
	}
	seen := make(map[uint16]bool, k)
	out := make([]uint16, 0, k)
	for len(out) < k {
		v := uint16(rng.Intn(max))
		if !seen[v] {
			seen[v] = true
			out = append(out, v)
		}
	}
	return out
}
