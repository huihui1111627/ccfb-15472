package store

import (
	"sort"

	"vecsvc/internal/hnsw"
	"vecsvc/internal/synth"
)

// predicate converts a synthetic FilterSpec into a graph-search predicate.
func (s *Store) predicate(f *synth.FilterSpec) hnsw.Predicate {
	if f == nil {
		return nil
	}
	ds := s.ds
	switch f.Op {
	case synth.OpTenantEQ:
		return func(id int32) bool { return ds.Tenant[id] == f.Val }
	case synth.OpTenantIN:
		set := make(map[uint16]struct{}, len(f.Vals))
		for _, v := range f.Vals {
			set[v] = struct{}{}
		}
		return func(id int32) bool {
			_, ok := set[ds.Tenant[id]]
			return ok
		}
	case synth.OpCatEQ:
		return func(id int32) bool { return ds.Category[id] == f.Val }
	case synth.OpCatIN:
		set := make(map[uint16]struct{}, len(f.Vals))
		for _, v := range f.Vals {
			set[v] = struct{}{}
		}
		return func(id int32) bool {
			_, ok := set[ds.Category[id]]
			return ok
		}
	case synth.OpTSGTE:
		thr := f.NumVal
		return func(id int32) bool { return ds.TS[id] >= thr }
	case synth.OpPriceLTE:
		thr := f.FloatVal
		return func(id int32) bool { return ds.Price[id] <= thr }
	case synth.OpAnd:
		l := s.predicate(f.Left)
		r := s.predicate(f.Right)
		return func(id int32) bool { return l(id) && r(id) }
	default:
		return nil
	}
}

// candidateSet returns the posting list of matching ids when the filter can
// be served from inverted lists plus cheap residual checks; ok=false means
// the filter is a pure numeric scan (no label component).
func (s *Store) candidateSet(f *synth.FilterSpec) (ids []int32, ok bool) {
	if f == nil {
		return nil, false
	}
	switch f.Op {
	case synth.OpTenantEQ:
		out := append([]int32(nil), s.tenants[f.Val]...)
		return out, true
	case synth.OpTenantIN:
		var out []int32
		for _, v := range f.Vals {
			out = append(out, s.tenants[v]...)
		}
		sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
		return out, true
	case synth.OpCatEQ:
		out := append([]int32(nil), s.cats[f.Val]...)
		return out, true
	case synth.OpCatIN:
		var out []int32
		for _, v := range f.Vals {
			out = append(out, s.cats[v]...)
		}
		sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
		return out, true
	case synth.OpAnd:
		// Use the smaller side's posting list as the seed, residual-predicate
		// the other side; fall back to full scan if neither side has a list.
		lids, lok := s.candidateSet(f.Left)
		rids, rok := s.candidateSet(f.Right)
		rpred := s.predicate(f.Right)
		lpred := s.predicate(f.Left)
		switch {
		case lok && (!rok || len(lids) <= len(rids)):
			out := lids[:0:0]
			for _, id := range lids {
				if rpred(id) {
					out = append(out, id)
				}
			}
			return out, true
		case rok:
			out := rids[:0:0]
			for _, id := range rids {
				if lpred(id) {
					out = append(out, id)
				}
			}
			return out, true
		default:
			return nil, false
		}
	default:
		// Numeric-only predicates: no inverted list, full-scan brute force.
		return nil, false
	}
}

// estimateSelectivity returns the exact matching fraction (attributes are
// cheap to scan once and cached at query-set build time by callers; here we
// accept an externally measured value via the argument in hot paths).
// FilteredSearchOptions configures a filtered query.
type FilteredSearchOptions struct {
	K             int
	Ef            int
	UsePQ         bool // traverse the graph with PQ approximate distances
	RecordPath    bool
	HybridEfBoost int // extra ef multiplier when the hybrid path retries
}

// FilteredResult extends hnsw.SearchResult with strategy diagnostics.
type FilteredResult struct {
	hnsw.SearchResult
	Strategy     string  `json:"strategy"`
	Selectivity  float64 `json:"selectivity"`
	MatchedCount int     `json:"matched_count"`
	RerankedRaw  bool    `json:"reranked_raw"`
	BruteScanned int     `json:"brute_scanned"`
}

// FilteredSearch executes a query with automatic strategy selection.
//
// selectivity is the exact fraction of rows matching f (callers compute it
// once per filter shape; nil f => 1.0).
func (s *Store) FilteredSearch(query []float32, f *synth.FilterSpec,
	selectivity float64, opts FilteredSearchOptions) FilteredResult {

	s.buildMu.Lock()
	// buildMu only serializes structural mutations; graph Search itself
	// takes the graph's RWMutex, so we release immediately.
	s.buildMu.Unlock()

	if opts.HybridEfBoost <= 0 {
		opts.HybridEfBoost = 2
	}
	// PQ traversal is approximate: fetch a candidate pool several times
	// larger than k, then rerank with exact float32 distances and truncate.
	// Re-ranking only the final k points cannot recover neighbors PQ
	// mis-ranked out of the pool, so the pool size is the main recall knob.
	pqMode := opts.UsePQ && s.HasPQ()
	fetchK := opts.K
	if pqMode {
		fetchK = opts.K * rerankFactor
	}
	out := FilteredResult{Selectivity: selectivity}
	pred := s.predicate(f)

	var src hnsw.DistanceSource
	if pqMode {
		src = s.pqDistSource(query)
	} else {
		src = s.rawDistSource(query)
	}

	switch {
	case f == nil || selectivity >= s.thr.Wide:
		// Wide / unfiltered: graph walk, non-matches remain routing nodes.
		out.Strategy = StrategyGraphRelaxed
		mode := hnsw.FilterOff
		if f != nil {
			mode = hnsw.FilterRelaxed
		}
		graphEf := opts.Ef
		if pqMode && graphEf < fetchK {
			graphEf = fetchK
		}
		out.SearchResult = s.idx.Search(src, hnsw.SearchOptions{
			K: fetchK, Ef: graphEf, Predicate: pred, Mode: mode,
			RecordPath: opts.RecordPath,
		})
		out.MatchedCount = int(selectivity * float64(s.ds.N))

	case selectivity <= s.thr.Strict:
		// Strict: brute force over the matching subset. Posting lists make
		// this cheap when labels are involved; numeric-only filters scan
		// all rows with the predicate. In PQ mode the subset is ranked with
		// ADC table lookups (no float32 distance over the full dimension),
		// then the pool is reranked exactly.
		out.Strategy = StrategyBruteForce
		scanK := opts.K
		if pqMode {
			scanK = fetchK
		}
		out.IDs, out.Dists, out.BruteScanned = s.bruteForceWith(query, f, pred, scanK, pqMode)
		out.MatchedCount = out.BruteScanned

	default:
		// Middle band. A relaxed graph walk usually reaches enough matches
		// once ef is widened, so try progressively larger candidate queues
		// before paying for a brute-force scan. Only when the matching
		// points form true islands (graph returns nothing even at high ef)
		// do we fall back to brute force on the candidate subset.
		out.Strategy = StrategyHybrid
		baseEf := opts.Ef * opts.HybridEfBoost
		if pqMode && baseEf < fetchK {
			baseEf = fetchK
		}
		var graphRes hnsw.SearchResult
		for _, mult := range []int{1, 4} {
			graphRes = s.idx.Search(src, hnsw.SearchOptions{
				K: fetchK, Ef: baseEf * mult, Predicate: pred,
				Mode: hnsw.FilterRelaxed, RecordPath: opts.RecordPath,
			})
			if len(graphRes.IDs) >= opts.K {
				break
			}
		}
		cands, hasList := s.candidateSet(f)
		if hasList {
			out.MatchedCount = len(cands)
		} else {
			out.MatchedCount = int(selectivity * float64(s.ds.N))
		}
		if len(graphRes.IDs) >= opts.K {
			out.SearchResult = graphRes
		} else {
			ids, dists, scanned := s.bruteForceWith(query, f, pred, fetchK, pqMode)
			out.SearchResult = hnsw.SearchResult{
				IDs: ids, Dists: dists, Path: graphRes.Path,
				Visited: graphRes.Visited,
			}
			out.BruteScanned = scanned
			out.Strategy = StrategyHybrid + "+brute_fallback"
		}
	}

	// Rerank: when the graph/scan used PQ approximate distances, re-evaluate
	// the candidate pool with exact float32 distances before truncating to k.
	if pqMode && len(out.IDs) > 0 {
		out = s.rerank(query, out, opts.K)
		out.RerankedRaw = true
	}
	return out
}

// rerank re-evaluates exact distances (and pulls extra PQ candidates when
// available) then keeps the true k nearest.
func (s *Store) rerank(query []float32, in FilteredResult, k int) FilteredResult {
	type pair struct {
		id int32
		d  float32
	}
	ps := make([]pair, len(in.IDs))
	for i, id := range in.IDs {
		off := int(id) * s.ds.Dim
		ps[i] = pair{id, hnsw.L2Squared(query, s.ds.Vectors[off:off+s.ds.Dim])}
	}
	sort.Slice(ps, func(i, j int) bool { return ps[i].d < ps[j].d })
	if len(ps) > k {
		ps = ps[:k]
	}
	in.IDs = make([]int32, len(ps))
	in.Dists = make([]float32, len(ps))
	for i, p := range ps {
		in.IDs[i], in.Dists[i] = p.id, p.d
	}
	return in
}
