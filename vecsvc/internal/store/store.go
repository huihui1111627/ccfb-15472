// Package store ties the HNSW graph, product quantization and synthetic
// attributes into one queryable vector store and implements the filtered
// search strategy selection (relaxed graph walk / hybrid / brute force).
package store

import (
	"runtime"
	"sync"

	"vecsvc/internal/hnsw"
	"vecsvc/internal/pq"
	"vecsvc/internal/synth"
)

// rerankFactor is the PQ candidate-pool multiplier: the approximate search
// fetches rerankFactor*k points, which exact reranking then truncates to k.
// PQ-ADC traversal drifts toward approximate neighborhoods, so a generous
// pool is needed to keep the true nearest neighbors reachable.
const rerankFactor = 16

// Strategy names returned to the dashboard / benchmark.
const (
	StrategyGraphRelaxed = "graph_relaxed" // wide filter: walk through non-matches
	StrategyHybrid       = "hybrid"        // middle: graph first, brute-force fallback
	StrategyBruteForce   = "brute_force"   // strict: scan the matching subset
)

// Thresholds for strategy selection, expressed on filter selectivity.
type Thresholds struct {
	// Wide filters (selectivity >= Wide): relaxed graph walk only.
	Wide float64
	// Strict filters (selectivity <= Strict): brute force the subset.
	Strict float64
}

// DefaultThresholds: above 8% the graph walk is productive; below 0.5% the
// candidate subset is small enough that brute force wins on both latency
// and recall; in between we try the graph and fall back to brute force when
// too few matches are reached.
func DefaultThresholds() Thresholds {
	return Thresholds{Wide: 0.08, Strict: 0.005}
}

// Store is the full vector retrieval service state.
type Store struct {
	ds    *synth.Dataset
	idx   *hnsw.Index
	pqm   *pq.Model
	codes []byte

	// Inverted lists for strict/hybrid filters.
	tenants [][]int32
	cats    [][]int32

	thr Thresholds

	buildMu sync.Mutex // serializes inserts/deletes/repairs at this layer
}

// New creates an empty store over the given dataset.
func New(ds *synth.Dataset, cfg hnsw.Config, thr Thresholds) *Store {
	capN := ds.N + ds.N/8
	s := &Store{
		ds:      ds,
		idx:     hnsw.New(cfg, capN),
		thr:     thr,
		tenants: make([][]int32, ds.NTenants),
		cats:    make([][]int32, ds.NCategories),
	}
	s.buildPostings()
	return s
}

func (s *Store) buildPostings() {
	tenantCap := s.ds.N/s.ds.NTenants + 16
	catCap := s.ds.N/s.ds.NCategories + 16
	for i := range s.tenants {
		s.tenants[i] = make([]int32, 0, tenantCap)
	}
	for i := range s.cats {
		s.cats[i] = make([]int32, 0, catCap)
	}
	for i := 0; i < s.ds.N; i++ {
		s.tenants[s.ds.Tenant[i]] = append(s.tenants[s.ds.Tenant[i]], int32(i))
		s.cats[s.ds.Category[i]] = append(s.cats[s.ds.Category[i]], int32(i))
	}
}

// SetPQ attaches a trained product quantizer and its codes, enabling
// compressed-distance graph traversal.
func (s *Store) SetPQ(m *pq.Model, codes []byte) {
	s.pqm = m
	s.codes = codes
}

// Index exposes the graph (benchmarks need direct access for build timing).
func (s *Store) Index() *hnsw.Index { return s.idx }

// Dataset exposes the underlying synthetic dataset.
func (s *Store) Dataset() *synth.Dataset { return s.ds }

// HasPQ reports whether compressed traversal is available.
func (s *Store) HasPQ() bool { return s.pqm != nil }

// rawDistSource returns exact float32 distances to query.
func (s *Store) rawDistSource(query []float32) hnsw.DistanceSource {
	return &rawSource{store: s, query: query}
}

type rawSource struct {
	store *Store
	query []float32
}

func (r *rawSource) Dist(id int32) float32 {
	off := int(id) * r.store.ds.Dim
	return hnsw.L2Squared(r.query, r.store.ds.Vectors[off:off+r.store.ds.Dim])
}

// pqDistSource returns ADC approximate distances to query.
func (s *Store) pqDistSource(query []float32) hnsw.DistanceSource {
	return &pqSource{table: s.pqm.BuildADC(query), codes: s.codes, m: s.pqm.M}
}

type pqSource struct {
	table *pq.ADCTable
	codes []byte
	m     int
}

func (p *pqSource) Dist(id int32) float32 {
	off := int(id) * p.m
	return p.table.Dist(p.codes[off : off+p.m])
}

// MemUsageBytes reports the process heap as measured by the runtime after
// a forced GC, plus the size of the PQ codes array (already inside heap,
// but surfaced separately so the benchmark can quote compressed vs raw
// footprint of the vector payload itself).
func (s *Store) MemUsageBytes() uint64 {
	runtime.GC()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	return ms.HeapAlloc
}

// RawVectorBytes is the byte size of the float32 vector payload.
func (s *Store) RawVectorBytes() uint64 {
	return uint64(cap(s.ds.Vectors)) * 4
}

// PQCodeBytes is the byte size of the product-quantized vector payload.
func (s *Store) PQCodeBytes() uint64 {
	return uint64(cap(s.codes))
}
