package store

import (
	"runtime"
	"sync"

	"vecsvc/internal/hnsw"
	"vecsvc/internal/pq"
	"vecsvc/internal/synth"
)

// topK is a small fixed-capacity max-heap over (id, dist) kept inline to
// avoid allocating heap.Interface objects inside the brute-force hot loop.
type topK struct {
	ids   []int32
	dists []float32
	k     int
}

func newTopK(k int) *topK {
	return &topK{ids: make([]int32, 0, k+1), dists: make([]float32, 0, k+1), k: k}
}

func (t *topK) consider(id int32, d float32) {
	if len(t.ids) < t.k {
		t.ids = append(t.ids, id)
		t.dists = append(t.dists, d)
		t.siftUp(len(t.ids) - 1)
		return
	}
	if d >= t.dists[0] {
		return
	}
	t.ids[0], t.dists[0] = id, d
	t.siftDown(0)
}

func (t *topK) siftUp(i int) {
	for i > 0 {
		p := (i - 1) / 2
		if t.dists[p] >= t.dists[i] {
			break
		}
		t.ids[p], t.ids[i] = t.ids[i], t.ids[p]
		t.dists[p], t.dists[i] = t.dists[i], t.dists[p]
		i = p
	}
}

func (t *topK) siftDown(i int) {
	n := len(t.ids)
	for {
		l, r, best := 2*i+1, 2*i+2, i
		if l < n && t.dists[l] > t.dists[best] {
			best = l
		}
		if r < n && t.dists[r] > t.dists[best] {
			best = r
		}
		if best == i {
			return
		}
		t.ids[best], t.ids[i] = t.ids[i], t.ids[best]
		t.dists[best], t.dists[i] = t.dists[i], t.dists[best]
		i = best
	}
}

// sorted pops the heap into ascending-distance slices.
func (t *topK) sorted() ([]int32, []float32) {
	ids := append([]int32(nil), t.ids...)
	dists := append([]float32(nil), t.dists...)
	// Simple insertion-style extraction via repeated max swap.
	n := len(ids)
	outIDs := make([]int32, n)
	outD := make([]float32, n)
	for size := n; size > 0; size-- {
		outIDs[size-1] = ids[0]
		outD[size-1] = dists[0]
		ids[0], dists[0] = ids[size-1], dists[size-1]
		// Re-heap the prefix [0,size).
		i := 0
		for {
			l, r, best := 2*i+1, 2*i+2, i
			if l < size-1 && dists[l] > dists[best] {
				best = l
			}
			if r < size-1 && dists[r] > dists[best] {
				best = r
			}
			if best == i {
				break
			}
			ids[best], ids[i] = ids[i], ids[best]
			dists[best], dists[i] = dists[i], dists[best]
			i = best
		}
	}
	return outIDs, outD
}

// bruteForce scans either a posting-list subset or the whole dataset
// (numeric-only filters) and returns the exact k nearest matches.
func (s *Store) bruteForce(query []float32, f *synth.FilterSpec,
	pred hnsw.Predicate, k int) (ids []int32, dists []float32, scanned int) {
	return s.bruteForceWith(query, f, pred, k, false)
}

// bruteForceWith is brute force over the matching subset, optionally using
// PQ ADC table lookups (usePQ=true) instead of full float32 distances. PQ
// mode is used to build a larger coarse-ranked pool that exact reranking
// truncates later.
func (s *Store) bruteForceWith(query []float32, f *synth.FilterSpec,
	pred hnsw.Predicate, k int, usePQ bool) (ids []int32, dists []float32, scanned int) {

	tk := newTopK(k)
	ds := s.ds
	var adc *pq.ADCTable
	if usePQ && s.HasPQ() {
		adc = s.pqm.BuildADC(query)
	}
	dist := func(id int32) float32 {
		if adc != nil {
			off := int(id) * s.pqm.M
			return adc.Dist(s.codes[off : off+s.pqm.M])
		}
		off := int(id) * ds.Dim
		return hnsw.L2Squared(query, ds.Vectors[off:off+ds.Dim])
	}
	if cands, ok := s.candidateSet(f); ok {
		scanned = len(cands)
		for _, id := range cands {
			if s.idx.IsDeleted(int32(id)) {
				continue
			}
			tk.consider(id, dist(id))
		}
	} else {
		scanned = ds.N
		for id := 0; id < ds.N; id++ {
			if pred != nil && !pred(int32(id)) {
				continue
			}
			if s.idx.IsDeleted(int32(id)) {
				continue
			}
			tk.consider(int32(id), dist(int32(id)))
		}
	}
	ids, dists = tk.sorted()
	return ids, dists, scanned
}

// GroundTruth computes exact nearest neighbors for many queries in parallel.
// It is used only by the benchmark for recall computation.
func (s *Store) GroundTruth(queries []*synth.Query, k, workers int) [][]int32 {
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	out := make([][]int32, len(queries))
	var wg sync.WaitGroup
	sem := make(chan struct{}, workers)
	for qi, q := range queries {
		wg.Add(1)
		sem <- struct{}{}
		go func(qi int, q *synth.Query) {
			defer wg.Done()
			defer func() { <-sem }()
			pred := s.predicate(q.Filter)
			ids, _, _ := s.bruteForce(q.Vector, q.Filter, pred, k)
			out[qi] = ids
		}(qi, q)
	}
	wg.Wait()
	return out
}
