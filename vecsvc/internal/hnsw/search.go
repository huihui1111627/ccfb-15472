package hnsw

import (
	"container/heap"
	"math"
)

// searchLayer is the workhorse greedy beam search on one layer.
//
// It maintains two heaps like the reference HNSW implementation:
//   - candidates (min-heap): closest unvisited node is expanded next;
//   - results    (max-heap): bounded dynamic list of the best ef nodes.
//
// In FilterRelaxed mode non-matching nodes are still expanded (they are the
// graph's routing fabric) but excluded from the returned list. In
// FilterStrict mode only matching nodes are expanded; the caller is expected
// to have arranged that matches form a well-connected subset (posting-list
// brute force or a relaxed fallback). Tombstoned nodes are never expanded
// and never returned.
//
// It returns the ascending-distance candidate list and the number of unique
// nodes whose distance was evaluated.
func (idx *Index) searchLayer(src DistanceSource, entryPoints []Candidate, ef int,
	layer int, opts SearchOptions, stamp int32) ([]Candidate, int) {

	cand := &MinHeap{}
	res := &MaxHeap{}
	heap.Init(cand)
	heap.Init(res)

	pred := opts.Predicate
	relaxed := opts.Mode == FilterRelaxed
	visited := 0

	for _, ep := range entryPoints {
		if idx.nodes[ep.ID].deleted || idx.markVisited(ep.ID, stamp) {
			continue
		}
		visited++
		heap.Push(cand, ep)
		match := pred == nil || pred(ep.ID)
		if match || relaxed {
			heap.Push(res, ep)
		}
	}

	for cand.Len() > 0 {
		c := heap.Pop(cand).(Candidate)
		farthest := float32(math.MaxFloat32)
		if res.Len() > 0 {
			farthest = (*res)[0].Dist
		}
		// Stop when the nearest unvisited candidate cannot beat the
		// current ef-th best result.
		if res.Len() >= ef && c.Dist > farthest {
			break
		}
		nd := idx.nodes[c.ID]
		if layer >= len(nd.neighbors) {
			continue
		}
		for _, nb := range nd.neighbors[layer] {
			if idx.nodes[nb].deleted || idx.markVisited(nb, stamp) {
				continue
			}
			visited++
			d := src.Dist(nb)
			match := pred == nil || pred(nb)
			if res.Len() < ef || d < farthest {
				heap.Push(cand, Candidate{ID: nb, Dist: d})
				if match || relaxed {
					heap.Push(res, Candidate{ID: nb, Dist: d})
					if res.Len() > ef {
						heap.Pop(res)
					}
				}
			}
		}
	}

	out := make([]Candidate, 0, res.Len())
	for res.Len() > 0 {
		c := heap.Pop(res).(Candidate)
		if pred == nil || pred(c.ID) {
			out = append(out, c)
		}
	}
	// heap.Pop on a max-heap yields farthest-first; reverse to ascending.
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out, visited
}

// Search runs a full top-down graph query and returns the k closest nodes.
func (idx *Index) Search(src DistanceSource, opts SearchOptions) SearchResult {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if opts.Ef < opts.K {
		opts.Ef = opts.K
	}
	res := SearchResult{}
	if idx.entry < 0 || len(idx.nodes) == 0 {
		return res
	}
	stamp := idx.freshStamp()

	pred := opts.Predicate
	var path []PathStep
	if opts.RecordPath {
		path = make([]PathStep, 0, 256)
	}
	track := func(id int32, layer int, d float32, entry bool) {
		if !opts.RecordPath || len(path) >= 600 {
			return
		}
		path = append(path, PathStep{
			ID: id, Layer: layer, Dist: d,
			Match: pred == nil || pred(id), FromEntry: entry,
		})
	}

	// Phase 1: greedy descent through the sparse upper layers (ef=1).
	ep := Candidate{ID: idx.entry, Dist: src.Dist(idx.entry)}
	track(idx.entry, idx.topL, ep.Dist, true)
	upperVisited := 1
	for layer := idx.topL; layer >= 1; layer-- {
		changed := true
		cur := ep
		for changed {
			changed = false
			nd := idx.nodes[cur.ID]
			if layer >= len(nd.neighbors) {
				break
			}
			for _, nb := range nd.neighbors[layer] {
				if idx.nodes[nb].deleted {
					continue
				}
				d := src.Dist(nb)
				track(nb, layer, d, false)
				upperVisited++
				if d < cur.Dist {
					cur = Candidate{ID: nb, Dist: d}
					changed = true
				}
			}
		}
		ep = cur
	}

	// Phase 2: bounded beam search on the dense bottom layer.
	wrapped := &trackingSource{inner: src, track: track, layer: 0}
	liveEPS := []Candidate{ep}
	if idx.nodes[ep.ID].deleted {
		id := idx.firstLive()
		if id < 0 {
			return res
		}
		liveEPS = []Candidate{{ID: id, Dist: wrapped.Dist(id)}}
	}
	bottom, bottomVisited := idx.searchLayer(wrapped, liveEPS, opts.Ef, 0, opts, stamp)
	res.Visited = upperVisited + bottomVisited
	res.Path = path

	if len(bottom) > opts.K {
		bottom = bottom[:opts.K]
	}
	res.IDs = make([]int32, len(bottom))
	res.Dists = make([]float32, len(bottom))
	for i, c := range bottom {
		res.IDs[i] = c.ID
		res.Dists[i] = c.Dist
	}
	return res
}

type trackingSource struct {
	inner DistanceSource
	track func(id int32, layer int, d float32, entry bool)
	layer int
}

func (t *trackingSource) Dist(id int32) float32 {
	d := t.inner.Dist(id)
	t.track(id, t.layer, d, false)
	return d
}

func (idx *Index) firstLive() int32 {
	for i, nd := range idx.nodes {
		if nd != nil && !nd.deleted {
			return int32(i)
		}
	}
	return -1
}

func clearInt32(s []int32) {
	for i := range s {
		s[i] = 0
	}
}
