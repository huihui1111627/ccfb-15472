package hnsw

// Delete tombstones a node. Tombstones remain in adjacency lists for a while
// (cheap to skip) so bursts of deletes do not trigger O(N) work per call;
// once their fraction crosses the threshold RepairIfNeeded compacts the
// graph by rebuilding it over the surviving nodes.
func (idx *Index) Delete(id int32) bool {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	if int(id) >= len(idx.nodes) || idx.nodes[id] == nil || idx.nodes[id].deleted {
		return false
	}
	idx.nodes[id].deleted = true
	idx.deletedCount++
	idx.tombstonesSinceRepair++
	return true
}

// IsDeleted reports the tombstone state.
func (idx *Index) IsDeleted(id int32) bool {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return int(id) >= len(idx.nodes) || idx.nodes[id] == nil || idx.nodes[id].deleted
}

// DeletedCount returns the number of tombstoned nodes.
func (idx *Index) DeletedCount() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.deletedCount
}

// RepairPending is the allocation-free fast path used by the store's delete
// hot loop: it reports whether RepairIfNeeded would currently do anything.
func (idx *Index) RepairPending(threshold float64) bool {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	live := len(idx.nodes) - idx.deletedCount
	return live > 0 && float64(idx.tombstonesSinceRepair)/float64(live) >= threshold
}

// RepairIfNeeded rebuilds the graph when the tombstone fraction exceeds
// threshold. Rebuilding is the general form of local edge repair: each
// surviving node re-runs the HNSW insertion procedure, which reconnects
// every link that used to point at a deleted node and re-prunes degrees
// with the diversity heuristic. The layer assignment of surviving nodes is
// preserved to keep their long-range edges stable.
//
// Returns true when a rebuild happened.
func (idx *Index) RepairIfNeeded(threshold float64,
	distOf func(id int32) func(int32) float32, rngUint func() uint64) bool {

	idx.mu.Lock()
	// Hysteresis: repair only when tombstones added since the last
	// compaction reach threshold *fraction of currently live nodes*.
	// Basing the denominator on live nodes (not total slots) keeps the
	// trigger stable after a rebuild leaves nil slots behind, and basing
	// the numerator on post-repair tombstones prevents rebuild storms.
	live := len(idx.nodes) - idx.deletedCount
	need := len(idx.nodes) == 0 ||
		live > 0 && float64(idx.tombstonesSinceRepair)/float64(live) >= threshold
	if !need {
		idx.mu.Unlock()
		return false
	}

	// Snapshot survivors and their old levels under the lock, then build a
	// fresh graph. distOf closures are supplied by the store and read
	// immutable vector storage, so the actual rebuild can run without
	// holding the write lock the whole time — but for simplicity and
	// deterministic behavior we hold it; rebuilds are offline operations.
	old := idx.nodes
	survivors := make([]int32, 0, len(old)-idx.deletedCount)
	levels := make([]int, 0, cap(survivors))
	for i, nd := range old {
		if nd != nil && !nd.deleted {
			survivors = append(survivors, int32(i))
			levels = append(levels, len(nd.neighbors)-1)
		}
	}
	idx.mu.Unlock()

	idx.rebuild(survivors, levels, distOf, rngUint)
	return true
}

// rebuild constructs a brand-new graph over the given nodes. Node ids are
// preserved (the store keeps vectors keyed by original id); empty slots keep
// nil entries and searches skip them.
func (idx *Index) rebuild(ids []int32, levels []int,
	distOf func(id int32) func(int32) float32, rngUint func() uint64) {

	idx.mu.Lock()
	defer idx.mu.Unlock()

	newNodes := make([]*node, len(idx.nodes))
	idx.nodes = newNodes
	idx.entry = -1
	idx.topL = -1
	idx.deletedCount = 0
	idx.tombstonesSinceRepair = 0
	idx.repairRuns++
	clearInt32(idx.visited)
	idx.visitStamp = 0

	for pos, id := range ids {
		level := levels[pos]
		nd := &node{neighbors: make([][]int32, level+1)}
		idx.nodes[id] = nd
		if idx.entry < 0 {
			idx.entry = id
			idx.topL = level
			continue
		}

		src := &funcSource{dist: distOf(id)}
		ep := Candidate{ID: idx.entry, Dist: distOf(id)(idx.entry)}
		for layer := idx.topL; layer > level; layer-- {
			stamp := idx.freshStampLocked()
			found, _ := idx.searchLayer(src, []Candidate{ep}, 1, layer,
				SearchOptions{Mode: FilterOff}, stamp)
			if len(found) > 0 {
				ep = found[0]
			}
		}
		for layer := minInt(idx.topL, level); layer >= 0; layer-- {
			stamp := idx.freshStampLocked()
			candidates, _ := idx.searchLayer(src, []Candidate{ep}, idx.EfConstruct, layer,
				SearchOptions{Mode: FilterOff}, stamp)
			if len(candidates) == 0 {
				candidates = []Candidate{ep}
			}
			chosen := idx.selectNeighborsHeuristic(candidates, idx.maxM(layer))
			nd.neighbors[layer] = append(nd.neighbors[layer][:0], chosenIDs(chosen)...)
			for _, c := range chosen {
				peer := idx.nodes[c.ID]
				if peer == nil || layer >= len(peer.neighbors) {
					continue
				}
				peer.neighbors[layer] = append(peer.neighbors[layer], id)
				if len(peer.neighbors[layer]) > idx.maxM(layer) {
					all := make([]Candidate, 0, len(peer.neighbors[layer]))
					for _, pid := range peer.neighbors[layer] {
						all = append(all, Candidate{ID: pid, Dist: distOf(c.ID)(pid)})
					}
					pruned := idx.selectNeighborsHeuristic(all, idx.maxM(layer))
					peer.neighbors[layer] = chosenIDs(pruned)
				}
			}
			ep = candidates[0]
		}
		if level > idx.topL {
			idx.entry = id
			idx.topL = level
		}
	}
}

// freshStampLocked is freshStamp for callers already holding the write lock
// (rebuild). It exists to make locking intent explicit at call sites.
func (idx *Index) freshStampLocked() int32 {
	return idx.freshStamp()
}

// Stats is a point-in-time snapshot of index counters.
type Stats struct {
	TotalNodes      int     `json:"total_nodes"`
	LiveNodes       int     `json:"live_nodes"`
	DeletedNodes    int     `json:"deleted_nodes"`
	DeletedFraction float64 `json:"deleted_fraction"`
	EntryPoint      int32   `json:"entry_point"`
	TopLayer        int     `json:"top_layer"`
	RepairRuns      int     `json:"repair_runs"`
	MeanDegree0     float64 `json:"mean_degree0"`
}

// Stats collects index-level counters for the dashboard and benchmarks.
func (idx *Index) Stats() Stats {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	s := Stats{
		TotalNodes:   len(idx.nodes),
		DeletedNodes: idx.deletedCount,
		EntryPoint:   idx.entry,
		TopLayer:     idx.topL,
		RepairRuns:   idx.repairRuns,
	}
	// After a rebuild, tombstoned slots become nil; count live nodes
	// directly so the reported total reflects the compacted graph.
	deg, live, tombstones := 0, 0, 0
	for _, nd := range idx.nodes {
		switch {
		case nd == nil:
			tombstones++
		case nd.deleted:
			tombstones++
		default:
			live++
			if len(nd.neighbors) > 0 {
				deg += len(nd.neighbors[0])
			}
		}
	}
	s.LiveNodes = live
	// DeletedNodes tracks active tombstones only; after rebuild it resets
	// to zero while nil slots still represent the removed documents.
	if idx.deletedCount == 0 && tombstones > 0 {
		s.DeletedNodes = tombstones
	}
	if s.TotalNodes > 0 {
		s.DeletedFraction = float64(tombstones) / float64(s.TotalNodes)
	}
	if live > 0 {
		s.MeanDegree0 = float64(deg) / float64(live)
	}
	return s
}
