package hnsw

// Insert adds a node whose distances are produced by distOf against every
// candidate id. rngUint supplies the layer-assignment randomness so callers
// keep deterministic control. The node id must equal len(nodes) at call
// time; the graph is append-only until a compaction.
func (idx *Index) Insert(id int32, distOf func(int32) float32, rngUint func() uint64) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	level := idx.randomLevel(rngUint)
	nd := &node{neighbors: make([][]int32, level+1)}
	if int(id) > len(idx.nodes) {
		idx.growNodes(int(id))
	}
	if int(id) == len(idx.nodes) {
		idx.nodes = append(idx.nodes, nd)
		if int(id) >= len(idx.visited) {
			idx.visited = append(idx.visited, make([]int32, int(id)+1-len(idx.visited))...)
		}
	} else {
		idx.nodes[id] = nd // slot reused after compaction
	}

	if idx.entry < 0 {
		idx.entry = id
		idx.topL = level
		return
	}

	src := &funcSource{dist: distOf}
	ep := Candidate{ID: idx.entry, Dist: distOf(idx.entry)}

	// Phase 1: greedy descent at layers above the new node's level.
	for layer := idx.topL; layer > level; layer-- {
		stamp := idx.freshStamp()
		found, _ := idx.searchLayer(src, []Candidate{ep}, 1, layer,
			SearchOptions{Mode: FilterOff}, stamp)
		if len(found) > 0 {
			ep = found[0]
		}
	}

	// Phase 2: search and connect at every shared layer down to 0.
	for layer := minInt(idx.topL, level); layer >= 0; layer-- {
		stamp := idx.freshStamp()
		candidates, _ := idx.searchLayer(src, []Candidate{ep}, idx.EfConstruct, layer,
			SearchOptions{Mode: FilterOff}, stamp)
		if len(candidates) == 0 {
			candidates = []Candidate{ep}
		}
		chosen := idx.selectNeighborsHeuristic(candidates, idx.maxM(layer))

		// Outgoing edges of the new node.
		nd.neighbors[layer] = append(nd.neighbors[layer][:0], chosenIDs(chosen)...)

		// Bidirectional edges with per-node degree pruning.
		for _, c := range chosen {
			peer := idx.nodes[c.ID]
			if layer >= len(peer.neighbors) {
				// Peer does not exist at this layer (should not happen
				// for layer 0); skip defensively.
				continue
			}
			peer.neighbors[layer] = append(peer.neighbors[layer], id)
			if len(peer.neighbors[layer]) > idx.maxM(layer) {
				all := make([]Candidate, 0, len(peer.neighbors[layer]))
				for _, pid := range peer.neighbors[layer] {
					// Prune the PEER's list using distances measured from
					// the peer itself — not from the new node, whose
					// distOf closure would corrupt the heuristic.
					all = append(all, Candidate{ID: pid, Dist: idx.edgeDist(c.ID, pid)})
				}
				pruned := idx.selectNeighborsHeuristic(all, idx.maxM(layer))
				peer.neighbors[layer] = chosenIDs(pruned)
			}
		}

		if len(candidates) > 0 {
			ep = candidates[0]
		}
	}

	if level > idx.topL {
		idx.entry = id
		idx.topL = level
	}
}

// selectNeighborsHeuristic implements the HNSW neighbor-selection heuristic
// (algorithm 4 in the Malkov & Yashunin paper). Plain nearest-M selection
// produces hubs and one-way links in clustered data; the heuristic keeps a
// candidate only if it is closer to the query than to any already-selected
// neighbor, which preserves diversity and keeps the graph connected.
func (idx *Index) selectNeighborsHeuristic(candidates []Candidate, m int) []Candidate {
	if len(candidates) <= m {
		out := make([]Candidate, len(candidates))
		copy(out, candidates)
		return out
	}
	work := make([]Candidate, len(candidates))
	copy(work, candidates)
	// candidates arrive ascending by distance; maintain that invariant.
	selected := make([]Candidate, 0, m)
	for len(work) > 0 && len(selected) < m {
		// Pop the closest remaining candidate.
		best := work[0]
		work = work[1:]
		good := true
		for _, s := range selected {
			// Distance best<->s is evaluated through the shared graph
			// distance closure stored on the index (set per insert).
			d := idx.edgeDist(best.ID, s.ID)
			if d < best.Dist {
				good = false
				break
			}
		}
		if good {
			selected = append(selected, best)
		}
	}
	// Fallback: if the heuristic pruned too aggressively (e.g. tiny
	// degenerate neighborhoods), fill with plain nearest neighbors.
	if len(selected) == 0 {
		for i := 0; i < m && i < len(candidates); i++ {
			selected = append(selected, candidates[i])
		}
	}
	return selected
}

func (idx *Index) maxM(layer int) int {
	if layer == 0 {
		return idx.MaxM0
	}
	return idx.M
}

func chosenIDs(cs []Candidate) []int32 {
	out := make([]int32, len(cs))
	for i, c := range cs {
		out[i] = c.ID
	}
	return out
}

func (idx *Index) growNodes(n int) {
	extra := make([]*node, n-len(idx.nodes))
	idx.nodes = append(idx.nodes, extra...)
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// edgeDist evaluates pairwise distances during insertion. The store
// registers a closure in edgeDistFn before each Insert; searches never use
// this path.
func (idx *Index) edgeDist(a, b int32) float32 {
	return idx.edgeDistFn(a, b)
}

type funcSource struct {
	dist func(int32) float32
}

func (f *funcSource) Dist(id int32) float32 { return f.dist(id) }
