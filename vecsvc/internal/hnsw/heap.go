package hnsw

// Candidate pairs a node id with its distance to the query. Two heap orders
// are needed by HNSW search:
//   - min-heap (candidate set C): always expand the closest unvisited node;
//   - max-heap (dynamic result set W): bounded at ef, ejecting the farthest.
type Candidate struct {
	ID   int32
	Dist float32
}

// MinHeap is a distance min-heap of Candidate.
type MinHeap []Candidate

func (h MinHeap) Len() int           { return len(h) }
func (h MinHeap) Less(i, j int) bool { return h[i].Dist < h[j].Dist }
func (h MinHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

// Push appends x. Intended for use with heap.Push.
func (h *MinHeap) Push(x any) { *h = append(*h, x.(Candidate)) }

// Pop removes and returns the last element. Intended for heap.Pop.
func (h *MinHeap) Pop() any {
	old := *h
	n := len(old)
	it := old[n-1]
	*h = old[:n-1]
	return it
}

// MaxHeap is a distance max-heap of Candidate.
type MaxHeap []Candidate

func (h MaxHeap) Len() int           { return len(h) }
func (h MaxHeap) Less(i, j int) bool { return h[i].Dist > h[j].Dist }
func (h MaxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

// Push appends x. Intended for use with heap.Push.
func (h *MaxHeap) Push(x any) { *h = append(*h, x.(Candidate)) }

// Pop removes and returns the last element. Intended for heap.Pop.
func (h *MaxHeap) Pop() any {
	old := *h
	n := len(old)
	it := old[n-1]
	*h = old[:n-1]
	return it
}
