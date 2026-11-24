// Package hnsw implements a hierarchical navigable small world graph index.
//
// The graph has multiple layers: upper layers are sparse and act as
// long-range express routes, the bottom layer (0) is dense and contains
// every node. Insertion greedily descends from the top entry point, then
// greedily searches the target layer with a bounded candidate queue (ef)
// and picks neighbors with a diversity-preserving heuristic. Deleted nodes
// are tombstoned; once tombstones exceed a threshold the surviving graph
// is rebuilt (local edge repair).
package hnsw

import (
	"math"
	"sync"
)

// DistanceSource abstracts distance evaluation so the same graph can run
// with exact float32 vectors or with PQ-encoded approximate distances.
type DistanceSource interface {
	Dist(id int32) float32
}

// Predicate returns true for nodes allowed by a filter. nil means no filter.
type Predicate func(id int32) bool

// FilterMode controls how the graph search treats a predicate:
//   - FilterOff: no predicate;
//   - FilterRelaxed: traverse all nodes, only return matching ones
//     (wide filters — non-matching nodes still route the search);
//   - FilterStrict: matching nodes must also be expanded (used on the
//     candidate subset produced by posting lists / brute force).
type FilterMode int

const (
	FilterOff FilterMode = iota
	FilterRelaxed
	FilterStrict
)

// SearchOptions configures one graph search.
type SearchOptions struct {
	K          int
	Ef         int
	Predicate  Predicate
	Mode       FilterMode
	RecordPath bool
}

// SearchResult is one query's outcome.
type SearchResult struct {
	IDs      []int32    `json:"ids"`
	Dists    []float32  `json:"dists"`
	Path     []PathStep `json:"path"`     // graph traversal trace (RecordPath only)
	Visited  int        `json:"visited"`  // nodes whose distance was evaluated
	Expanded int        `json:"expanded"` // nodes whose neighbor lists were popped
}

// PathStep records one expansion for web visualization.
type PathStep struct {
	ID        int32   `json:"id"`
	Layer     int     `json:"layer"`
	Dist      float32 `json:"dist"`
	Match     bool    `json:"match"`
	FromEntry bool    `json:"from_entry"`
}

// node holds per-node adjacency. Neighbors[0] is the dense bottom layer.
type node struct {
	neighbors [][]int32
	deleted   bool
}

// Index is the HNSW graph.
type Index struct {
	mu sync.RWMutex

	M           int // degree at layers >= 1
	MaxM0       int // degree cap at layer 0 (typically 2*M)
	EfConstruct int
	LevelMul    float64 // 1/ln(M)

	nodes []*node
	entry int32
	topL  int

	// Generation-stamped visited set avoids clearing a big array per query.
	visited    []int32
	visitStamp int32

	// edgeDistFn evaluates pairwise node distances during neighbor
	// pruning. It is installed by the store around each Insert.
	edgeDistFn func(a, b int32) float32

	// Stats
	deletedCount int
	// tombstonesSinceRepair counts tombstones accumulated after the most
	// recent compaction. It implements repair hysteresis: after a rebuild
	// the counter resets so a single further delete cannot immediately
	// trigger another O(N) rebuild.
	tombstonesSinceRepair int
	repairRuns            int
}

// Config holds index construction parameters.
type Config struct {
	M           int
	MaxM0       int
	EfConstruct int
}

// DefaultConfig returns the standard HNSW parameter set (M=16, Mmax0=32).
func DefaultConfig() Config {
	return Config{M: 16, MaxM0: 32, EfConstruct: 128}
}

// New creates an empty graph preallocated for capacity nodes.
func New(cfg Config, capacity int) *Index {
	if cfg.M <= 0 {
		cfg = DefaultConfig()
	}
	if cfg.MaxM0 <= 0 {
		cfg.MaxM0 = 2 * cfg.M
	}
	if cfg.EfConstruct <= 0 {
		cfg.EfConstruct = 128
	}
	return &Index{
		M:           cfg.M,
		MaxM0:       cfg.MaxM0,
		EfConstruct: cfg.EfConstruct,
		LevelMul:    1 / math.Log(float64(cfg.M)),
		entry:       -1,
		topL:        -1,
		visited:     make([]int32, capacity),
	}
}

// Len returns the number of live nodes.
func (idx *Index) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.nodes) - idx.deletedCount
}

// DeletedFraction reports the current tombstone ratio.
func (idx *Index) DeletedFraction() float64 {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	if len(idx.nodes) == 0 {
		return 0
	}
	return float64(idx.deletedCount) / float64(len(idx.nodes))
}

// RepairRuns returns how many times the graph has been compacted.
func (idx *Index) RepairRuns() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.repairRuns
}

// randomLevel samples an exponentially decaying layer assignment.
// Capped at 16 so a single unlucky draw cannot dominate the graph.
func (idx *Index) randomLevel(rngUint func() uint64) int {
	// -ln(uniform) * levelMul, implemented via a passed-in random source
	// to keep the graph package free of global RNG state.
	u := float64(rngUint()>>11) / float64(uint64(1)<<53)
	if u < 1e-12 {
		u = 1e-12
	}
	l := int(-math.Log(u) * idx.LevelMul)
	if l > 16 {
		l = 16
	}
	return l
}

// SetEdgeDist installs the pairwise distance closure used by the
// diversity heuristic during insertion and rebuild.
func (idx *Index) SetEdgeDist(fn func(a, b int32) float32) {
	idx.mu.Lock()
	idx.edgeDistFn = fn
	idx.mu.Unlock()
}

// markVisited stamps id and reports whether it was already visited.
func (idx *Index) markVisited(id int32, stamp int32) bool {
	if int(id) >= len(idx.visited) {
		// Should not happen: visited is grown alongside nodes.
		extra := make([]int32, int(id)+1-len(idx.visited))
		idx.visited = append(idx.visited, extra...)
	}
	if idx.visited[id] == stamp {
		return true
	}
	idx.visited[id] = stamp
	return false
}

// freshStamp returns a generation id not currently present in the visited
// array. Every layer traversal within one operation gets its own stamp so
// that, e.g., a node visited during the upper-layer descent can still be
// expanded on the bottom layer.
func (idx *Index) freshStamp() int32 {
	idx.visitStamp++
	if idx.visitStamp == 0 {
		clearInt32(idx.visited)
		idx.visitStamp = 1
	}
	return idx.visitStamp
}
