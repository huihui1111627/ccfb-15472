package store

import (
	"vecsvc/internal/hnsw"
)

// repairThreshold is the tombstone fraction at which the graph is compacted.
const repairThreshold = 0.10

// Insert appends a new vector with attributes and links it into the graph.
// The dataset slices are grown; ids are the dataset row index.
func (s *Store) Insert(vec []float32, tenant, category uint16, ts int64, price float32,
	rngUint func() uint64) int32 {

	s.buildMu.Lock()
	defer s.buildMu.Unlock()

	id := int32(s.ds.N)
	s.ds.N++
	s.ds.Vectors = append(s.ds.Vectors, vec...)
	s.ds.Tenant = append(s.ds.Tenant, tenant)
	s.ds.Category = append(s.ds.Category, category)
	s.ds.TS = append(s.ds.TS, ts)
	s.ds.Price = append(s.ds.Price, price)
	s.ds.Cluster = append(s.ds.Cluster, 0)
	s.tenants[tenant] = append(s.tenants[tenant], id)
	s.cats[category] = append(s.cats[category], id)

	// If PQ is attached, encode the new vector and grow the codes array.
	if s.pqm != nil {
		code := make([]byte, s.pqm.M)
		s.pqm.Encode(vec, code)
		s.codes = append(s.codes, code...)
	}

	query := vec
	distOf := func(cand int32) float32 {
		off := int(cand) * s.ds.Dim
		return hnsw.L2Squared(query, s.ds.Vectors[off:off+s.ds.Dim])
	}
	s.idx.SetEdgeDist(func(a, b int32) float32 {
		da := int(a) * s.ds.Dim
		db := int(b) * s.ds.Dim
		return hnsw.L2Squared(s.ds.Vectors[da:da+s.ds.Dim],
			s.ds.Vectors[db:db+s.ds.Dim])
	})
	s.idx.Insert(id, distOf, rngUint)
	return id
}

// Delete marks a row deleted and compacts the graph once the tombstone
// fraction crosses repairThreshold. Returns true if a rebuild ran.
func (s *Store) Delete(id int32, rngUint func() uint64) bool {
	s.buildMu.Lock()
	ok := s.idx.Delete(id)
	s.buildMu.Unlock()
	if !ok {
		return false
	}
	// Repair is a side effect; the delete itself already succeeded, so its
	// return value must reflect the tombstone operation, not whether a
	// rebuild happened to be due.
	s.maybeRepair(rngUint)
	return true
}

func (s *Store) maybeRepair(rngUint func() uint64) bool {
	// Cheap fast path: a rebuild can only be due once tombstones-since-
	// repair reach threshold * live nodes. RepairPending reads counters
	// under a short RLock and avoids allocating the distance closures on
	// every single delete (which otherwise dominates delete latency).
	if !s.idx.RepairPending(repairThreshold) {
		return false
	}
	dim := s.ds.Dim
	distFactory := func(node int32) func(int32) float32 {
		va := s.ds.Vectors[int(node)*dim : int(node)*dim+dim]
		return func(cand int32) float32 {
			off := int(cand) * dim
			return hnsw.L2Squared(va, s.ds.Vectors[off:off+dim])
		}
	}
	s.idx.SetEdgeDist(func(a, b int32) float32 {
		da := int(a) * dim
		db := int(b) * dim
		return hnsw.L2Squared(s.ds.Vectors[da:da+dim],
			s.ds.Vectors[db:db+dim])
	})
	return s.idx.RepairIfNeeded(repairThreshold, distFactory, rngUint)
}

// Update replaces a vector in place. The old id is tombstoned and a new row
// is inserted so all graph edges are rebuilt through the normal insertion
// path; attributes are carried over unless overridden. The returned id is
// the new row (documented behavior — callers must not assume id stability).
func (s *Store) Update(id int32, newVec []float32, rngUint func() uint64) (int32, bool) {
	if s.idx.IsDeleted(id) {
		return -1, false
	}
	tenant, category := s.ds.Tenant[id], s.ds.Category[id]
	ts, price := s.ds.TS[id], s.ds.Price[id]
	s.Delete(id, rngUint)
	newID := s.Insert(newVec, tenant, category, ts, price, rngUint)
	return newID, true
}
