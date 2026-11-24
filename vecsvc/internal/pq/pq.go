// Package pq implements product quantization compression for vectors.
//
// Each d-dimensional vector is split into m sub-vectors; every sub-space
// keeps k centroids learned by k-means, and a vector is encoded as m bytes
// (one centroid id per sub-space). Search uses Asymmetric Distance
// Computation: per query we precompute the m*k table of query-to-centroid
// distances, then an encoded distance is m table lookups plus additions —
// no float arithmetic over the original dimension and ~16x less memory.
package pq

import (
	"fmt"
	"sync"

	"vecsvc/internal/synth"
)

const (
	// K is the number of centroids per sub-quantizer (byte codes => 256).
	K = 256
	// KMeansIters bounds Lloyd iterations when training.
	KMeansIters = 10
)

// Model is a trained product quantizer.
type Model struct {
	Dim       int
	M         int // number of sub-quantizers
	SubDim    int // Dim / M
	K         int // centroids per sub-quantizer (<=256, codes stay one byte)
	Centroids []float32
}

// New trains a PQ model on a sample of the dataset (training on the full
// dataset is unnecessary and far too expensive; a few tens of thousands of
// sampled vectors give stable centroids). The number of centroids adapts
// to the sample size so small smoke tests still train successfully.
func New(ds *synth.Dataset, m, sampleSize int, seed uint64, workers int) (*Model, error) {
	if ds.Dim%m != 0 {
		return nil, fmt.Errorf("dim %d not divisible by m=%d", ds.Dim, m)
	}
	subDim := ds.Dim / m
	if sampleSize > ds.N {
		sampleSize = ds.N
	}
	// k-means needs strictly more points than centroids; leave headroom so
	// empty-cluster reseeding always has a donor point.
	k := K
	if sampleSize < k*2 {
		k = sampleSize / 2
	}
	if k < 2 {
		return nil, fmt.Errorf("PQ sample too small: %d rows", sampleSize)
	}
	rng := synth.NewPCG64(seed, 0x51ed270b)

	// Uniform-stride sampling with a random offset.
	sampled := make([]int, sampleSize)
	step := ds.N / sampleSize
	if step < 1 {
		step = 1
	}
	offset := rng.Intn(step + 1)
	for i := range sampled {
		sampled[i] = (i*step + offset) % ds.N
	}

	model := &Model{
		Dim:       ds.Dim,
		M:         m,
		SubDim:    subDim,
		K:         k,
		Centroids: make([]float32, m*k*subDim),
	}

	// Each sub-space trains independently and in parallel.
	var wg sync.WaitGroup
	sem := make(chan struct{}, workers)
	errs := make([]error, m)
	for sub := 0; sub < m; sub++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(sub int) {
			defer wg.Done()
			defer func() { <-sem }()
			errs[sub] = trainSubspace(ds, sampled, sub, subDim, k,
				model.Centroids[sub*k*subDim:(sub+1)*k*subDim], rng.Uint64())
		}(sub)
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	return model, nil
}

func subDist(a, b []float32) float32 {
	var d float32
	for j := range a {
		x := a[j] - b[j]
		d += x * x
	}
	return d
}

func trainSubspace(ds *synth.Dataset, rows []int, sub, subDim, k int, centroids []float32, seed uint64) error {
	rng := synth.NewPCG64(seed, seed^0xdeadbeef)
	n := len(rows)
	if n < k {
		return fmt.Errorf("PQ sample (%d) smaller than k (%d)", n, k)
	}
	lo := sub * subDim
	subView := func(row int) []float32 {
		vo := row*ds.Dim + lo
		return ds.Vectors[vo : vo+subDim]
	}

	// Initial centroids: k distinct random sample points. Full k-means++
	// initialization is O(n*k^2) and dominates everything else; in the
	// low-dimensional sub-spaces of a product quantizer (sub-dim 4..16),
	// random init plus a few Lloyd iterations with empty-cluster reseeding
	// reaches essentially the same distortion at a fraction of the cost.
	picked := make(map[int]bool, k)
	for c := 0; c < k; c++ {
		var row int
		for {
			row = rows[rng.Intn(n)]
			if !picked[row] {
				picked[row] = true
				break
			}
		}
		copy(centroids[c*subDim:(c+1)*subDim], subView(row))
	}

	assign := make([]byte, n)
	sums := make([]float32, k*subDim)
	counts := make([]int, k)
	for iter := 0; iter < KMeansIters; iter++ {
		clear(sums)
		clearInt(counts)
		moved := 0
		for i, row := range rows {
			v := subView(row)
			best := byte(0)
			bestD := subDist(v, centroids[0:subDim])
			for c := 1; c < k; c++ {
				if d := subDist(v, centroids[c*subDim:(c+1)*subDim]); d < bestD {
					bestD = d
					best = byte(c)
				}
			}
			if assign[i] != best {
				moved++
				assign[i] = best
			}
			counts[best]++
			base := int(best) * subDim
			vo := row*ds.Dim + lo
			for j := 0; j < subDim; j++ {
				sums[base+j] += ds.Vectors[vo+j]
			}
		}
		for c := 0; c < k; c++ {
			if counts[c] == 0 {
				// Re-seed empty clusters at the point farthest from its centroid.
				far, farD := 0, float32(-1)
				for i, row := range rows {
					v := subView(row)
					cc := int(assign[i]) * subDim
					if d := subDist(v, centroids[cc:cc+subDim]); d > farD {
						farD, far = d, i
					}
				}
				copy(centroids[c*subDim:(c+1)*subDim], subView(rows[far]))
				continue
			}
			inv := 1 / float32(counts[c])
			base := c * subDim
			for j := 0; j < subDim; j++ {
				centroids[base+j] = sums[base+j] * inv
			}
		}
		if moved == 0 && iter > 0 {
			break
		}
	}
	return nil
}

func clearInt(s []int) {
	for i := range s {
		s[i] = 0
	}
}

// EncodeAll encodes every dataset row into m bytes.
func (md *Model) EncodeAll(ds *synth.Dataset, workers int) []byte {
	codes := make([]byte, ds.N*md.M)
	var wg sync.WaitGroup
	sem := make(chan struct{}, workers)
	chunk := (ds.N + workers - 1) / workers
	for start := 0; start < ds.N; start += chunk {
		end := start + chunk
		if end > ds.N {
			end = ds.N
		}
		wg.Add(1)
		sem <- struct{}{}
		go func(start, end int) {
			defer wg.Done()
			defer func() { <-sem }()
			for row := start; row < end; row++ {
				md.Encode(ds.Vectors[row*md.Dim:(row+1)*md.Dim], codes[row*md.M:(row+1)*md.M])
			}
		}(start, end)
	}
	wg.Wait()
	return codes
}

// Encode encodes one vector into the provided m-byte buffer.
func (md *Model) Encode(vec []float32, code []byte) {
	for sub := 0; sub < md.M; sub++ {
		lo := sub * md.SubDim
		q := vec[lo : lo+md.SubDim]
		// Centroids are laid out as [sub0: K*SubDim][sub1: K*SubDim]...,
		// so this sub-space's block starts at sub*K*SubDim.
		block := sub * md.K * md.SubDim
		best := byte(0)
		bestD := subDist(q, md.Centroids[block:block+md.SubDim])
		coff := block + md.SubDim
		for c := 1; c < md.K; c++ {
			if d := subDist(q, md.Centroids[coff:coff+md.SubDim]); d < bestD {
				bestD = d
				best = byte(c)
			}
			coff += md.SubDim
		}
		code[sub] = best
	}
}

// ADCTable holds per-query distances from each query sub-vector to all
// centroids of that sub-quantizer.
type ADCTable struct {
	Table []float32 // M*K
	M     int
	K     int
}

// BuildADC computes the asymmetric distance table for one query vector.
func (md *Model) BuildADC(query []float32) *ADCTable {
	t := &ADCTable{Table: make([]float32, md.M*md.K), M: md.M, K: md.K}
	for sub := 0; sub < md.M; sub++ {
		qsub := query[sub*md.SubDim : (sub+1)*md.SubDim]
		base := sub * md.K
		for c := 0; c < md.K; c++ {
			t.Table[base+c] = subDist(qsub, md.Centroids[(sub*md.K+c)*md.SubDim:(sub*md.K+c+1)*md.SubDim])
		}
	}
	return t
}

// Dist returns the PQ-approximated squared distance for one encoded vector.
func (t *ADCTable) Dist(code []byte) float32 {
	var d float32
	for sub, c := range code {
		d += t.Table[sub*t.K+int(c)]
	}
	return d
}
