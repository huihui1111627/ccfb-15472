package synth

import "fmt"

// Dataset is a synthetic embedding dataset with per-vector attributes.
// Vectors are stored as one flat float32 array (row major) so that large
// datasets occupy one contiguous allocation instead of millions of slices
// that the GC would have to track.
type Dataset struct {
	N        int
	Dim      int
	Vectors  []float32 // length N*Dim
	Tenant   []uint16  // length N, low-cardinality label
	Category []uint16  // length N, low-cardinality label
	TS       []int64   // length N, numeric timestamp attribute
	Price    []float32 // length N, numeric attribute in [0,1000)
	Cluster  []uint16  // length N, ground-truth cluster id (diagnostics)

	NTenants    int
	NCategories int
	TSSpan      int64 // timestamps are uniform in [0, TSSpan)
}

// Vector returns a copy-free view of vector i. The view aliases the
// underlying array and must not be retained across dataset mutations.
func (d *Dataset) Vector(i int) []float32 {
	return d.Vectors[i*d.Dim : (i+1)*d.Dim]
}

// Config controls synthetic dataset generation.
type Config struct {
	N           int
	Dim         int
	Clusters    int
	NTenants    int
	NCategories int
	TSSpan      int64
	Seed        uint64
}

// DefaultConfig returns reasonable defaults: 128-dim vectors, 64 well
// separated clusters, 32 tenants, 16 categories.
func DefaultConfig(n int) Config {
	return Config{
		N:           n,
		Dim:         128,
		Clusters:    64,
		NTenants:    32,
		NCategories: 16,
		TSSpan:      365 * 24 * 3600, // one year of seconds
		Seed:        42,
	}
}

// Generate builds a dataset. Cluster centers are drawn from a wide normal
// distribution and points are drawn tightly around their center, giving an
// embedding-like distribution with obvious cluster structure (much more
// realistic than i.i.d. uniform/normal vectors, which are nearly mutually
// equidistant in high dimensions).
func Generate(cfg Config) (*Dataset, error) {
	if cfg.N <= 0 || cfg.Dim <= 0 || cfg.Clusters <= 0 {
		return nil, fmt.Errorf("invalid synth config: %+v", cfg)
	}
	rng := NewPCG64(cfg.Seed, 0x9e3779b97f4a7c15)

	// Cluster centers: wide spread so clusters are separable.
	centers := make([]float32, cfg.Clusters*cfg.Dim)
	for i := range centers {
		centers[i] = float32(rng.Gaussian() * 4.0)
	}

	ds := &Dataset{
		N:           cfg.N,
		Dim:         cfg.Dim,
		Vectors:     make([]float32, cfg.N*cfg.Dim),
		Tenant:      make([]uint16, cfg.N),
		Category:    make([]uint16, cfg.N),
		TS:          make([]int64, cfg.N),
		Price:       make([]float32, cfg.N),
		Cluster:     make([]uint16, cfg.N),
		NTenants:    cfg.NTenants,
		NCategories: cfg.NCategories,
		TSSpan:      cfg.TSSpan,
	}

	const pointSigma = 0.6
	for i := 0; i < cfg.N; i++ {
		c := rng.Intn(cfg.Clusters)
		ds.Cluster[i] = uint16(c)
		base := c * cfg.Dim
		off := i * cfg.Dim
		for j := 0; j < cfg.Dim; j++ {
			ds.Vectors[off+j] = centers[base+j] + float32(rng.Gaussian()*pointSigma)
		}
		ds.Tenant[i] = uint16(rng.Intn(cfg.NTenants))
		ds.Category[i] = uint16(rng.Intn(cfg.NCategories))
		ds.TS[i] = int64(rng.Uint64() % uint64(cfg.TSSpan))
		ds.Price[i] = float32(rng.Float64() * 1000)
	}
	return ds, nil
}
