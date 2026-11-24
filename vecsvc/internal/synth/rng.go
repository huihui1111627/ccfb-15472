// Package synth generates synthetic embedding-like vectors with a clear
// cluster structure plus categorical and numeric attributes used for filtered
// search experiments.
package synth

import "math"

// PCG64 is a deterministic, fast 64-bit permuted congruential generator.
// It is used instead of math/rand so dataset generation is reproducible
// across Go versions and platforms.
type PCG64 struct {
	state uint64
	inc   uint64
}

// NewPCG64 creates a PCG64 stream seeded with two 64-bit values.
func NewPCG64(seed1, seed2 uint64) *PCG64 {
	r := &PCG64{state: 0, inc: (seed2<<1 | 1)}
	r.Uint64()
	r.state += seed1
	r.Uint64()
	return r
}

// Uint64 returns the next pseudo-random uint64.
func (r *PCG64) Uint64() uint64 {
	old := r.state
	r.state = old*6364136223846793005 + r.inc
	xorshifted := uint64(((old >> 18) ^ old) >> 27)
	rot := int(old >> 59)
	return (xorshifted >> rot) | (xorshifted << ((-rot) & 31))
}

// Float64 returns a uniform pseudo-random value in [0, 1).
func (r *PCG64) Float64() float64 {
	return float64(r.Uint64()>>11) / float64(uint64(1)<<53)
}

// Intn returns a uniform pseudo-random int in [0, n). n must be > 0.
func (r *PCG64) Intn(n int) int {
	if n <= 0 {
		panic("Intn: non-positive bound")
	}
	return int(r.Uint64() % uint64(n))
}

// Gaussian returns a standard normal sample (Box-Muller).
func (r *PCG64) Gaussian() float64 {
	u := r.Float64()
	if u < 1e-15 {
		u = 1e-15
	}
	v := r.Float64()
	return math.Sqrt(-2*math.Log(u)) * math.Cos(2*math.Pi*v)
}
