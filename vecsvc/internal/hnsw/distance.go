package hnsw

import "math"

// L2Squared returns the squared Euclidean distance between two equal-length
// vectors. The loop is 4-way unrolled so the Go compiler can vectorize it on
// arm64/amd64; it is the single hottest function in both graph search and
// brute force, so it takes raw backing-array views rather than slice headers
// per row where possible.
func L2Squared(a, b []float32) float32 {
	n := len(a)
	var d0, d1, d2, d3 float32
	i := 0
	for ; i+4 <= n; i += 4 {
		x0 := a[i] - b[i]
		x1 := a[i+1] - b[i+1]
		x2 := a[i+2] - b[i+2]
		x3 := a[i+3] - b[i+3]
		d0 += x0 * x0
		d1 += x1 * x1
		d2 += x2 * x2
		d3 += x3 * x3
	}
	var tail float32
	for ; i < n; i++ {
		x := a[i] - b[i]
		tail += x * x
	}
	return d0 + d1 + d2 + d3 + tail
}

// L2 is the plain Euclidean distance.
func L2(a, b []float32) float32 {
	return float32(math.Sqrt(float64(L2Squared(a, b))))
}
