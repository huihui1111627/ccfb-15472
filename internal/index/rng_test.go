package index

import (
	"math"
	"math/rand/v2"
)

// 轻量随机工具, 供测试辅助使用。
func rngNew(seed int) *rand.Rand {
	return rand.New(rand.NewPCG(uint64(seed), uint64(seed)^0x9e3779b9))
}

func rngIntn(r *rand.Rand, n int) int { return r.IntN(n) }

func rngGauss(r *rand.Rand) float32 {
	return float32(r.NormFloat64())
}

func normalizeInPlace(v []float32) {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	inv := float32(1.0 / math.Sqrt(s))
	for i := range v {
		v[i] *= inv
	}
}
