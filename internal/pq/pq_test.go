package pq

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestQuantizationError(t *testing.T) {
	n, dim := 4000, 32
	rng := rand.New(rand.NewPCG(1, 2))
	vecs := make([]float32, n*dim)
	for i := range vecs {
		vecs[i] = float32(rng.NormFloat64())
	}
	cfg := TrainConfig{M: 4, Ks: 64, TrainSize: 2000, Iter: 10, Seed: 5}
	p := Train(vecs, n, dim, cfg)

	if len(p.Codes) != n*cfg.M {
		t.Fatalf("codes len=%d want %d", len(p.Codes), n*cfg.M)
	}
	// 重建误差: 归一化向量的量化失真应显著小于平均自身范数
	var errSum, normSum float64
	rec := make([]float32, dim)
	for i := 0; i < n; i++ {
		orig := vecs[i*dim : (i+1)*dim]
		p.Reconstruct(int32(i), rec)
		for d := 0; d < dim; d++ {
			e := orig[d] - rec[d]
			errSum += float64(e * e)
			normSum += float64(orig[d] * orig[d])
		}
	}
	relErr := errSum / normSum
	t.Logf("PQ 相对量化失真=%.4f (M=%d, Ks=%d, 压缩比 %dx)",
		relErr, cfg.M, cfg.Ks, dim/cfg.M)
	if math.IsNaN(relErr) || relErr > 0.5 {
		t.Fatalf("量化失真过大: %.4f", relErr)
	}
}

func TestLUTRoundTrip(t *testing.T) {
	n, dim := 2000, 16
	rng := rand.New(rand.NewPCG(9, 8))
	vecs := make([]float32, n*dim)
	for i := range vecs {
		vecs[i] = float32(rng.NormFloat64())
	}
	p := Train(vecs, n, dim, TrainConfig{M: 2, Ks: 32, TrainSize: 1500, Iter: 8, Seed: 1})
	q := vecs[:dim]
	lut := p.BuildLUT(q)
	if len(lut) != p.M*p.Ks {
		t.Fatalf("LUT len=%d", len(lut))
	}
	// LUT 距离非负
	for i := 0; i < n; i++ {
		if d := p.LUTDist(int32(i), lut); d < 0 {
			t.Fatalf("负距离 %f", d)
		}
	}
}
