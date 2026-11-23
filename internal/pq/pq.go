// Package pq 实现乘积量化(Product Quantization)压缩。
//
// 把 D 维向量切成 M 个子空间, 每个子空间独立做 Ks 质心 k-means,
// 每个子向量用 1 字节编码(Ks=256), 一条 128 维向量从 512 字节
// 压缩到 16 字节(32x)。检索时为查询预计算 M*Ks 距离表(LUT),
// 单条压缩距离退化为 M 次查表 + 加法。
package pq

import (
	"math"
	"math/rand/v2"
	"runtime"
	"sync"
)

// PQ 训练好的量化器 + 全量编码。
type PQ struct {
	M, Ks, DSub, Dim int
	Centroids        []float32 // M*Ks*DSub, 行主序: 子空间 m 的第 k 个质心
	Codes            []uint8   // N*M, 行主序
	N                int
}

// TrainConfig 训练参数。
type TrainConfig struct {
	M         int // 子空间数(必须整除 Dim)
	Ks        int // 每子空间质心数
	TrainSize int // 训练样本上限(从全量均匀抽样)
	Iter      int // k-means 迭代轮数
	Seed      uint64
}

func DefaultTrainConfig(dim int) TrainConfig {
	return TrainConfig{M: dim / 8, Ks: 256, TrainSize: 200_000, Iter: 12, Seed: 7}
}

// Train 在抽样数据上训练 PQ。
func Train(vectors []float32, n, dim int, cfg TrainConfig) *PQ {
	if cfg.M == 0 {
		cfg = DefaultTrainConfig(dim)
	}
	if dim%cfg.M != 0 {
		panic("pq: dim must be divisible by M")
	}
	dsub := dim / cfg.M
	rng := rand.New(rand.NewPCG(cfg.Seed, cfg.Seed^0x51ed270b))

	ts := min(cfg.TrainSize, n)
	sample := make([]float32, ts*dim)
	step := float64(n) / float64(ts)
	for i := 0; i < ts; i++ {
		src := int(float64(i) * step)
		copy(sample[i*dim:(i+1)*dim], vectors[src*dim:(src+1)*dim])
	}

	p := &PQ{M: cfg.M, Ks: cfg.Ks, DSub: dsub, Dim: dim,
		Centroids: make([]float32, cfg.M*cfg.Ks*dsub), N: n}

	var wg sync.WaitGroup
	for m := 0; m < cfg.M; m++ {
		wg.Add(1)
		go func(m int) {
			defer wg.Done()
			trainSubspace(sample, ts, dim, m, dsub, cfg.Ks, cfg.Iter, rng.Uint64(),
				p.Centroids[m*cfg.Ks*dsub:(m+1)*cfg.Ks*dsub])
		}(m)
	}
	wg.Wait()

	// 全量编码
	p.Codes = make([]uint8, n*cfg.M)
	workers := runtime.NumCPU()
	batch := (n + workers - 1) / workers
	var ewg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo := w * batch
		hi := min(lo+batch, n)
		if lo >= hi {
			continue
		}
		ewg.Add(1)
		go func(lo, hi int) {
			defer ewg.Done()
			for i := lo; i < hi; i++ {
				p.Encode(vectors[i*dim:(i+1)*dim], p.Codes[i*cfg.M:(i+1)*cfg.M])
			}
		}(lo, hi)
	}
	ewg.Wait()
	return p
}

// trainSubspace 单个子空间的 Lloyd k-means, 质心随机初始化(样本点)。
func trainSubspace(sample []float32, ts, dim, m, dsub, ks, iter int, seed uint64, centroids []float32) {
	rng := rand.New(rand.NewPCG(seed, seed^1))
	// 初始化: 随机选 ks 个样本子向量
	perm := rng.Perm(ts)
	for k := 0; k < ks; k++ {
		i := perm[k]
		off := i*dim + m*dsub
		copy(centroids[k*dsub:(k+1)*dsub], sample[off:off+dsub])
	}
	assign := make([]int, ts)
	sums := make([]float32, ks*dsub)
	counts := make([]int, ks)
	for it := 0; it < iter; it++ {
		// 分配阶段(并行化按样本分片)
		assignSubspace(sample, ts, dim, m, dsub, ks, centroids, assign)
		// 更新阶段
		for i := range sums {
			sums[i] = 0
		}
		for i := range counts {
			counts[i] = 0
		}
		for i := 0; i < ts; i++ {
			k := assign[i]
			counts[k]++
			off := i*dim + m*dsub
			co := k * dsub
			for d := 0; d < dsub; d++ {
				sums[co+d] += sample[off+d]
			}
		}
		for k := 0; k < ks; k++ {
			if counts[k] == 0 {
				// 空簇: 用一个随机样本重新播种
				i := rng.IntN(ts)
				off := i*dim + m*dsub
				copy(centroids[k*dsub:(k+1)*dsub], sample[off:off+dsub])
				continue
			}
			inv := 1 / float32(counts[k])
			co := k * dsub
			for d := 0; d < dsub; d++ {
				centroids[co+d] = sums[co+d] * inv
			}
		}
	}
}

func assignSubspace(sample []float32, ts, dim, m, dsub, ks int, centroids []float32, assign []int) {
	// 小任务保持简单: 串行分配, 子空间维度低(8), 主成本在全量编码而非训练
	for i := 0; i < ts; i++ {
		off := i*dim + m*dsub
		best, bestD := 0, float32(math.MaxFloat32)
		for k := 0; k < ks; k++ {
			co := k * dsub
			d := float32(0)
			for j := 0; j < dsub; j++ {
				diff := sample[off+j] - centroids[co+j]
				d += diff * diff
			}
			if d < bestD {
				bestD, best = d, k
			}
		}
		assign[i] = best
	}
}

// Encode 编码一条向量到 out(长度 M)。
func (p *PQ) Encode(v []float32, out []uint8) {
	for m := 0; m < p.M; m++ {
		base := m * p.DSub
		best, bestD := uint8(0), float32(math.MaxFloat32)
		cbase := m * p.Ks * p.DSub
		for k := 0; k < p.Ks; k++ {
			co := cbase + k*p.DSub
			d := float32(0)
			for j := 0; j < p.DSub; j++ {
				diff := v[base+j] - p.Centroids[co+j]
				d += diff * diff
			}
			if d < bestD {
				bestD, best = d, uint8(k)
			}
		}
		out[m] = best
	}
}

// BuildLUT 为查询向量构建距离表: lut[m*Ks+k] = d(q_m, c_mk)^2。
func (p *PQ) BuildLUT(q []float32) []float32 {
	lut := make([]float32, p.M*p.Ks)
	for m := 0; m < p.M; m++ {
		qbase := m * p.DSub
		cbase := m * p.Ks * p.DSub
		for k := 0; k < p.Ks; k++ {
			co := cbase + k*p.DSub
			d := float32(0)
			for j := 0; j < p.DSub; j++ {
				diff := q[qbase+j] - p.Centroids[co+j]
				d += diff * diff
			}
			lut[m*p.Ks+k] = d
		}
	}
	return lut
}

// LUTDist 用距离表计算单条编码的近似平方 L2 距离(ADC)。
func (p *PQ) LUTDist(id int32, lut []float32) float32 {
	off := int(id) * p.M
	d := float32(0)
	for m := 0; m < p.M; m++ {
		d += lut[m*p.Ks+int(p.Codes[off+m])]
	}
	return d
}

// Reconstruct 近似重建一条向量(供调试/展示)。
func (p *PQ) Reconstruct(id int32, out []float32) {
	off := int(id) * p.M
	for m := 0; m < p.M; m++ {
		k := int(p.Codes[off+m])
		co := m*p.Ks*p.DSub + k*p.DSub
		copy(out[m*p.DSub:(m+1)*p.DSub], p.Centroids[co:co+p.DSub])
	}
}

// Bytes 压缩数据占用字节数(不含质心)。
func (p *PQ) Bytes() int { return len(p.Codes) }
