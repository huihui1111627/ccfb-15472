// Package sim 生成仿真的嵌入向量与结构化属性。
//
// 向量由高斯混合模型(GMM)生成,具有明显的簇结构,再做 L2 归一化,
// 与真实文本/图像嵌入的形态一致:单位超球面上形成多个高密度团块。
// 每个向量附带真实检索中常见的过滤属性:租户(低基数)、类目、
// 时间戳、区域、价格(偏态数值),以及多值标签集合。
package sim

import (
	"math"
	"math/rand/v2"
	"runtime"
	"sync"

	"vecsearch/internal/store"
)

// Vector 一条仿真记录的原始向量(调用方写入列式存储,这里仅用于生成阶段)。
type Vector = []float32

// Attrs 一条记录的结构化属性(定义在 store 包, 这里做别名便于生成侧使用)。
type Attrs = store.Attrs

// Dataset 仿真数据集(列式)。
type Dataset struct {
	N, Dim, Clusters, Tenants, Categories, Regions, NumLabels int
	T0, TSpan                                                 uint32
	Vectors                                                   []float32 // 长度 N*Dim, 行主序
	Tenant                                                    []uint16
	Category                                                  []uint16
	Timestamp                                                 []uint32
	Region                                                    []uint8
	Price                                                     []float32
	Labels                                                    [][]uint16
	// 生成簇心,供查询生成复用(查询向量也按相同 GMM 采样)
	centers []Vector
	rg      *rand.Rand
}

// Config 生成参数。
type Config struct {
	N, Dim, Clusters, Tenants, Categories, Regions, NumLabels int
	Seed                                                      uint64
	// T0/TSpan 时间戳范围
	T0, TSpan uint32
}

func DefaultConfig(n int) Config {
	return Config{
		N: n, Dim: 128, Clusters: 256, Tenants: 64, Categories: 128,
		Regions: 12, NumLabels: 256, T0: 1_600_000_000, TSpan: 90 * 86400,
		Seed: 42,
	}
}

// gaussian Box-Muller 标准正态。
func gaussian(rng *rand.Rand) float32 {
	// rand/v2 没有标准正态方法(只有标准正态的顶层函数被移除),自己实现
	u1 := math.Max(rng.Float64(), 1e-12)
	u2 := rng.Float64()
	return float32(math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2))
}

// Generate 按 GMM + 相关属性生成数据集,向量生成并行化。
func Generate(cfg Config) *Dataset {
	if cfg.Dim == 0 {
		cfg = DefaultConfig(cfg.N)
	}
	rng := rand.New(rand.NewPCG(cfg.Seed, cfg.Seed^0x9e3779b9))
	ds := &Dataset{
		N: cfg.N, Dim: cfg.Dim, Clusters: cfg.Clusters,
		Tenants: cfg.Tenants, Categories: cfg.Categories, Regions: cfg.Regions,
		NumLabels: cfg.NumLabels, T0: cfg.T0, TSpan: cfg.TSpan,
		Vectors: make([]float32, cfg.N*cfg.Dim),
		Tenant:  make([]uint16, cfg.N), Category: make([]uint16, cfg.N),
		Timestamp: make([]uint32, cfg.N), Region: make([]uint8, cfg.N),
		Price: make([]float32, cfg.N), Labels: make([][]uint16, cfg.N),
		rg: rng,
	}
	// 簇心: 各向同性高斯,归一化后落在单位超球面上
	ds.centers = make([]Vector, cfg.Clusters)
	for k := range ds.centers {
		c := make(Vector, cfg.Dim)
		for d := range c {
			c[d] = gaussian(rng)
		}
		normalize(c)
		ds.centers[k] = c
	}
	// 每个簇的采样权重: Dirichlet 式的不均匀权重(幂律),制造大小簇
	weights := make([]float64, cfg.Clusters)
	var wsum float64
	for k := range weights {
		weights[k] = 1.0 / math.Pow(float64(k+1), 0.8)
		wsum += weights[k]
	}
	cdf := make([]float64, cfg.Clusters)
	var acc float64
	for k, w := range weights {
		acc += w / wsum
		cdf[k] = acc
	}

	// 预生成每条记录的簇归属(主 rng 串行,保证可复现)
	assign := make([]int, cfg.N)
	for i := range assign {
		r := rng.Float64()
		lo, hi := 0, cfg.Clusters
		for lo < hi {
			mid := (lo + hi) / 2
			if cdf[mid] < r {
				lo = mid + 1
			} else {
				hi = mid
			}
		}
		assign[i] = lo
	}

	// 并行生成向量: 每批一个 worker, 用确定性派生 rng
	workers := runtime.NumCPU()
	batch := (cfg.N + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo := w * batch
		hi := min(lo+batch, cfg.N)
		if lo >= hi {
			continue
		}
		wg.Add(1)
		go func(lo, hi, worker int) {
			defer wg.Done()
			wr := rand.New(rand.NewPCG(cfg.Seed+uint64(worker)*1_000_003, 777))
			buf := make(Vector, cfg.Dim)
			for i := lo; i < hi; i++ {
				c := ds.centers[assign[i]]
				off := i * cfg.Dim
				v := ds.Vectors[off : off+cfg.Dim : off+cfg.Dim]
				// 簇心 + 0.35 倍噪声, 簇间分离明显且簇内有展宽
				for d := 0; d < cfg.Dim; d++ {
					buf[d] = c[d] + 0.35*gaussian(wr)
				}
				normalize(buf)
				copy(v, buf)
			}
		}(lo, hi, w)
	}
	wg.Wait()

	// 属性: 与簇有弱相关,使"过滤 + 向量"场景更真实
	for i := 0; i < cfg.N; i++ {
		// 租户: 幂律, 80% 数据落在前 20% 租户
		tenant := int(math.Pow(float64(cfg.Tenants), rng.Float64()*1.7) * 0.9999)
		ds.Tenant[i] = uint16(min(tenant, cfg.Tenants-1))
		// 类目: 与簇弱相关
		ds.Category[i] = uint16((assign[i]*7 + rng.IntN(cfg.Categories/4+1)) % cfg.Categories)
		ds.Timestamp[i] = cfg.T0 + uint32(rng.Float64()*float64(cfg.TSpan))
		ds.Region[i] = uint8(rng.IntN(cfg.Regions))
		// 价格: 对数正态
		ds.Price[i] = float32(math.Exp(1.2 + 0.9*rng.NormFloat64()))
		// 标签: 每条 1~4 个
		nl := 1 + rng.IntN(4)
		labs := make([]uint16, nl)
		for j := range labs {
			labs[j] = uint16(rng.IntN(cfg.NumLabels))
		}
		ds.Labels[i] = labs
	}
	return ds
}

// Query 一个仿真查询: 查询向量 + 可选过滤条件。
type Query struct {
	Vector Vector
	Filter *store.Filter
}

// GenQueries 从同一 GMM 采样查询向量(与数据同分布,召回有意义)。
func (ds *Dataset) GenQueries(n int, seed uint64) []Vector {
	rng := rand.New(rand.NewPCG(seed, seed^0xabcdef))
	qs := make([]Vector, n)
	for i := range qs {
		k := rng.IntN(ds.Clusters)
		c := ds.centers[k]
		q := make(Vector, ds.Dim)
		for d := range q {
			q[d] = c[d] + 0.35*gaussian(rng)
		}
		normalize(q)
		qs[i] = q
	}
	return qs
}

func normalize(v []float32) {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	if s == 0 {
		return
	}
	inv := float32(1.0 / math.Sqrt(s))
	for i := range v {
		v[i] *= inv
	}
}

// L2Squared 返回两个向量的平方 L2 距离。向量均已归一化,
// 与余弦距离只差常数,排序等价。
func L2Squared(a, b []float32) float32 {
	var s float32
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}
