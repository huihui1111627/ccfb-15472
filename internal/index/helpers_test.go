package index

// 测试辅助: 用与数据相同的 GMM 分布生成查询向量。
// 不依赖 sim 的生成器内部结构, 直接从现有数据点加噪近似,
// 保证测试自包含; 评估代码则使用 sim.GenQueries。
func (idx *Index) testQueries(n, seed int) [][]float32 {
	rng := rngNew(seed)
	out := make([][]float32, n)
	for i := range out {
		src := int32(rngIntn(rng, idx.n))
		base := idx.Vector(src)
		q := make([]float32, idx.dim)
		for d := range q {
			q[d] = base[d] + 0.15*rngGauss(rng)
		}
		normalizeInPlace(q)
		out[i] = q
	}
	return out
}
