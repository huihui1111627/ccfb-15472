package sim

import "vecsearch/internal/pq"

// PQConfigForTest 测试/小规模场景的 PQ 配置。
func PQConfigForTest(dim int) pq.TrainConfig {
	c := pq.DefaultTrainConfig(dim)
	c.TrainSize = 20_000
	c.Iter = 8
	return c
}
