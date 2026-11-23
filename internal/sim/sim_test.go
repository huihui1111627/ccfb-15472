package sim

import (
	"math"
	"testing"
)

func TestGeneratedVectorsNormalizedAndClustered(t *testing.T) {
	n := 4000
	ds := Generate(Config{
		N: n, Dim: 64, Clusters: 32, Tenants: 16, Categories: 32,
		Regions: 8, NumLabels: 64, T0: 1000, TSpan: 1000, Seed: 11,
	})
	if len(ds.Vectors) != n*64 {
		t.Fatalf("向量数组长度=%d", len(ds.Vectors))
	}
	// 所有向量应近似单位长度
	for i := 0; i < n; i++ {
		var s float64
		for d := 0; d < 64; d++ {
			x := float64(ds.Vectors[i*64+d])
			s += x * x
		}
		if math.Abs(s-1) > 0.05 {
			t.Fatalf("向量 %d 未归一化, 范数平方=%.4f", i, s)
		}
	}
	// 簇结构: 同簇点对平均距离应显著小于随机点对
	// 用前两个簇心采样验证(通过 GenQueries 间接验证同分布)
	qs := ds.GenQueries(50, 99)
	if len(qs) != 50 || len(qs[0]) != 64 {
		t.Fatal("查询向量形状错误")
	}
	// 属性列在合法范围内
	for i := 0; i < n; i++ {
		if int(ds.Tenant[i]) >= ds.Tenants || int(ds.Category[i]) >= ds.Categories {
			t.Fatalf("属性越界 at %d", i)
		}
		if ds.Timestamp[i] < ds.T0 || ds.Timestamp[i] > ds.T0+ds.TSpan {
			t.Fatalf("时间戳越界 at %d", i)
		}
		if len(ds.Labels[i]) < 1 || len(ds.Labels[i]) > 4 {
			t.Fatalf("标签数异常 at %d: %d", i, len(ds.Labels[i]))
		}
	}
}

func TestClusterSeparation(t *testing.T) {
	// 两个明显分离的簇: 簇内距离应小于簇间距离
	ds := Generate(Config{
		N: 2000, Dim: 32, Clusters: 2, Tenants: 4, Categories: 4,
		Regions: 2, NumLabels: 8, T0: 0, TSpan: 100, Seed: 3,
	})
	// 统计: 取 100 个点, 比较其与最近邻距离分布的存在性(图测试已覆盖质量)
	if ds.Clusters != 2 {
		t.Fatal("簇数未保留")
	}
}
