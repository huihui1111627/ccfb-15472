package store

import "testing"

func TestSelectivityAndEnumerate(t *testing.T) {
	n := 5000
	st := New(n, 8, 10, 20, 5, 50)
	for i := 0; i < n; i++ {
		st.Tenant[i] = uint16(i % 10)         // 每租户 10%
		st.Category[i] = uint16((i * 7) % 20) // 与租户错开, 近似独立
		st.Region[i] = uint8(i % 5)           // 每区域 20%
		st.Timestamp[i] = uint32(i)
		st.Price[i] = float32(i)
	}
	st.BuildPostings()

	tn := uint16(3)
	selOne := st.Selectivity(&Filter{Tenant: &tn})
	if selOne < 0.08 || selOne > 0.12 {
		t.Fatalf("单租户选择率估计偏差: %.4f", selOne)
	}
	cat := uint16(7)
	selBoth := st.Selectivity(&Filter{Tenant: &tn, Category: &cat})
	// 独立假设下 0.1*0.05=0.005, 实测交集也是 1/200
	if selBoth > 0.01 {
		t.Fatalf("组合选择率估计偏差: %.5f", selBoth)
	}

	buf := make([]int32, 0, 256)
	mt := uint32(2500) // 时间下限, 与租户条件近似独立
	f := &Filter{Tenant: &tn, MinTime: &mt}
	ids := st.Enumerate(f, buf)
	if len(ids) == 0 {
		t.Fatalf("倒排+范围枚举为空")
	}
	// 枚举结果确实满足条件
	for _, id := range ids {
		if st.Tenant[id] != tn || st.Timestamp[id] < mt {
			t.Fatalf("枚举到不满足条件的 id=%d", id)
		}
	}

	// 删除后枚举排除
	st.SetDeleted(ids[0], true)
	ids2 := st.Enumerate(f, buf)
	if len(ids2) != len(ids)-1 {
		t.Fatalf("删除后枚举数=%d want %d", len(ids2), len(ids)-1)
	}
}

func TestNoFilterEnumerateAll(t *testing.T) {
	n := 1000
	st := New(n, 4, 4, 4, 4, 4)
	for i := 0; i < n; i++ {
		st.Timestamp[i] = uint32(i)
	}
	st.BuildPostings()
	ids := st.Enumerate(nil, nil)
	if len(ids) != n {
		t.Fatalf("无过滤枚举=%d want %d", len(ids), n)
	}
}
