package index

import (
	"testing"

	"vecsearch/internal/store"
)

func TestUpdate(t *testing.T) {
	idx := buildSmokeIndex(t, 3000, false)
	q := idx.testQueries(10, 77)[0]
	st := idx.Store()
	attrs := store.Attrs{
		Tenant: st.Tenant[42], Category: st.Category[42],
		Timestamp: st.Timestamp[42], Region: st.Region[42],
		Price: st.Price[42], Labels: st.Labels[42],
	}
	// 用查询向量替换节点 42, 它应进入该查询 top
	idx.Update(42, q, attrs)
	after := idx.Search(q, nil, SearchOptions{K: 10, Ef: 128})
	found := false
	for _, id := range after.IDs {
		if id == 42 {
			found = true
		}
	}
	t.Logf("更新后节点42进入top10=%v top1=%d", found, after.IDs[0])
	if !found {
		t.Fatal("更新后该节点未被检索到")
	}
}
