// Package eval 实现验收评估:
//   - 不同过滤选择率档位下的召回率/时延/策略分布
//   - ef 参数对召回-时延的权衡曲线
//   - 精确 / HybridPQ / 全程PQ 三种距离模式的召回对比
//   - 大量删除 + 局部修复后的召回变化
//
// 结果以结构化 Report 输出 JSON, 供 Web 页面直接渲染。
package eval

import (
	"fmt"
	"math"
	"sort"

	"vecsearch/internal/index"
	"vecsearch/internal/sim"
	"vecsearch/internal/store"
)

// FilterCase 一个过滤用例及其实测选择率。
type FilterCase struct {
	Name        string        `json:"name"`
	Filter      *store.Filter `json:"-"`
	Selectivity float64       `json:"selectivity"`
	Matches     int           `json:"matches"`
}

// Band 选择率档位。
type Band struct {
	Label string        `json:"label"`
	Cases []*FilterCase `json:"cases"`
}

// Metrics 一组查询的聚合指标。
type Metrics struct {
	N            int            `json:"n"`
	RecallAtK    float64        `json:"recall_at_k"`
	LatencyUs    LatencyStats   `json:"latency_us"`
	Visited      float64        `json:"avg_visited"`
	StrategyDist map[string]int `json:"strategy_dist"`
}

type LatencyStats struct {
	Mean float64 `json:"mean"`
	P50  float64 `json:"p50"`
	P95  float64 `json:"p95"`
	P99  float64 `json:"p99"`
	Max  float64 `json:"max"`
}

// EfPoint ef 权衡曲线上的一点。
type EfPoint struct {
	Ef      int     `json:"ef"`
	Recall  float64 `json:"recall"`
	LatUs   float64 `json:"latency_us"`
	P95Us   float64 `json:"p95_us"`
	Visited float64 `json:"avg_visited"`
}

// DeletionPoint 删除 sweep 的一点。
type DeletionPoint struct {
	DeletedFrac    float64 `json:"deleted_frac"`
	RecallNoFilter float64 `json:"recall_no_filter"`
	RecallFiltered float64 `json:"recall_filtered"`
	RepairMs       int64   `json:"repair_ms"`
	Alive          int     `json:"alive"`
}

// Report 完整验收报告。
type Report struct {
	GeneratedAt string      `json:"generated_at"`
	N           int         `json:"n"`
	Dim         int         `json:"dim"`
	K           int         `json:"k"`
	BuildMs     int64       `json:"build_ms"`
	Stats       index.Stats `json:"stats"`

	// 选择率 -> 模式 -> 指标
	BySelectivity map[string]map[string]*Metrics `json:"by_selectivity"`
	Bands         []Band                         `json:"bands"`

	EfCurve       []EfPoint          `json:"ef_curve"`
	PQCompare     map[string]float64 `json:"pq_compare"`
	DeletionSweep []DeletionPoint    `json:"deletion_sweep"`

	// 单次查询的搜索路径样例(可视化)
	PathSample *PathSample `json:"path_sample,omitempty"`
}

type PathSample struct {
	QueryID     int            `json:"query_id"`
	Filter      string         `json:"filter"`
	Selectivity float64        `json:"selectivity"`
	Strategy    index.Strategy `json:"strategy"`
	Path        []PathNode     `json:"path"`
	ResultIDs   []int32        `json:"result_ids"`
}

type PathNode struct {
	Node        int32   `json:"node"`
	Layer       int     `json:"layer"`
	Dist        float64 `json:"dist"`
	FilteredOut bool    `json:"filtered_out"`
}

// ---- 过滤器生成 ----

// BuildBands 从数据集真实取值生成过滤器, 实测选择率后分档。
// 目标三档: 宽(>5%)、中(0.5%~5%)、严(<0.5%), 外加无过滤。
func BuildBands(ds *sim.Dataset, st *store.Store, perBand int) []Band {
	all := genFilterPool(ds, st)
	sort.Slice(all, func(i, j int) bool { return all[i].Selectivity < all[j].Selectivity })

	wide, mid, strict := &Band{Label: "宽 (>5%)"}, &Band{Label: "中 (0.5%-5%)"}, &Band{Label: "严 (<0.5%)"}
	for _, fc := range all {
		switch {
		case fc.Selectivity > 0.05 && len(wide.Cases) < perBand:
			wide.Cases = append(wide.Cases, fc)
		case fc.Selectivity <= 0.05 && fc.Selectivity > 0.005 && len(mid.Cases) < perBand:
			mid.Cases = append(mid.Cases, fc)
		case fc.Selectivity <= 0.005 && len(strict.Cases) < perBand:
			strict.Cases = append(strict.Cases, fc)
		}
	}
	bands := []Band{}
	for _, b := range []*Band{wide, mid, strict} {
		if len(b.Cases) > 0 {
			bands = append(bands, *b)
		}
	}
	return bands
}

func genFilterPool(ds *sim.Dataset, st *store.Store) []*FilterCase {
	var out []*FilterCase
	consider := func(name string, f *store.Filter) {
		buf := make([]int32, 0, 4096)
		ids := st.Enumerate(f, buf)
		sel := float64(len(ids)) / float64(st.N)
		if sel <= 0 || sel > 0.95 {
			return
		}
		out = append(out, &FilterCase{Name: name, Filter: f, Selectivity: sel, Matches: len(ids)})
	}

	// 等值条件: 取若干真实值
	seen := map[string]bool{}
	add := func(name string, f *store.Filter) {
		key := fmt.Sprintf("%v", f)
		if seen[key] {
			return
		}
		seen[key] = true
		consider(name, f)
	}
	step := ds.N / 40
	for i := 0; i < ds.N && len(out) < 60; i += max(step, 1) {
		region := ds.Region[i]
		add("region", &store.Filter{Region: &region})
		cat := ds.Category[i]
		add("category", &store.Filter{Category: &cat})
		tn := ds.Tenant[i]
		add("tenant", &store.Filter{Tenant: &tn})
		if len(ds.Labels[i]) > 0 {
			lab := ds.Labels[i][0]
			add("label", &store.Filter{Label: &lab})
		}
		// 组合条件制造更严选择率
		add("cat+region", &store.Filter{Category: &cat, Region: &region})
		add("tenant+cat", &store.Filter{Tenant: &tn, Category: &cat})
		add("tenant+cat+region", &store.Filter{Tenant: &tn, Category: &cat, Region: &region})
	}
	// 时间/价格范围
	for _, q := range []float64{0.1, 0.3, 0.5, 0.7, 0.9} {
		mt := ds.T0 + uint32(q*float64(ds.TSpan))
		add("mintime", &store.Filter{MinTime: &mt})
		mp := float32(math.Exp(1.2 + 0.9*(q*2-1)))
		add("minprice", &store.Filter{MinPrice: &mp})
	}
	return out
}

// pickFilter 为第 qi 个查询循环取档位内的过滤用例。
func pickFilter(b *Band, qi int) *FilterCase {
	return b.Cases[qi%len(b.Cases)]
}
