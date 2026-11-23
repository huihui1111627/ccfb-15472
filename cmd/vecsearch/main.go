// vecsearch 命令行入口。
//
//	vecsearch bench -n 2000000        完整验收评估, 结果写 web/data/report.json
//	vecsearch serve -n 200000         启动 Web 服务(交互式查询/路径/策略可视化)
//	vecsearch bench -n 10000000 -pq   千万级 + PQ 压缩
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"vecsearch/internal/eval"
	"vecsearch/internal/hnsw"
	"vecsearch/internal/index"
	"vecsearch/internal/pq"
	"vecsearch/internal/server"
	"vecsearch/internal/sim"
	"vecsearch/internal/store"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}
	cmd := os.Args[1]
	fs := flag.NewFlagSet(cmd, flag.ExitOnError)
	n := fs.Int("n", 1_000_000, "仿真向量条数")
	dim := fs.Int("dim", 128, "向量维度")
	usePQ := fs.Bool("pq", true, "启用乘积量化")
	ef := fs.Int("ef", 128, "查询候选队列大小")
	queries := fs.Int("q", 100, "评估查询数(每档)")
	profile := fs.String("profile", "balanced", "构建档位: fast(M8/efc64) | balanced(M12/efc100) | quality(M16/efc200)")
	addr := fs.String("addr", ":8080", "serve 监听地址")
	out := fs.String("out", "web/data/report.json", "报告输出路径")
	_ = fs.Parse(os.Args[2:])

	m, efc := preset(*profile)

	switch cmd {
	case "bench":
		runBench(*n, *dim, *usePQ, *ef, *queries, m, efc, *out)
	case "serve":
		runServe(*n, *dim, *usePQ, *ef, m, efc, *addr)
	default:
		usage()
		os.Exit(1)
	}
}

// preset 把构建档位映射为 HNSW 的 M 与 efConstruction。
// 仿真数据有 256 个紧密簇, 需要较大的度维持长程连接, 故默认偏稳。
func preset(profile string) (m, efc int) {
	switch profile {
	case "fast":
		return 12, 100
	case "quality":
		return 24, 200
	default: // balanced
		return 16, 200
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: vecsearch <bench|serve> [flags]")
	flag.PrintDefaults()
}

// buildIndex 生成仿真数据并构建索引, 返回评估所需的全部句柄。
func buildIndex(n, dim int, usePQ bool, m, efc int) (*index.Index, *sim.Dataset) {
	fmt.Printf("==> 生成 %d 条 %d 维仿真向量(带簇结构 + 属性)\n", n, dim)
	t0 := time.Now()
	cfg := sim.DefaultConfig(n)
	cfg.Dim = dim
	ds := sim.Generate(cfg)
	fmt.Printf("    数据生成完成 %v\n", time.Since(t0).Round(time.Millisecond))

	st := store.New(n, dim, cfg.Tenants, cfg.Categories, cfg.Regions, cfg.NumLabels)
	st.Tenant, st.Category = ds.Tenant, ds.Category
	st.Timestamp, st.Region = ds.Timestamp, ds.Region
	st.Price, st.Labels = ds.Price, ds.Labels
	st.BuildPostings()

	opts := index.DefaultOptions()
	opts.HNSW = hnsw.Config{M: m, EfConstruction: efc, Seed: 3}
	if usePQ {
		pqc := pq.DefaultTrainConfig(dim)
		if n < 200_000 {
			pqc.TrainSize = n / 2
		}
		opts.PQ = &pqc
	}
	fmt.Printf("==> 构建 HNSW 图 (M=%d, efConstruction=%d)\n", m, opts.HNSW.EfConstruction)
	idx, buildTime, err := index.Build(st, ds.Vectors, dim, opts)
	if err != nil {
		panic(err)
	}
	fmt.Printf("    索引构建完成, 耗时 %v\n", buildTime.Round(time.Millisecond))
	return idx, ds
}

func runBench(n, dim int, usePQ bool, ef, queries, m, efc int, outPath string) {
	idx, ds := buildIndex(n, dim, usePQ, m, efc)
	ec := eval.DefaultEvalConfig()
	ec.Queries = queries
	ec.Ef = ef
	ec.K = 10

	qs := ds.GenQueries(ec.Queries, ec.Seed)
	bands := eval.BuildBands(ds, idx.Store(), 6)
	if len(bands) == 0 {
		panic("没有可用的过滤档位, 检查过滤器生成")
	}

	report := &eval.Report{
		GeneratedAt: time.Now().Format(time.RFC3339),
		N:           n, Dim: dim, K: ec.K,
		BuildMs:       idx.BuildTime().Milliseconds(),
		Stats:         idx.Stats(),
		Bands:         bands,
		BySelectivity: map[string]map[string]*MetricsJSON{},
	}

	// 1) 各选择率档位 × 三模式
	fmt.Println("==> 评估不同选择率下的召回/时延/策略")
	bandResults := map[string]map[string]*eval.Metrics{}
	// 无过滤基线
	bandResults["无过滤"] = map[string]*eval.Metrics{
		"exact": eval.RunNoFilter(idx, qs, index.ModeExact, ec),
	}
	for bi := range bands {
		b := bands[bi]
		bandResults[b.Label] = map[string]*eval.Metrics{
			"exact":     eval.RunBand(idx, qs, b, index.ModeExact, ec),
			"hybrid_pq": eval.RunBand(idx, qs, b, index.ModeHybridPQ, ec),
			"full_pq":   eval.RunBand(idx, qs, b, index.ModePQ, ec),
		}
		for mode, mm := range bandResults[b.Label] {
			fmt.Printf("    [%-12s] %-9s recall=%.4f p50=%.0fus p95=%.0fus 策略=%v\n",
				b.Label, mode, mm.RecallAtK, mm.LatencyUs.P50, mm.LatencyUs.P95, mm.StrategyDist)
		}
	}
	report.BySelectivity = toJSONMetrics(bandResults)

	// 2) ef 权衡曲线
	fmt.Println("==> ef 召回-时延权衡曲线")
	report.EfCurve = eval.EfCurve(idx, qs, ec)
	for _, p := range report.EfCurve {
		fmt.Printf("    ef=%4d recall=%.4f 均值=% .0fus p95=%.0fus\n",
			p.Ef, p.Recall, p.LatUs, p.P95Us)
	}

	// 3) PQ 对比
	if usePQ {
		fmt.Println("==> 压缩前后召回对比")
		report.PQCompare = eval.PQCompare(idx, qs, ec)
		for k, v := range report.PQCompare {
			fmt.Printf("    %-10s recall=%.4f\n", k, v)
		}
	}

	// 4) 搜索路径样例
	report.PathSample = eval.CapturePath(idx, qs[0], bands[0].Cases[0].Filter, ec)

	// 5) 删除 sweep(会修改索引, 放最后)
	fmt.Println("==> 大量删除 + 局部修复后的召回变化")
	report.DeletionSweep = eval.DeletionSweep(idx, qs, bands, ec, 11)
	report.Stats = idx.Stats()

	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		panic(err)
	}
	// 原子写: 先写临时文件再 rename, 避免 serve 并发读到半截 JSON
	tmpPath := outPath + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		panic(err)
	}
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(report); err != nil {
		f.Close()
		os.Remove(tmpPath)
		panic(err)
	}
	if err := f.Close(); err != nil {
		os.Remove(tmpPath)
		panic(err)
	}
	if err := os.Rename(tmpPath, outPath); err != nil {
		panic(err)
	}
	fmt.Printf("==> 报告已写入 %s\n", outPath)
	fmt.Printf("    内存: HeapAlloc=%.1fMB 估计总量=%.1fMB (原始向量=%.1fMB, PQ编码=%.1fMB)\n",
		float64(report.Stats.HeapAllocBytes)/1e6,
		float64(report.Stats.TotalEstBytes)/1e6,
		float64(report.Stats.RawVectorBytes)/1e6,
		float64(report.Stats.PQCodesBytes)/1e6)
}

// MetricsJSON 与 eval.Metrics 结构一致, 仅用于规避同包类型别名。
type MetricsJSON = eval.Metrics

func toJSONMetrics(in map[string]map[string]*eval.Metrics) map[string]map[string]*MetricsJSON {
	out := make(map[string]map[string]*MetricsJSON, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func runServe(n, dim int, usePQ bool, ef, m, efc int, addr string) {
	idx, ds := buildIndex(n, dim, usePQ, m, efc)
	srv := server.New(idx, ds, ef)
	fmt.Printf("==> Web 服务启动: http://localhost%s\n", addr)
	if err := srv.Listen(addr); err != nil {
		panic(err)
	}
}
