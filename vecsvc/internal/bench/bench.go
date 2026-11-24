// Package bench runs the acceptance benchmark: it builds the index over a
// synthetic dataset, computes brute-force ground truth, and measures recall
// and latency across filter selectivities, ef settings, exact vs PQ
// distances, and before/after a large delete wave.
package bench

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"runtime"
	"sort"
	"time"

	"vecsvc/internal/hnsw"
	"vecsvc/internal/pq"
	"vecsvc/internal/store"
	"vecsvc/internal/synth"
)

// Config controls a benchmark run.
type Config struct {
	N              int
	Dim            int
	Clusters       int
	K              int // nearest neighbors
	QueriesPerCase int
	Efs            []int
	PQM            int // PQ sub-quantizers (0 disables PQ)
	PQSample       int
	DeleteFraction float64
	Seed           uint64
	Workers        int
	OutJSON        string
}

// DefaultConfig returns a runnable-in-minutes configuration for a 16GB laptop.
func DefaultConfig() Config {
	return Config{
		N:              200000,
		Dim:            128,
		Clusters:       64,
		K:              10,
		QueriesPerCase: 50,
		Efs:            []int{16, 32, 64, 128, 256},
		PQM:            16,
		PQSample:       50000,
		DeleteFraction: 0.30,
		Seed:           42,
		Workers:        runtime.NumCPU(),
	}
}

// Report is the full JSON output consumed by the web dashboard.
type Report struct {
	GeneratedAt    string            `json:"generated_at"`
	Config         Config            `json:"config"`
	Build          BuildReport       `json:"build"`
	PQ             *PQReport         `json:"pq,omitempty"`
	Cases          []CaseReport      `json:"cases"`
	EfSweep        []EfPoint         `json:"ef_sweep"`
	Deletion       *DeletionReport   `json:"deletion,omitempty"`
	IndexStats     hnsw.Stats        `json:"index_stats"`
	StrategyByCase map[string]string `json:"strategy_by_case"`
}

// BuildReport captures index construction cost.
type BuildReport struct {
	DatasetGenMS   int64   `json:"dataset_gen_ms"`
	IndexBuildMS   int64   `json:"index_build_ms"`
	InsertPerSec   float64 `json:"insert_per_sec"`
	HeapBytes      uint64  `json:"heap_bytes"`
	RawVectorBytes uint64  `json:"raw_vector_bytes"`
	PQCodeBytes    uint64  `json:"pq_code_bytes,omitempty"`
}

// PQReport captures quantizer cost and quality.
type PQReport struct {
	TrainMS     int64   `json:"train_ms"`
	EncodeMS    int64   `json:"encode_ms"`
	CodeBytes   uint64  `json:"code_bytes"`
	Compression float64 `json:"compression_x"`
}

// CaseReport is one filter-selectivity case at the reference ef.
type CaseReport struct {
	Name           string  `json:"name"`
	Selectivity    float64 `json:"selectivity"`
	Strategy       string  `json:"strategy"`
	Recall         float64 `json:"recall"`
	P50LatencyUS   float64 `json:"p50_latency_us"`
	P99LatencyUS   float64 `json:"p99_latency_us"`
	MeanVisited    float64 `json:"mean_visited"`
	RecallPQ       float64 `json:"recall_pq,omitempty"`
	P50LatencyPQUS float64 `json:"p50_latency_pq_us,omitempty"`
}

// EfPoint traces the recall/latency trade-off of the bounded candidate
// queue on unfiltered queries.
type EfPoint struct {
	Ef           int     `json:"ef"`
	Recall       float64 `json:"recall"`
	P50LatencyUS float64 `json:"p50_latency_us"`
	MeanVisited  float64 `json:"mean_visited"`
}

// DeletionReport measures recall after a delete wave and graph repair.
type DeletionReport struct {
	DeletedFraction   float64 `json:"deleted_fraction"`
	DeleteMS          int64   `json:"delete_ms"`
	RepairRuns        int     `json:"repair_runs"`
	RecallBefore      float64 `json:"recall_before"`
	RecallAfter       float64 `json:"recall_after"`
	P50LatencyAfterUS float64 `json:"p50_latency_after_us"`
}

// Run executes the full benchmark and writes the JSON report.
func Run(cfg Config) (*Report, error) {
	if cfg.Workers <= 0 {
		cfg.Workers = runtime.NumCPU()
	}
	rep := &Report{Config: cfg, StrategyByCase: map[string]string{}}

	// 1. Dataset.
	t0 := time.Now()
	dsCfg := synth.DefaultConfig(cfg.N)
	dsCfg.Dim = cfg.Dim
	dsCfg.Clusters = cfg.Clusters
	dsCfg.Seed = cfg.Seed
	ds, err := synth.Generate(dsCfg)
	if err != nil {
		return nil, err
	}
	log.Printf("[bench] dataset generated in %s (n=%d dim=%d)", time.Since(t0).Round(time.Millisecond), ds.N, ds.Dim)
	rep.Build.DatasetGenMS = time.Since(t0).Milliseconds()

	// 2. Build the HNSW index.
	st := store.New(ds, hnsw.DefaultConfig(), store.DefaultThresholds())
	rng := synth.NewPCG64(cfg.Seed, 777)
	dim := ds.Dim
	t0 = time.Now()
	for i := 0; i < ds.N; i++ {
		query := ds.Vectors[i*dim : (i+1)*dim]
		distOf := func(cand int32) float32 {
			off := int(cand) * dim
			return hnsw.L2Squared(query, ds.Vectors[off:off+dim])
		}
		st.Index().SetEdgeDist(func(a, b int32) float32 {
			da, db := int(a)*dim, int(b)*dim
			return hnsw.L2Squared(ds.Vectors[da:da+dim], ds.Vectors[db:db+dim])
		})
		st.Index().Insert(int32(i), distOf, rng.Uint64)
	}
	log.Printf("[bench] HNSW built in %s", time.Since(t0).Round(time.Millisecond))
	rep.Build.IndexBuildMS = time.Since(t0).Milliseconds()
	rep.Build.InsertPerSec = float64(ds.N) / (float64(rep.Build.IndexBuildMS) / 1000)
	rep.Build.HeapBytes = st.MemUsageBytes()
	rep.Build.RawVectorBytes = st.RawVectorBytes()

	// 3. Queries and ground truth.
	querySets, err := synth.GenerateQuerySets(ds, cfg.QueriesPerCase, cfg.Seed+1)
	if err != nil {
		return nil, err
	}
	allQueries := flatten(querySets)
	t0 = time.Now()
	gt := st.GroundTruth(allQueries, cfg.K, cfg.Workers)
	log.Printf("[bench] ground truth for %d queries in %s", len(allQueries), time.Since(t0).Round(time.Millisecond))

	// 4. Optional PQ training and encoding.
	if cfg.PQM > 0 {
		t0 = time.Now()
		model, err := pq.New(ds, cfg.PQM, minInt(cfg.PQSample, cfg.N), cfg.Seed+2, cfg.Workers)
		if err != nil {
			return nil, err
		}
		trainMS := time.Since(t0).Milliseconds()
		log.Printf("[bench] PQ trained (m=%d, sample=%d) in %dms", cfg.PQM, minInt(cfg.PQSample, cfg.N), trainMS)
		t0 = time.Now()
		codes := model.EncodeAll(ds, cfg.Workers)
		encodeMS := time.Since(t0).Milliseconds()
		log.Printf("[bench] PQ encoded all %d vectors in %dms", ds.N, encodeMS)
		st.SetPQ(model, codes)
		rep.PQ = &PQReport{
			TrainMS: trainMS, EncodeMS: encodeMS,
			CodeBytes:   st.PQCodeBytes(),
			Compression: float64(st.RawVectorBytes()) / float64(st.PQCodeBytes()),
		}
		rep.Build.PQCodeBytes = st.PQCodeBytes()
	}

	// 5. Per-selectivity cases (reference ef = 64, exact and PQ).
	log.Printf("[bench] measuring %d filter-selectivity cases...", len(querySets))
	refEf := 64
	offset := 0
	for _, qs := range querySets {
		caseGT := gt[offset : offset+len(qs.Queries)]
		offset += len(qs.Queries)

		exact := measureCase(st, qs, caseGT, cfg.K, refEf, false)
		exact.Name = qs.Name
		exact.Selectivity = qs.ActualSel
		rep.StrategyByCase[qs.Name] = exact.Strategy
		if rep.PQ != nil {
			pqRes := measureCase(st, qs, caseGT, cfg.K, refEf, true)
			exact.RecallPQ = pqRes.Recall
			exact.P50LatencyPQUS = pqRes.P50LatencyUS
		}
		rep.Cases = append(rep.Cases, exact)
	}

	// 6. ef sweep on unfiltered queries.
	unfiltered := querySets[0]
	ufGT := gt[0:len(unfiltered.Queries)]
	for _, ef := range cfg.Efs {
		c := measureCase(st, unfiltered, ufGT, cfg.K, ef, false)
		rep.EfSweep = append(rep.EfSweep, EfPoint{
			Ef: ef, Recall: c.Recall,
			P50LatencyUS: c.P50LatencyUS, MeanVisited: c.MeanVisited,
		})
	}

	rep.IndexStats = st.Index().Stats()

	// 7. Delete wave and recall re-measurement.
	if cfg.DeleteFraction > 0 {
		log.Printf("[bench] running delete wave (fraction=%.2f)...", cfg.DeleteFraction)
		rep.Deletion = runDeletion(st, ds, querySets, gt, cfg, rng)
		log.Printf("[bench] delete wave done: repair runs=%d", rep.Deletion.RepairRuns)
		rep.IndexStats = st.Index().Stats()
	}

	rep.GeneratedAt = time.Now().Format(time.RFC3339)
	if cfg.OutJSON != "" {
		if err := writeJSON(cfg.OutJSON, rep); err != nil {
			return nil, err
		}
	}
	return rep, nil
}

func flatten(sets []*synth.QuerySet) []*synth.Query {
	var out []*synth.Query
	for _, qs := range sets {
		out = append(out, qs.Queries...)
	}
	return out
}

func measureCase(st *store.Store, qs *synth.QuerySet, gt [][]int32,
	k, ef int, usePQ bool) CaseReport {

	latUS := make([]float64, 0, len(qs.Queries))
	visited := 0
	hits := 0
	denom := 0
	strategy := ""
	for i, q := range qs.Queries {
		t := time.Now()
		res := st.FilteredSearch(q.Vector, q.Filter, qs.ActualSel, store.FilteredSearchOptions{
			K: k, Ef: ef, UsePQ: usePQ,
		})
		latUS = append(latUS, float64(time.Since(t).Microseconds()))
		visited += res.Visited + res.BruteScanned
		strategy = res.Strategy
		hits += overlap(res.IDs, gt[i])
		denom += len(gt[i])
	}
	sort.Float64s(latUS)
	return CaseReport{
		Strategy:     strategy,
		Recall:       safeDiv(hits, denom),
		P50LatencyUS: percentile(latUS, 50),
		P99LatencyUS: percentile(latUS, 99),
		MeanVisited:  safeDiv(visited, len(qs.Queries)),
	}
}

func runDeletion(st *store.Store, ds *synth.Dataset, sets []*synth.QuerySet,
	gt [][]int32, cfg Config, rng *synth.PCG64) *DeletionReport {

	// Recall before deletion on the middle-selectivity case (most sensitive
	// to graph damage).
	mid := middleSet(sets)
	midGT := groundTruthFor(sets, gt, mid.Name)
	before := measureCase(st, mid, midGT, cfg.K, 64, false)

	nDelete := int(float64(ds.N) * cfg.DeleteFraction)
	t0 := time.Now()
	// One full shuffle of a compact index array, then take the first
	// nDelete ids. A real slice (instead of a lazily-swapped map) makes
	// every sampled id unique by construction, so every Delete succeeds
	// and no iteration is wasted retrying tombstoned nodes.
	order := make([]int32, ds.N)
	for i := range order {
		order[i] = int32(i)
	}
	for i := ds.N - 1; i > 0; i-- {
		j := rng.Intn(i + 1)
		order[i], order[j] = order[j], order[i]
	}
	deleted := 0
	for ; deleted < nDelete; deleted++ {
		st.Delete(order[deleted], rng.Uint64)
	}
	deleteMS := time.Since(t0).Milliseconds()

	// Recompute ground truth after deletes for a fair comparison.
	newGT := st.GroundTruth(mid.Queries, cfg.K, cfg.Workers)
	after := measureCase(st, mid, newGT, cfg.K, 64, false)

	return &DeletionReport{
		// Use the actual number removed over the original size; the graph's
		// live DeletedFraction resets after each compaction rebuild.
		DeletedFraction:   float64(nDelete) / float64(ds.N),
		DeleteMS:          deleteMS,
		RepairRuns:        st.Index().RepairRuns(),
		RecallBefore:      before.Recall,
		RecallAfter:       after.Recall,
		P50LatencyAfterUS: after.P50LatencyUS,
	}
}

func middleSet(sets []*synth.QuerySet) *synth.QuerySet {
	for _, qs := range sets {
		if qs.Name == "cat_and_ts_mid" {
			return qs
		}
	}
	return sets[len(sets)/2]
}

func groundTruthFor(sets []*synth.QuerySet, gt [][]int32, name string) [][]int32 {
	off := 0
	for _, qs := range sets {
		if qs.Name == name {
			return gt[off : off+len(qs.Queries)]
		}
		off += len(qs.Queries)
	}
	return nil
}

func overlap(got, want []int32) int {
	set := make(map[int32]struct{}, len(got))
	for _, id := range got {
		set[id] = struct{}{}
	}
	n := 0
	for _, id := range want {
		if _, ok := set[id]; ok {
			n++
		}
	}
	return n
}

func safeDiv(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return float64(a) / float64(b)
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(float64(len(sorted)) * p / 100)
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func writeJSON(path string, rep *Report) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("write report: %w", err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(rep)
}
