// vecsvc is the command-line entry point for the vector retrieval service.
//
// Subcommands:
//
//	vecsvc bench [-n N] [-queries Q] [-pqm M] [-delete F] [-out report.json]
//	    Run the full acceptance benchmark and write a JSON report.
//
//	vecsvc serve [-n N] [-addr :8080] [-pqm M]
//	    Build an in-memory index over synthetic data and serve the dashboard.
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"runtime"
	"time"

	"vecsvc/internal/bench"
	"vecsvc/internal/hnsw"
	"vecsvc/internal/pq"
	"vecsvc/internal/server"
	"vecsvc/internal/store"
	"vecsvc/internal/synth"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "bench":
		runBench(os.Args[2:])
	case "serve":
		runServe(os.Args[2:])
	case "-h", "--help", "help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "unknown subcommand %q\n", os.Args[1])
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `vecsvc — HNSW vector retrieval service with PQ compression and filtered search

usage:
  vecsvc bench  [-n N] [-queries Q] [-ef "16,32,64,128,256"] [-pqm M]
                [-delete FRACTION] [-out report.json]
  vecsvc serve  [-n N] [-addr ADDR] [-pqm M] [-no-pq]
`)
}

func runBench(args []string) {
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	n := fs.Int("n", 200000, "number of synthetic vectors")
	queries := fs.Int("queries", 50, "queries per filter-selectivity case")
	pqm := fs.Int("pqm", 16, "PQ sub-quantizers (0 disables PQ)")
	pqSample := fs.Int("pqsample", 50000, "vectors sampled to train PQ")
	delFrac := fs.Float64("delete", 0.30, "fraction deleted in the delete-wave experiment")
	out := fs.String("out", "report.json", "output JSON report path")
	_ = fs.Parse(args)

	cfg := bench.DefaultConfig()
	cfg.N = *n
	cfg.QueriesPerCase = *queries
	cfg.PQM = *pqm
	cfg.PQSample = *pqSample
	cfg.DeleteFraction = *delFrac
	cfg.OutJSON = *out
	cfg.Workers = runtime.NumCPU()

	log.Printf("starting benchmark: n=%d dim=%d queries/case=%d pq_m=%d",
		cfg.N, cfg.Dim, cfg.QueriesPerCase, cfg.PQM)
	t := time.Now()
	rep, err := bench.Run(cfg)
	if err != nil {
		log.Fatalf("benchmark failed: %v", err)
	}
	printSummary(rep)
	log.Printf("report written to %s (total %s)", *out, time.Since(t).Round(time.Millisecond))
}

func runServe(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	n := fs.Int("n", 100000, "number of synthetic vectors to index")
	addr := fs.String("addr", ":8080", "listen address")
	pqm := fs.Int("pqm", 16, "PQ sub-quantizers (0 disables PQ)")
	_ = fs.Parse(args)

	dsCfg := synth.DefaultConfig(*n)
	log.Printf("generating %d synthetic %d-d vectors in %d clusters...", dsCfg.N, dsCfg.Dim, dsCfg.Clusters)
	t := time.Now()
	ds, err := synth.Generate(dsCfg)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("dataset ready in %s", time.Since(t).Round(time.Millisecond))

	st := store.New(ds, hnsw.DefaultConfig(), store.DefaultThresholds())
	rng := synth.NewPCG64(dsCfg.Seed, 777)
	dim := ds.Dim
	log.Printf("building HNSW index (M=%d, efConstruct=%d)...", 16, 128)
	t = time.Now()
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
		if (i+1)%50000 == 0 {
			log.Printf("  inserted %d/%d (%s)", i+1, ds.N, time.Since(t).Round(time.Millisecond))
		}
	}
	log.Printf("index built in %s, stats: %+v", time.Since(t).Round(time.Millisecond), st.Index().Stats())

	if *pqm > 0 {
		log.Printf("training PQ (m=%d) on %d sampled vectors...", *pqm, minN(*n, 100000))
		t = time.Now()
		model, err := pq.New(ds, *pqm, minN(*n, 100000), dsCfg.Seed+2, runtime.NumCPU())
		if err != nil {
			log.Fatal(err)
		}
		codes := model.EncodeAll(ds, runtime.NumCPU())
		st.SetPQ(model, codes)
		log.Printf("PQ ready in %s, compression %.1fx (raw %d bytes -> codes %d bytes)",
			time.Since(t).Round(time.Millisecond),
			float64(st.RawVectorBytes())/float64(st.PQCodeBytes()),
			st.RawVectorBytes(), st.PQCodeBytes())
	}

	srv := server.New(st, dsCfg)
	mux := http.NewServeMux()
	srv.Routes(mux)
	log.Printf("dashboard: http://localhost%s", *addr)
	log.Fatal(http.ListenAndServe(*addr, mux))
}

func printSummary(r *bench.Report) {
	fmt.Println("\n=== build ===")
	fmt.Printf("dataset gen: %dms | index build: %dms (%.0f inserts/s)\n",
		r.Build.DatasetGenMS, r.Build.IndexBuildMS, r.Build.InsertPerSec)
	fmt.Printf("heap: %s | raw vectors: %s", humanBytes(r.Build.HeapBytes), humanBytes(r.Build.RawVectorBytes))
	if r.PQ != nil {
		fmt.Printf(" | PQ codes: %s (%.1fx, train %dms, encode %dms)",
			humanBytes(r.Build.PQCodeBytes), r.PQ.Compression, r.PQ.TrainMS, r.PQ.EncodeMS)
	}
	fmt.Println()

	fmt.Println("\n=== recall@10 / latency by filter selectivity (ef=64) ===")
	fmt.Printf("%-22s %10s %22s %8s %10s %10s\n", "case", "select.", "strategy", "recall", "p50(us)", "p99(us)")
	for _, c := range r.Cases {
		fmt.Printf("%-22s %10.5f %22s %8.4f %10.1f %10.1f\n",
			c.Name, c.Selectivity, c.Strategy, c.Recall, c.P50LatencyUS, c.P99LatencyUS)
		if r.PQ != nil {
			fmt.Printf("%-22s %10s %22s %8.4f %10.1f %10s\n",
				"  └ PQ traversal", "", "", c.RecallPQ, c.P50LatencyPQUS, "")
		}
	}

	fmt.Println("\n=== ef sweep (unfiltered) ===")
	for _, e := range r.EfSweep {
		fmt.Printf("ef=%-4d recall=%.4f p50=%8.1fus mean_visited=%7.1f\n",
			e.Ef, e.Recall, e.P50LatencyUS, e.MeanVisited)
	}

	if r.Deletion != nil {
		d := r.Deletion
		fmt.Println("\n=== deletion wave ===")
		fmt.Printf("deleted %.1f%% in %dms | repair runs=%d | recall %.4f -> %.4f | p50 after %.1fus\n",
			d.DeletedFraction*100, d.DeleteMS, d.RepairRuns,
			d.RecallBefore, d.RecallAfter, d.P50LatencyAfterUS)
	}
}

func humanBytes(b uint64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%dB", b)
	}
	div, exp := uint64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f%ciB", float64(b)/float64(div), "KMGTPE"[exp])
}

func minN(a, b int) int {
	if a < b {
		return a
	}
	return b
}
