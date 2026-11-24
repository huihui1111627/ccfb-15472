// Package server exposes the vector store over HTTP and serves the
// observability dashboard (recall/latency curves, search-path trace,
// strategy selection).
package server

import (
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
	"strconv"
	"time"

	"vecsvc/internal/bench"
	"vecsvc/internal/store"
	"vecsvc/internal/synth"
)

//go:embed web/*
var webFS embed.FS

// Server holds shared service state.
type Server struct {
	store  *store.Store
	report *bench.Report
	dsCfg  synth.Config
}

// New wires the HTTP handlers.
func New(st *store.Store, dsCfg synth.Config) *Server {
	return &Server{store: st, dsCfg: dsCfg}
}

// Routes registers every endpoint on mux.
func (s *Server) Routes(mux *http.ServeMux) {
	mux.HandleFunc("GET /api/health", s.health)
	mux.HandleFunc("GET /api/stats", s.stats)
	mux.HandleFunc("POST /api/search", s.search)
	mux.HandleFunc("GET /api/attrs", s.attrs)
	mux.HandleFunc("POST /api/bench/run", s.runBench)
	mux.HandleFunc("GET /api/report", s.getReport)
	mux.HandleFunc("GET /api/filters/cases", s.filterCases)

	static, err := fs.Sub(webFS, "web")
	if err != nil {
		panic(err)
	}
	mux.Handle("GET /", http.FileServerFS(static))
}

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "ts": time.Now().Unix()})
}

func (s *Server) stats(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"index":            s.store.Index().Stats(),
		"mem_heap_bytes":   s.store.MemUsageBytes(),
		"raw_vector_bytes": s.store.RawVectorBytes(),
		"pq_code_bytes":    s.store.PQCodeBytes(),
		"has_pq":           s.store.HasPQ(),
	})
}

// SearchRequest is the JSON body of POST /api/search.
type SearchRequest struct {
	K          int               `json:"k"`
	Ef         int               `json:"ef"`
	UsePQ      bool              `json:"use_pq"`
	RecordPath bool              `json:"record_path"`
	Filter     *synth.FilterSpec `json:"filter"`
	// QuerySeed optionally generates a deterministic jittered query around
	// a dataset vector; when zero, QueryVector must be supplied.
	QuerySeed   int64     `json:"query_seed"`
	QueryVector []float32 `json:"query_vector"`
}

func (s *Server) search(w http.ResponseWriter, r *http.Request) {
	var req SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	if req.K <= 0 {
		req.K = 10
	}
	if req.Ef <= 0 {
		req.Ef = 64
	}
	query := req.QueryVector
	if len(query) == 0 {
		query = s.makeQuery(req.QuerySeed)
	}
	sel := 1.0
	if req.Filter != nil {
		sel = req.Filter.Selectivity(s.store.Dataset())
	}
	res := s.store.FilteredSearch(query, req.Filter, sel, store.FilteredSearchOptions{
		K: req.K, Ef: req.Ef, UsePQ: req.UsePQ, RecordPath: req.RecordPath,
	})
	writeJSON(w, http.StatusOK, map[string]any{
		"result":          res,
		"selectivity":     sel,
		"strategy_reason": explainStrategy(sel, s.store),
		"vector_dim":      len(query),
	})
}

// makeQuery produces a jittered query from a seed, mirroring
// synth.GenerateQuerySets so the dashboard can trace arbitrary queries.
func (s *Server) makeQuery(seed int64) []float32 {
	ds := s.store.Dataset()
	if seed == 0 {
		seed = 1
	}
	rng := synth.NewPCG64(uint64(seed), 0x1234abcd)
	base := rng.Intn(ds.N)
	v := make([]float32, ds.Dim)
	for j := range v {
		v[j] = ds.Vectors[base*ds.Dim+j] + float32(rng.Gaussian()*0.15)
	}
	return v
}

func explainStrategy(sel float64, st *store.Store) string {
	thr := store.DefaultThresholds()
	switch {
	case sel >= thr.Wide:
		return fmt.Sprintf("selectivity %.4f >= wide threshold %.3f: relaxed graph walk — non-matching nodes are traversed as routing nodes but excluded from results", sel, thr.Wide)
	case sel <= thr.Strict:
		return fmt.Sprintf("selectivity %.4f <= strict threshold %.4f: brute force over the matching subset (posting lists where available)", sel, thr.Strict)
	default:
		return fmt.Sprintf("selectivity %.4f is in the middle band (%.4f, %.3f): relaxed graph search first, brute-force fallback if fewer than k matches are reached", sel, thr.Strict, thr.Wide)
	}
}

// BenchRequest triggers a benchmark run.
type BenchRequest struct {
	N              int     `json:"n"`
	QueriesPerCase int     `json:"queries_per_case"`
	PQM            int     `json:"pq_m"`
	DeleteFraction float64 `json:"delete_fraction"`
}

func (s *Server) runBench(w http.ResponseWriter, r *http.Request) {
	var req BenchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	cfg := bench.DefaultConfig()
	if req.N > 0 {
		cfg.N = req.N
	}
	if req.QueriesPerCase > 0 {
		cfg.QueriesPerCase = req.QueriesPerCase
	}
	if req.PQM != 0 {
		cfg.PQM = req.PQM
	}
	if req.DeleteFraction > 0 {
		cfg.DeleteFraction = req.DeleteFraction
	}
	cfg.OutJSON = "" // keep report in memory; dashboard fetches /api/report
	rep, err := bench.Run(cfg)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	s.report = rep
	writeJSON(w, http.StatusOK, rep)
}

func (s *Server) getReport(w http.ResponseWriter, r *http.Request) {
	if s.report == nil {
		writeErr(w, http.StatusNotFound, fmt.Errorf("no benchmark report yet — run POST /api/bench/run or the CLI benchmark"))
		return
	}
	writeJSON(w, http.StatusOK, s.report)
}

// attrs returns attributes for a small set of result ids so the dashboard
// can annotate query results without shipping the whole attribute arrays.
func (s *Server) attrs(w http.ResponseWriter, r *http.Request) {
	idStr := r.URL.Query()["id"]
	ds := s.store.Dataset()
	out := make([]map[string]any, 0, len(idStr))
	for _, raw := range idStr {
		id, err := strconv.Atoi(raw)
		if err != nil || id < 0 || id >= ds.N {
			continue
		}
		out = append(out, map[string]any{
			"id": id, "tenant": ds.Tenant[id], "category": ds.Category[id],
			"ts": ds.TS[id], "price": ds.Price[id], "cluster": ds.Cluster[id],
			"deleted": s.store.Index().IsDeleted(int32(id)),
		})
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) filterCases(w http.ResponseWriter, r *http.Request) {
	sets, err := synth.GenerateQuerySets(s.store.Dataset(), 1, 99)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	out := make([]map[string]any, 0, len(sets))
	for _, qs := range sets {
		out = append(out, map[string]any{
			"name": qs.Name, "target": qs.Target,
			"actual_selectivity": qs.ActualSel, "filter": qs.Filter,
		})
	}
	writeJSON(w, http.StatusOK, out)
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

func writeErr(w http.ResponseWriter, code int, err error) {
	writeJSON(w, code, map[string]string{"error": err.Error()})
}
