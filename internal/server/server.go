// Package server 提供检索服务的 HTTP API 与可视化页面。
package server

import (
	"embed"
	"encoding/json"
	"io/fs"
	"net/http"
	"os"

	"vecsearch/internal/hnsw"
	"vecsearch/internal/index"
	"vecsearch/internal/sim"
	"vecsearch/internal/store"
)

//go:embed static
var staticFS embed.FS

// Server 检索服务。
type Server struct {
	idx *index.Index
	ds  *sim.Dataset
	ef  int
	mux *http.ServeMux
}

func New(idx *index.Index, ds *sim.Dataset, ef int) *Server {
	s := &Server{idx: idx, ds: ds, ef: ef, mux: http.NewServeMux()}
	s.routes()
	return s
}

func (s *Server) Listen(addr string) error {
	return http.ListenAndServe(addr, s.mux)
}

func (s *Server) routes() {
	sub, _ := fs.Sub(staticFS, "static")
	s.mux.Handle("GET /", http.FileServer(http.FS(sub)))
	s.mux.HandleFunc("GET /api/stats", s.handleStats)
	s.mux.HandleFunc("GET /api/filters", s.handleFilters)
	s.mux.HandleFunc("POST /api/search", s.handleSearch)
	s.mux.HandleFunc("GET /api/report", s.handleReport)
}

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, s.idx.Stats())
}

// FilterOption 可选过滤条件(取真实属性值)。
type FilterOption struct {
	Key         string  `json:"key"`
	Label       string  `json:"label"`
	Value       uint32  `json:"value"`
	Selectivity float64 `json:"selectivity"`
	Matches     int     `json:"matches"`
}

func (s *Server) handleFilters(w http.ResponseWriter, r *http.Request) {
	st := s.idx.Store()
	opts := []FilterOption{}
	add := func(key, label string, f *store.Filter, val uint32) {
		buf := make([]int32, 0, 4096)
		ids := st.Enumerate(f, buf)
		opts = append(opts, FilterOption{
			Key: key, Label: label, Value: val,
			Selectivity: float64(len(ids)) / float64(st.N), Matches: len(ids),
		})
	}
	// 取少量代表值
	for i := 0; i < s.ds.N; i += s.ds.N / 8 {
		region := s.ds.Region[i]
		add("region", "区域", &store.Filter{Region: &region}, uint32(region))
		cat := s.ds.Category[i]
		add("category", "类目", &store.Filter{Category: &cat}, uint32(cat))
		tn := s.ds.Tenant[i]
		add("tenant", "租户", &store.Filter{Tenant: &tn}, uint32(tn))
	}
	for _, q := range []float64{0.2, 0.5, 0.8} {
		mt := s.ds.T0 + uint32(q*float64(s.ds.TSpan))
		add("mintime", "时间下限", &store.Filter{MinTime: &mt}, mt)
	}
	// 多值标签(取前几条记录的首标签)与价格下限, 覆盖全部过滤能力
	labelSeen := map[uint16]bool{}
	for i := 0; i < s.ds.N && len(labelSeen) < 4; i += s.ds.N / 16 {
		if len(s.ds.Labels[i]) == 0 {
			continue
		}
		lab := s.ds.Labels[i][0]
		if labelSeen[lab] {
			continue
		}
		labelSeen[lab] = true
		add("label", "标签", &store.Filter{Label: &lab}, uint32(lab))
	}
	for _, p := range []float32{3, 6, 10} {
		mp := p
		add("minprice", "价格下限", &store.Filter{MinPrice: &mp}, uint32(p))
	}
	writeJSON(w, opts)
}

// SearchRequest 查询请求。
type SearchRequest struct {
	QueryID   int        `json:"query_id"` // 使用仿真查询集中的第几条; 缺省随机
	K         int        `json:"k"`
	Ef        int        `json:"ef"`
	Mode      string     `json:"mode"` // exact | hybrid_pq | full_pq
	TrackPath bool       `json:"track_path"`
	Filter    *FilterReq `json:"filter"`
}

type FilterReq struct {
	Tenant   *uint16  `json:"tenant"`
	Category *uint16  `json:"category"`
	Region   *uint8   `json:"region"`
	MinTime  *uint32  `json:"min_time"`
	MinPrice *float32 `json:"min_price"`
	Label    *uint16  `json:"label"`
}

func (s *Server) handleSearch(w http.ResponseWriter, r *http.Request) {
	var req SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if req.K == 0 {
		req.K = 10
	}
	if req.Ef == 0 {
		req.Ef = s.ef
	}
	qs := s.ds.GenQueries(req.QueryID+1, 2024)
	q := qs[req.QueryID]

	mode := index.ModeExact
	switch req.Mode {
	case "hybrid_pq":
		mode = index.ModeHybridPQ
	case "full_pq":
		mode = index.ModePQ
	}
	var f *store.Filter
	if req.Filter != nil {
		f = &store.Filter{
			Tenant: req.Filter.Tenant, Category: req.Filter.Category,
			Region: req.Filter.Region, MinTime: req.Filter.MinTime,
			MinPrice: req.Filter.MinPrice, Label: req.Filter.Label,
		}
	}
	res := s.idx.Search(q, f, index.SearchOptions{
		K: req.K, Ef: req.Ef, Mode: mode,
		TrackPath: req.TrackPath, RerankMul: 8,
	})
	// 附带命中节点的属性, 方便页面展示
	type hit struct {
		ID       int32   `json:"id"`
		Dist     float32 `json:"dist"`
		Tenant   uint16  `json:"tenant"`
		Category uint16  `json:"category"`
	}
	hits := make([]hit, len(res.IDs))
	st := s.idx.Store()
	for i, id := range res.IDs {
		hits[i] = hit{ID: id, Dist: res.Dists[i], Tenant: st.Tenant[id], Category: st.Category[id]}
	}
	writeJSON(w, map[string]any{
		"result": res, "hits": hits,
		"path": pathDTO(res.Path),
	})
}

func pathDTO(p []hnsw.PathEntry) []map[string]any {
	out := make([]map[string]any, len(p))
	for i, e := range p {
		out[i] = map[string]any{
			"node": e.Node, "layer": e.Layer,
			"dist": e.Dist, "filtered_out": e.FilteredOut,
		}
	}
	return out
}

func (s *Server) handleReport(w http.ResponseWriter, r *http.Request) {
	// 返回 bench 写出的报告; 缺失时给前端一个可识别的空响应而非 404
	if _, err := os.Stat("web/data/report.json"); err != nil {
		writeJSON(w, map[string]any{"available": false})
		return
	}
	http.ServeFile(w, r, "web/data/report.json")
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	_ = json.NewEncoder(w).Encode(v)
}
