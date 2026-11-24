package pq_test

import (
	"fmt"
	"testing"

	"vecsvc/internal/hnsw"
	"vecsvc/internal/pq"
	"vecsvc/internal/synth"
)

func TestPQDistanceRanking(t *testing.T) {
	ds, _ := synth.Generate(synth.Config{
		N: 3000, Dim: 128, Clusters: 64,
		NTenants: 32, NCategories: 16, TSSpan: 100000, Seed: 42,
	})
	model, err := pq.New(ds, 16, 3000, 7, 4)
	if err != nil {
		t.Fatal(err)
	}
	codes := model.EncodeAll(ds, 4)
	fmt.Printf("PQ model: M=%d SubDim=%d K=%d codeLen=%d\n", model.M, model.SubDim, model.K, model.M)

	// For a handful of queries, compare exact nearest neighbors against
	// PQ-ADC ranking over the WHOLE dataset.
	const k = 10
	totalHit, total := 0, 0
	for qi := 0; qi < 20; qi++ {
		base := qi * 137 % ds.N
		query := ds.Vectors[base*ds.Dim : (base+1)*ds.Dim]

		// Exact top-k.
		type pair struct {
			id int32
			d  float32
		}
		exact := make([]pair, ds.N)
		for i := 0; i < ds.N; i++ {
			off := i * ds.Dim
			exact[i] = pair{int32(i), hnsw.L2Squared(query, ds.Vectors[off:off+ds.Dim])}
		}
		for i := 0; i < k; i++ {
			best := i
			for j := i + 1; j < len(exact); j++ {
				if exact[j].d < exact[best].d {
					best = j
				}
			}
			exact[i], exact[best] = exact[best], exact[i]
		}
		exactSet := map[int32]bool{}
		for i := 0; i < k; i++ {
			exactSet[exact[i].id] = true
		}

		// PQ ADC top-k over the whole dataset.
		adc := model.BuildADC(query)
		pqRanked := make([]pair, ds.N)
		for i := 0; i < ds.N; i++ {
			pqRanked[i] = pair{int32(i), adc.Dist(codes[i*model.M : (i+1)*model.M])}
		}
		for i := 0; i < k; i++ {
			best := i
			for j := i + 1; j < len(pqRanked); j++ {
				if pqRanked[j].d < pqRanked[best].d {
					best = j
				}
			}
			pqRanked[i], pqRanked[best] = pqRanked[best], pqRanked[i]
		}
		for i := 0; i < k; i++ {
			total++
			if exactSet[pqRanked[i].id] {
				totalHit++
			}
		}
	}
	recall := float64(totalHit) / float64(total)
	fmt.Printf("PQ full-scan recall@10 = %.3f (32x compression; ~0.4-0.6 is healthy)\n", recall)
	// At 512 float32 bytes -> 16 PQ bytes (32x), full-scan recall around
	// 0.4-0.6 is the expected operating point. We only fail on values that
	// indicate a real indexing/ranking defect, not ordinary quantization loss.
	if recall < 0.35 {
		t.Fatalf("PQ full-scan recall@10 = %.3f, quantizer is broken", recall)
	}
}
