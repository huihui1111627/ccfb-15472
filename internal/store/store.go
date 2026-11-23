// Package store 是向量与属性的列式存储,负责:
//   - 原始向量与结构化属性的紧凑存放
//   - 删除位图(软删除)
//   - 低基数列的倒排表,供严格过滤时直接枚举候选
//   - 过滤条件、选择率估计与候选枚举
package store

import (
	"math"
	"sort"
)

// Filter 结构化过滤条件,nil 字段表示不约束。
type Filter struct {
	Tenant   *uint16
	Category *uint16
	Region   *uint8
	MinTime  *uint32  // timestamp >= MinTime
	MinPrice *float32 // price >= MinPrice
	Label    *uint16  // 标签集合包含该标签
}

// Attrs 一条记录的结构化属性(更新时使用)。
type Attrs struct {
	Tenant    uint16
	Category  uint16
	Timestamp uint32
	Region    uint8
	Price     float32
	Labels    []uint16
}

// Store 列式存储。向量由上层(index)以 []float32 大数组持有,
// 这里持有属性列、删除位图与倒排表。
type Store struct {
	N, Dim int

	Tenant    []uint16
	Category  []uint16
	Timestamp []uint32
	Region    []uint8
	Price     []float32
	Labels    [][]uint16

	deleted []bool // 软删除位图

	// 倒排表(按 id 升序, 构建时按 id 顺序 append 天然有序)
	tenantPosts [][]int32
	catPosts    [][]int32
	regionPosts [][]int32
	labelPosts  [][]int32

	// 数值列分位数,用于选择率估计
	timeQuantiles []float32
	priceQuant    []float32
}

// New 构建空存储(属性列由调用方填充,或用 FromDataset)。
func New(n, dim, tenants, cats, regions, labels int) *Store {
	return &Store{
		N: n, Dim: dim,
		Tenant: make([]uint16, n), Category: make([]uint16, n),
		Timestamp: make([]uint32, n), Region: make([]uint8, n),
		Price:       make([]float32, n),
		Labels:      make([][]uint16, n),
		deleted:     make([]bool, n),
		tenantPosts: make([][]int32, tenants), catPosts: make([][]int32, cats),
		regionPosts: make([][]int32, regions), labelPosts: make([][]int32, labels),
	}
}

// BuildPostings 在属性列填充完成后构建倒排表与分位数。
func (s *Store) BuildPostings() {
	for i := 0; i < s.N; i++ {
		id := int32(i)
		s.tenantPosts[s.Tenant[i]] = append(s.tenantPosts[s.Tenant[i]], id)
		s.catPosts[s.Category[i]] = append(s.catPosts[s.Category[i]], id)
		s.regionPosts[s.Region[i]] = append(s.regionPosts[s.Region[i]], id)
		for _, l := range s.Labels[i] {
			s.labelPosts[l] = append(s.labelPosts[l], id)
		}
	}
	s.timeQuantiles = quantiles32(s.Timestamp, 20)
	prices := make([]float32, s.N)
	copy(prices, s.Price)
	s.priceQuant = quantilesFloat(prices, 20)
}

// Match 判断一条记录是否满足全部过滤条件(不含删除判断)。
func (s *Store) Match(i int32, f *Filter) bool {
	if f == nil {
		return true
	}
	if f.Tenant != nil && s.Tenant[i] != *f.Tenant {
		return false
	}
	if f.Category != nil && s.Category[i] != *f.Category {
		return false
	}
	if f.Region != nil && s.Region[i] != *f.Region {
		return false
	}
	if f.MinTime != nil && s.Timestamp[i] < *f.MinTime {
		return false
	}
	if f.MinPrice != nil && s.Price[i] < *f.MinPrice {
		return false
	}
	if f.Label != nil {
		hit := false
		for _, l := range s.Labels[i] {
			if l == *f.Label {
				hit = true
				break
			}
		}
		if !hit {
			return false
		}
	}
	return true
}

// Deleted / SetDeleted 软删除接口。
func (s *Store) Deleted(i int32) bool { return s.deleted[i] }

func (s *Store) SetDeleted(i int32, v bool) { s.deleted[i] = v }

// AliveCount 返回未删除记录数。
func (s *Store) AliveCount() int {
	c := 0
	for _, d := range s.deleted {
		if !d {
			c++
		}
	}
	return c
}

// DeletedCount 返回已删除记录数。
func (s *Store) DeletedCount() int {
	c := 0
	for _, d := range s.deleted {
		if d {
			c++
		}
	}
	return c
}

// Selectivity 估计过滤条件的选择率(0,1],基于倒排表精确基数 +
// 数值列分位数,假设各条件近似独立。
func (s *Store) Selectivity(f *Filter) float64 {
	if f == nil {
		return 1
	}
	p := 1.0
	if f.Tenant != nil {
		p *= float64(len(s.tenantPosts[*f.Tenant])) / float64(s.N)
	}
	if f.Category != nil {
		p *= float64(len(s.catPosts[*f.Category])) / float64(s.N)
	}
	if f.Region != nil {
		p *= float64(len(s.regionPosts[*f.Region])) / float64(s.N)
	}
	if f.Label != nil {
		p *= float64(len(s.labelPosts[*f.Label])) / float64(s.N)
	}
	if f.MinTime != nil {
		p *= rangeSelectivity32(s.timeQuantiles, float32(*f.MinTime))
	}
	if f.MinPrice != nil {
		p *= rangeSelectivity32(s.priceQuant, *f.MinPrice)
	}
	return math.Max(p, 1e-9)
}

// Enumerate 返回满足过滤条件且未删除的候选 id(调用方不得修改)。
// 有等值条件时从最小倒排表出发做有序交集,否则全表枚举。
// 用于严格过滤下的暴力检索与策略基准。
func (s *Store) Enumerate(f *Filter, buf []int32) []int32 {
	buf = buf[:0]
	var lists [][]int32
	if f != nil {
		if f.Tenant != nil {
			lists = append(lists, s.tenantPosts[*f.Tenant])
		}
		if f.Category != nil {
			lists = append(lists, s.catPosts[*f.Category])
		}
		if f.Region != nil {
			lists = append(lists, s.regionPosts[*f.Region])
		}
		if f.Label != nil {
			lists = append(lists, s.labelPosts[*f.Label])
		}
	}
	if len(lists) == 0 {
		for i := int32(0); i < int32(s.N); i++ {
			if !s.deleted[i] && s.Match(i, f) {
				buf = append(buf, i)
			}
		}
		return buf
	}
	// 最小表驱动交集
	sort.Slice(lists, func(a, b int) bool { return len(lists[a]) < len(lists[b]) })
	cand := lists[0]
	for _, l := range lists[1:] {
		cand = intersectSorted(cand, l)
	}
	for _, id := range cand {
		if s.deleted[id] {
			continue
		}
		if s.Match(id, f) { // 数值范围等剩余条件
			buf = append(buf, id)
		}
	}
	return buf
}

func intersectSorted(a, b []int32) []int32 {
	out := make([]int32, 0, min(len(a), len(b)))
	i, j := 0, 0
	for i < len(a) && j < len(b) {
		switch {
		case a[i] < b[j]:
			i++
		case a[i] > b[j]:
			j++
		default:
			out = append(out, a[i])
			i++
			j++
		}
	}
	return out
}

// quantiles32 等频分位数(抽样排序, q 段)。
func quantiles32(x []uint32, q int) []float32 {
	sample := min(len(x), 200_000)
	v := make([]uint32, sample)
	step := len(x) / sample
	for i := 0; i < sample; i++ {
		v[i] = x[i*step]
	}
	sort.Slice(v, func(a, b int) bool { return v[a] < v[b] })
	out := make([]float32, q+1)
	for k := 0; k <= q; k++ {
		out[k] = float32(v[min(k*sample/q, sample-1)])
	}
	return out
}

func quantilesFloat(x []float32, q int) []float32 {
	sample := min(len(x), 200_000)
	step := len(x) / sample
	v := make([]float32, sample)
	for i := 0; i < sample; i++ {
		v[i] = x[i*step]
	}
	sort.Sort(float64s(v))
	out := make([]float32, q+1)
	for k := 0; k <= q; k++ {
		out[k] = v[min(k*sample/q, sample-1)]
	}
	return out
}

// float64s 辅助: 把 []float32 当 []float64 排序不可行, 直接实现。
type float64s []float32

func (f float64s) Len() int           { return len(f) }
func (f float64s) Less(i, j int) bool { return f[i] < f[j] }
func (f float64s) Swap(i, j int)      { f[i], f[j] = f[j], f[i] }

// rangeSelectivity32 基于等频分位数估计 P(x >= t)。
func rangeSelectivity32(q []float32, t float32) float64 {
	for k := 0; k < len(q)-1; k++ {
		if t <= q[k] {
			return 1 - float64(k)/float64(len(q)-1)
		}
		if t < q[k+1] {
			frac := (t - q[k]) / max(q[k+1]-q[k], 1e-9)
			return 1 - (float64(k)+float64(frac))/float64(len(q)-1)
		}
	}
	return 1.0 / float64(len(q)-1)
}
