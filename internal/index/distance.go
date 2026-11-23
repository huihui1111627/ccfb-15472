package index

// 距离计算热路径。所有向量存放在同一连续底层数组 base 中,
// 每条定长 dim, 按偏移取数以便编译器消除边界检查并自动向量化。

// l2sqOff 计算 base 中偏移 offA 与 offB 两条定长向量的平方 L2 距离。
//
//go:nosplit
func l2sqOff(base []float32, dim int, offA, offB int) float32 {
	// 直接在同一底层数组上按索引访问; 末尾的 _ 访问向编译器证明
	// 最大偏移不越界, 从而消除循环内的边界检查。
	_ = base[offA+dim-1]
	_ = base[offB+dim-1]
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	for ; i+8 <= dim; i += 8 {
		d0 := base[offA+i] - base[offB+i]
		d1 := base[offA+i+1] - base[offB+i+1]
		d2 := base[offA+i+2] - base[offB+i+2]
		d3 := base[offA+i+3] - base[offB+i+3]
		d4 := base[offA+i+4] - base[offB+i+4]
		d5 := base[offA+i+5] - base[offB+i+5]
		d6 := base[offA+i+6] - base[offB+i+6]
		d7 := base[offA+i+7] - base[offB+i+7]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
		s4 += d4 * d4
		s5 += d5 * d5
		s6 += d6 * d6
		s7 += d7 * d7
	}
	var tail float32
	for ; i < dim; i++ {
		d := base[offA+i] - base[offB+i]
		tail += d * d
	}
	return (s0 + s1 + s2 + s3) + (s4 + s5 + s6 + s7) + tail
}

// l2sq 保留切片接口(查询向量是独立分配的), 内部走偏移版本。
func l2sq(a, b []float32) float32 {
	n := len(a)
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	i := 0
	for ; i+8 <= n; i += 8 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		d4 := a[i+4] - b[i+4]
		d5 := a[i+5] - b[i+5]
		d6 := a[i+6] - b[i+6]
		d7 := a[i+7] - b[i+7]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
		s4 += d4 * d4
		s5 += d5 * d5
		s6 += d6 * d6
		s7 += d7 * d7
	}
	var tail float32
	for ; i < n; i++ {
		d := a[i] - b[i]
		tail += d * d
	}
	return (s0 + s1 + s2 + s3) + (s4 + s5 + s6 + s7) + tail
}
