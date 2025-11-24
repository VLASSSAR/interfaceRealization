package main

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"time"
)

/********** БАЗОВЫЕ ТИПЫ **********/

type Vector []float64
type Matrix [][]float64

func (v Vector) Copy() Vector { c := make(Vector, len(v)); copy(c, v); return c }
func Zeros(n int) Vector      { return make(Vector, n) }
func Eye(n int) Matrix {
	m := make(Matrix, n)
	for i := range m {
		m[i] = make([]float64, n)
		m[i][i] = 1
	}
	return m
}
func (v Vector) Add(u Vector) {
	for i := range v {
		v[i] += u[i]
	}
}
func (v Vector) Sub(u Vector) {
	for i := range v {
		v[i] -= u[i]
	}
}
func (v Vector) Scale(a float64) {
	for i := range v {
		v[i] *= a
	}
}
func Dot(a, b Vector) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
func MatTMat(m Matrix) Matrix { // M^T * M
	t := Transpose(m)
	return MatMul(t, m)
}
func MatVec(m Matrix, x Vector) Vector {
	y := make(Vector, len(m))
	for i := range m {
		s := 0.0
		for j := range m[i] {
			s += m[i][j] * x[j]
		}
		y[i] = s
	}
	return y
}
func MatMul(a, b Matrix) Matrix {
	n, k, m := len(a), len(a[0]), len(b[0])
	res := make(Matrix, n)
	for i := 0; i < n; i++ {
		res[i] = make([]float64, m)
		for j := 0; j < m; j++ {
			s := 0.0
			for t := 0; t < k; t++ {
				s += a[i][t] * b[t][j]
			}
			res[i][j] = s
		}
	}
	return res
}
func Transpose(a Matrix) Matrix {
	n, m := len(a), len(a[0])
	t := make(Matrix, m)
	for i := 0; i < m; i++ {
		t[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			t[i][j] = a[j][i]
		}
	}
	return t
}

// Решение A x = b (простой Гаусс с частичным выбором)
func Solve(A Matrix, b Vector) (Vector, error) {
	n := len(A)
	M := make(Matrix, n)
	for i := range M {
		M[i] = make([]float64, n+1)
		for j := 0; j < n; j++ {
			M[i][j] = A[i][j]
		}
		M[i][n] = b[i]
	}
	for i := 0; i < n; i++ {
		// pivot
		p := i
		for r := i + 1; r < n; r++ {
			if math.Abs(M[r][i]) > math.Abs(M[p][i]) {
				p = r
			}
		}
		if math.Abs(M[p][i]) < 1e-12 {
			return nil, errors.New("singular")
		}
		M[i], M[p] = M[p], M[i]
		// normalize row
		diag := M[i][i]
		for j := i; j <= n; j++ {
			M[i][j] /= diag
		}
		// eliminate
		for r := 0; r < n; r++ {
			if r == i {
				continue
			}
			f := M[r][i]
			for j := i; j <= n; j++ {
				M[r][j] -= f * M[i][j]
			}
		}
	}
	x := make(Vector, n)
	for i := 0; i < n; i++ {
		x[i] = M[i][n]
	}
	return x, nil
}

/********** ИНТЕРФЕЙСЫ **********/

// Functions
type ParametricFunction interface {
	Bind(params Vector) Function
}
type Function interface {
	Value(x Vector) float64
}
type DifferentiableFunction interface {
	Function
	// ∂f(x)/∂θ  (градиент по параметрам исходной ParametricFunction)
	Gradient(x Vector) Vector
}

// Functionals
type Functional interface {
	Value(f Function) float64
}
type DifferentiableFunctional interface {
	Functional
	// ∂J/∂θ для f, которая умеет ∂f/∂θ
	Gradient(f DifferentiableFunction) Vector
}
type LeastSquaresFunctional interface {
	Functional
	Residual(f Function) Vector
	Jacobian(f DifferentiableFunction) Matrix // ∂r_i/∂θ
}

// Оптимизатор
type Optimizator interface {
	Minimize(obj Functional, fn ParametricFunction, initial Vector, min, max Vector) Vector
}

/********** ФУНКЦИИ **********/

// 1) Линейная в R^n: f(x)=w·x + b, параметры θ=[w,b]
type LinearN struct {
	n int
}
type boundLinear struct{ theta Vector }       // [w..., b]
func (p LinearN) Bind(params Vector) Function { return &boundLinear{theta: params.Copy()} }
func (b *boundLinear) Value(x Vector) float64 {
	w := b.theta[:len(b.theta)-1]
	bias := b.theta[len(b.theta)-1]
	return Dot(w, x) + bias
}
func (b *boundLinear) Gradient(x Vector) Vector {
	g := make(Vector, len(b.theta))
	copy(g, x)
	g[len(g)-1] = 1.0 // ∂f/∂b
	return g
}

// 2) Полином степени n в R¹: f(x)=Σ a_k x^k (НЕ DifferentiableFunction по заданию)
type Poly1D struct {
	deg int
}
type boundPoly struct{ a Vector }            // a0..an
func (p Poly1D) Bind(params Vector) Function { return &boundPoly{a: params.Copy()} }
func (b *boundPoly) Value(x Vector) float64 {
	x1 := x[0]
	pwr := 1.0
	s := 0.0
	for _, ak := range b.a {
		s += ak * pwr
		pwr *= x1
	}
	return s
}

// 3) Кусочно-линейная 1D: узлы фиксированы, параметры — значения в узлах (DifferentiableFunction)
type PiecewiseLinear1D struct {
	knots Vector // фиксированные x-узлы
}
type boundPWL struct {
	knots Vector
	y     Vector // параметры (значения в узлах)
}

func (p PiecewiseLinear1D) Bind(params Vector) Function {
	// предполагаем, что len(params)==len(knots)
	return &boundPWL{knots: p.knots, y: params.Copy()}
}
func (b *boundPWL) Value(x Vector) float64 {
	xv := x[0]
	// найдём отрезок [xi, xi+1]
	i := 0
	for i < len(b.knots)-2 && xv > b.knots[i+1] {
		i++
	}
	x0, x1 := b.knots[i], b.knots[i+1]
	t := (xv - x0) / (x1 - x0)
	return (1-t)*b.y[i] + t*b.y[i+1]
}
func (b *boundPWL) Gradient(x Vector) Vector {
	xv := x[0]
	i := 0
	for i < len(b.knots)-2 && xv > b.knots[i+1] {
		i++
	}
	x0, x1 := b.knots[i], b.knots[i+1]
	t := (xv - x0) / (x1 - x0)
	g := make(Vector, len(b.y))
	g[i] = 1 - t
	g[i+1] = t
	return g
}

// 4) Кубический сплайн 1D (для простоты: натянутые отрезки; только Value)
type CubicSpline1D struct {
	knots Vector // x
	coef  Matrix // коэффициенты на каждом интервале [a,b]: c0+c1*h+c2*h^2+c3*h^3
}
type boundSpline struct{ sp CubicSpline1D }

func (s CubicSpline1D) Bind(params Vector) Function {
	// params — значения в узлах; коэффициенты считаем (натуральный сплайн)
	n := len(s.knots)
	y := params.Copy()
	// простая C2-аппроксимация (здесь — “not-a-knot” упрощённо)
	// Для краткости используем линейную интерполяцию и поднимаем до куба с нулевыми вторыми производными.
	coef := make(Matrix, n-1)
	for i := 0; i < n-1; i++ {
		h := s.knots[i+1] - s.knots[i]
		// Возьмём куб, совпадающий значениями и нулевыми втор.производными
		c0 := y[i]
		c1 := (y[i+1] - y[i]) / h
		c2 := 0.0
		c3 := 0.0
		coef[i] = Vector{c0, c1, c2, c3}
	}
	cp := s
	cp.coef = coef
	return &boundSpline{sp: cp}
}
func (b *boundSpline) Value(x Vector) float64 {
	xv := x[0]
	i := 0
	for i < len(b.sp.knots)-2 && xv > b.sp.knots[i+1] {
		i++
	}
	h := xv - b.sp.knots[i]
	c := b.sp.coef[i] // c0..c3
	return ((c[3]*h+c[2])*h+c[1])*h + c[0]
}

/********** ФУНКЦИОНАЛЫ **********/

// Набор точек (x_i, y_i)
type Samples struct {
	X []Vector
	Y Vector
}

// L1: sum |r_i|  (субградиент sign)
type L1Functional struct{ Data Samples }

func (f L1Functional) Value(fn Function) float64 {
	s := 0.0
	for i := range f.Data.X {
		r := fn.Value(f.Data.X[i]) - f.Data.Y[i]
		s += math.Abs(r)
	}
	return s
}
func (f L1Functional) Gradient(fn DifferentiableFunction) Vector {
	// θ-размер берём из градиента первого примера
	g := fn.Gradient(f.Data.X[0])
	grad := Zeros(len(g))
	for i := range f.Data.X {
		r := fn.Value(f.Data.X[i]) - f.Data.Y[i]
		sign := 0.0
		if r > 0 {
			sign = 1
		} else if r < 0 {
			sign = -1
		} // r==0 ⇒ 0 (субградиент)
		gi := fn.Gradient(f.Data.X[i])
		for j := range grad {
			grad[j] += sign * gi[j]
		}
	}
	return grad
}

// L2: sum r_i^2  + LS-атрибуты
type L2Functional struct{ Data Samples }

func (f L2Functional) Value(fn Function) float64 {
	s := 0.0
	for i := range f.Data.X {
		r := fn.Value(f.Data.X[i]) - f.Data.Y[i]
		s += r * r
	}
	return s
}
func (f L2Functional) Gradient(fn DifferentiableFunction) Vector {
	grad := Zeros(len(fn.Gradient(f.Data.X[0])))
	for i := range f.Data.X {
		r := fn.Value(f.Data.X[i]) - f.Data.Y[i]
		gi := fn.Gradient(f.Data.X[i])
		for j := range grad {
			grad[j] += 2 * r * gi[j]
		}
	}
	return grad
}
func (f L2Functional) Residual(fn Function) Vector {
	r := make(Vector, len(f.Data.X))
	for i := range f.Data.X {
		r[i] = fn.Value(f.Data.X[i]) - f.Data.Y[i]
	}
	return r
}
func (f L2Functional) Jacobian(fn DifferentiableFunction) Matrix {
	m := len(f.Data.X)
	p := len(fn.Gradient(f.Data.X[0]))
	J := make(Matrix, m)
	for i := 0; i < m; i++ {
		J[i] = fn.Gradient(f.Data.X[i])
		if len(J[i]) != p {
			panic("inconsistent gradient length")
		}
	}
	return J
}

// Linf: max |r_i|
type LinfFunctional struct{ Data Samples }

func (f LinfFunctional) Value(fn Function) float64 {
	mx := 0.0
	for i := range f.Data.X {
		r := math.Abs(fn.Value(f.Data.X[i]) - f.Data.Y[i])
		if r > mx {
			mx = r
		}
	}
	return mx
}

// Численный интеграл по [a,b] (1D), усреднённый (трапеции)
type IntegralFunctional struct {
	A, B float64
	N    int
}

func (f IntegralFunctional) Value(fn Function) float64 {
	if f.N <= 0 {
		f.N = 256
	}
	h := (f.B - f.A) / float64(f.N)
	sum := 0.0
	for i := 0; i <= f.N; i++ {
		x := f.A + float64(i)*h
		w := 1.0
		if i == 0 || i == f.N {
			w = 0.5
		}
		sum += w * fn.Value(Vector{x})
	}
	return sum * h
}

/********** ОПТИМИЗАТОРЫ **********/

// 1) Имитация отжига (универсальный)
type Annealing struct {
	Iter int
	T0   float64
	Step float64
}

func (a Annealing) Minimize(obj Functional, pf ParametricFunction, initial Vector, min, max Vector) Vector {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	theta := initial.Copy()
	best := theta.Copy()
	cur := obj.Value(pf.Bind(theta))
	bestVal := cur

	T := a.T0
	if T <= 0 {
		T = 1.0
	}
	step := a.Step
	if step <= 0 {
		step = 0.1
	}

	next := theta.Copy()
	n := len(theta)

	for it := 0; it < maxInt(a.Iter, 2000); it++ {
		// предложение
		copy(next, theta)
		for j := 0; j < n; j++ {
			next[j] += (rng.Float64()*2 - 1) * step
			if len(min) == n && next[j] < min[j] {
				next[j] = min[j]
			}
			if len(max) == n && next[j] > max[j] {
				next[j] = max[j]
			}
		}
		val := obj.Value(pf.Bind(next))
		accept := val < cur || rng.Float64() < math.Exp((cur-val)/T)
		if accept {
			theta, next = next, theta
			cur = val
			if cur < bestVal {
				bestVal = cur
				copy(best, theta)
			}
		}
		// охлаждение
		T *= 0.995
	}
	return best
}

// 2) Градиентный спуск (требует DifferentiableFunctional)
type GradDescent struct {
	Iters int
	LR    float64
}

func (gd GradDescent) Minimize(obj Functional, pf ParametricFunction, initial Vector, min, max Vector) Vector {
	df, ok := obj.(DifferentiableFunctional)
	if !ok {
		panic("GradDescent requires DifferentiableFunctional")
	}
	theta := initial.Copy()
	lr := gd.LR
	if lr <= 0 {
		lr = 1e-2
	}
	iters := maxInt(gd.Iters, 500)

	for i := 0; i < iters; i++ {
		// нужна DifferentiableFunction → пробуем связать с линейной/кусочно-линейной
		fn, ok := pf.Bind(theta).(DifferentiableFunction)
		if !ok {
			panic("bound function is not DifferentiableFunction")
		}
		g := df.Gradient(fn)
		// шаг
		for j := range theta {
			theta[j] -= lr * g[j]
			if len(min) == len(theta) && theta[j] < min[j] {
				theta[j] = min[j]
			}
			if len(max) == len(theta) && theta[j] > max[j] {
				theta[j] = max[j]
			}
		}
	}
	return theta
}

// 3) Гаусса-Ньютона (требует LeastSquaresFunctional)
type GaussNewton struct {
	Iters int
}

func (gn GaussNewton) Minimize(obj Functional, pf ParametricFunction, initial Vector, min, max Vector) Vector {
	ls, ok := obj.(LeastSquaresFunctional)
	if !ok {
		panic("Gauss-Newton requires ILeastSquaresFunctional")
	}
	theta := initial.Copy()
	iters := maxInt(gn.Iters, 30)

	for k := 0; k < iters; k++ {
		fn := pf.Bind(theta)
		df, ok := fn.(DifferentiableFunction)
		if !ok {
			panic("bound function is not DifferentiableFunction")
		}
		r := ls.Residual(fn) // m
		J := ls.Jacobian(df) // m x p
		JT := Transpose(J)   // p x m
		A := MatMul(JT, J)   // p x p
		b := MatVec(JT, r)   // p
		for i := range b {   // -J^T r
			b[i] = -b[i]
		}
		delta, err := Solve(A, b) // (J^T J) Δ = -J^T r
		if err != nil {
			break
		}
		for i := range theta {
			theta[i] += delta[i]
			if len(min) == len(theta) && theta[i] < min[i] {
				theta[i] = min[i]
			}
			if len(max) == len(theta) && theta[i] > max[i] {
				theta[i] = max[i]
			}
		}
		// простая остановка
		if math.Sqrt(Dot(delta, delta)) < 1e-8 {
			break
		}
	}
	return theta
}

/********** ПРИМЕР ИСПОЛЬЗОВАНИЯ **********/

func main() {
	// Данные: y ≈ 2*x1 - 0.5*x2 + 1
	data := Samples{
		X: []Vector{
			{0, 0}, {1, 0}, {0, 1}, {2, -1}, {3, 2},
		},
		Y: Vector{1, 3, 0.5, 2*2 - 0.5*(-1) + 1, 2*3 - 0.5*2 + 1},
	}

	// Линейная модель в R^2 → параметры: [w1, w2, b]
	fn := LinearN{n: 2}
	fL2 := L2Functional{Data: data}

	initial := Vector{0, 0, 0}
	// 1) Гаусса-Ньютона (для LS)
	thetaGN := (GaussNewton{Iters: 20}).Minimize(fL2, fn, initial, nil, nil)
	fmt.Printf("Gauss-Newton:  theta = %.5v,  L2 = %.6f\n", thetaGN, fL2.Value(fn.Bind(thetaGN)))

	// 2) Градиентный спуск
	thetaGD := (GradDescent{Iters: 2000, LR: 0.05}).Minimize(fL2, fn, initial, nil, nil)
	fmt.Printf("GradientDescent: theta = %.5v,  L2 = %.6f\n", thetaGD, fL2.Value(fn.Bind(thetaGD)))

	// 3) Отжиг (универсальный)
	thetaSA := (Annealing{Iter: 20000, T0: 1.0, Step: 0.2}).Minimize(fL2, fn, initial, nil, nil)
	fmt.Printf("Annealing:      theta = %.5v,  L2 = %.6f\n", thetaSA, fL2.Value(fn.Bind(thetaSA)))

	// Пример кусочно-линейной аппроксимации 1D
	knots := Vector{-1, 0, 1, 2}
	pwl := PiecewiseLinear1D{knots: knots}
	s1 := Samples{
		X: []Vector{{-1}, {0}, {0.5}, {1.7}},
		Y: Vector{-1, 0, 1, 2},
	}
	l1 := L1Functional{Data: s1}
	theta0 := Vector{0, 0, 0, 0}
	thetaPWL := (GradDescent{Iters: 1000, LR: 0.1}).Minimize(l1, pwl, theta0, nil, nil)
	fmt.Printf("PWL L1 fit:     y(knots)=%.3v,  L1=%.6f\n", thetaPWL, l1.Value(pwl.Bind(thetaPWL)))
}

/********** УТИЛЫ **********/
func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
