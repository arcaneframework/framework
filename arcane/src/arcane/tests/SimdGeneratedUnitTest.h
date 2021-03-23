// WARNING: This file is generated. Do not edit.



template<typename SimdRealType>
class SimdTestBinarySub
{
 public:
  static Real apply (Real a,Real b)
  {
    return (a - b);
  }

  static SimdRealType apply(SimdRealType a,SimdRealType b)
  {
    return (a - b);
  }

  static SimdRealType apply(SimdRealType a,Real b)
  {
    return (a - b);
  }

  static SimdRealType apply(Real a,SimdRealType b)
  {
    return (a - b);
  }
};


template<typename SimdRealType>
class SimdTestBinaryAdd
{
 public:
  static Real apply (Real a,Real b)
  {
    return (a + b);
  }

  static SimdRealType apply(SimdRealType a,SimdRealType b)
  {
    return (a + b);
  }

  static SimdRealType apply(SimdRealType a,Real b)
  {
    return (a + b);
  }

  static SimdRealType apply(Real a,SimdRealType b)
  {
    return (a + b);
  }
};


template<typename SimdRealType>
class SimdTestBinaryMul
{
 public:
  static Real apply (Real a,Real b)
  {
    return (a * b);
  }

  static SimdRealType apply(SimdRealType a,SimdRealType b)
  {
    return (a * b);
  }

  static SimdRealType apply(SimdRealType a,Real b)
  {
    return (a * b);
  }

  static SimdRealType apply(Real a,SimdRealType b)
  {
    return (a * b);
  }
};


template<typename SimdRealType>
class SimdTestBinaryDiv
{
 public:
  static Real apply (Real a,Real b)
  {
    return (a / b);
  }

  static SimdRealType apply(SimdRealType a,SimdRealType b)
  {
    return (a / b);
  }

  static SimdRealType apply(SimdRealType a,Real b)
  {
    return (a / b);
  }

  static SimdRealType apply(Real a,SimdRealType b)
  {
    return (a / b);
  }
};


template<typename SimdRealType>
class SimdTestBinaryMin
{
 public:
  static Real apply (Real a,Real b)
  {
    return (math::min(a,b));
  }

  static SimdRealType apply(SimdRealType a,SimdRealType b)
  {
    return (math::min(a,b));
  }

  static SimdRealType apply(SimdRealType a,Real b)
  {
    return (math::min(a,b));
  }

  static SimdRealType apply(Real a,SimdRealType b)
  {
    return (math::min(a,b));
  }
};


template<typename SimdRealType>
class SimdTestBinaryMax
{
 public:
  static Real apply (Real a,Real b)
  {
    return (math::max(a,b));
  }

  static SimdRealType apply(SimdRealType a,SimdRealType b)
  {
    return (math::max(a,b));
  }

  static SimdRealType apply(SimdRealType a,Real b)
  {
    return (math::max(a,b));
  }

  static SimdRealType apply(Real a,SimdRealType b)
  {
    return (math::max(a,b));
  }
};


template<typename SimdRealType>
inline void _doAllBinary(const SimdUnitTest& st,const SimdTestValue<SimdRealType>& simd_value)
{
  SimdBinaryOperatorTester< SimdTestBinarySub<SimdRealType>, SimdRealType > ::doTest(st,simd_value,"Sub");
  SimdBinaryOperatorTester< SimdTestBinaryAdd<SimdRealType>, SimdRealType > ::doTest(st,simd_value,"Add");
  SimdBinaryOperatorTester< SimdTestBinaryMul<SimdRealType>, SimdRealType > ::doTest(st,simd_value,"Mul");
  SimdBinaryOperatorTester< SimdTestBinaryDiv<SimdRealType>, SimdRealType > ::doTest(st,simd_value,"Div");
  SimdBinaryOperatorTester< SimdTestBinaryMin<SimdRealType>, SimdRealType > ::doTest(st,simd_value,"Min");
  SimdBinaryOperatorTester< SimdTestBinaryMax<SimdRealType>, SimdRealType > ::doTest(st,simd_value,"Max");
}




template<typename SimdRealType>
class SimdTestUnarySquareRoot
{
 public:
  static Real apply (Real a)
  {
    return math::sqrt(a);
  }

  static SimdRealType apply(SimdRealType a)
  {
    return math::sqrt(a);
  }

};


template<typename SimdRealType>
class SimdTestUnaryExponential
{
 public:
  static Real apply (Real a)
  {
    return math::exp(a);
  }

  static SimdRealType apply(SimdRealType a)
  {
    return math::exp(a);
  }

};


template<typename SimdRealType>
class SimdTestUnaryLog10
{
 public:
  static Real apply (Real a)
  {
    return math::log10(a);
  }

  static SimdRealType apply(SimdRealType a)
  {
    return math::log10(a);
  }

};


template<typename SimdRealType>
class SimdTestUnaryUnaryMinus
{
 public:
  static Real apply (Real a)
  {
    return -(a);
  }

  static SimdRealType apply(SimdRealType a)
  {
    return -(a);
  }

};


template<typename SimdRealType>
inline void _doAllUnary(const SimdUnitTest& st,const SimdTestValue<SimdRealType>& simd_value)
{
  SimdUnaryOperatorTester< SimdTestUnarySquareRoot<SimdRealType>, SimdRealType > ::doTest(st,simd_value,"SquareRoot");
  SimdUnaryOperatorTester< SimdTestUnaryExponential<SimdRealType>, SimdRealType > ::doTest(st,simd_value,"Exponential");
  SimdUnaryOperatorTester< SimdTestUnaryLog10<SimdRealType>, SimdRealType > ::doTest(st,simd_value,"Log10");
  SimdUnaryOperatorTester< SimdTestUnaryUnaryMinus<SimdRealType>, SimdRealType > ::doTest(st,simd_value,"UnaryMinus");
}