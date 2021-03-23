// WARNING: This file is generated. Do not edit.


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
 // IsEmulated

// Binary operation operator-
inline AVX512SimdReal operator- (AVX512SimdReal a,AVX512SimdReal b)
{
  return AVX512SimdReal(
     _mm512_sub_pd (a.v0,b.v0)
      );
}

inline AVX512SimdReal operator-(AVX512SimdReal a,Real b)
{
  return AVX512SimdReal(
    _mm512_sub_pd (a.v0,_mm512_set1_pd(b))
    );
}

inline AVX512SimdReal operator-(Real b,AVX512SimdReal a)
{
  return AVX512SimdReal(
    _mm512_sub_pd (_mm512_set1_pd(b), a.v0)
      );
}
// Binary operation operator+
inline AVX512SimdReal operator+ (AVX512SimdReal a,AVX512SimdReal b)
{
  return AVX512SimdReal(
     _mm512_add_pd (a.v0,b.v0)
      );
}

inline AVX512SimdReal operator+(AVX512SimdReal a,Real b)
{
  return AVX512SimdReal(
    _mm512_add_pd (a.v0,_mm512_set1_pd(b))
    );
}

inline AVX512SimdReal operator+(Real b,AVX512SimdReal a)
{
  return AVX512SimdReal(
    _mm512_add_pd (_mm512_set1_pd(b), a.v0)
      );
}
// Binary operation operator*
inline AVX512SimdReal operator* (AVX512SimdReal a,AVX512SimdReal b)
{
  return AVX512SimdReal(
     _mm512_mul_pd (a.v0,b.v0)
      );
}

inline AVX512SimdReal operator*(AVX512SimdReal a,Real b)
{
  return AVX512SimdReal(
    _mm512_mul_pd (a.v0,_mm512_set1_pd(b))
    );
}

inline AVX512SimdReal operator*(Real b,AVX512SimdReal a)
{
  return AVX512SimdReal(
    _mm512_mul_pd (_mm512_set1_pd(b), a.v0)
      );
}
// Binary operation operator/
inline AVX512SimdReal operator/ (AVX512SimdReal a,AVX512SimdReal b)
{
  return AVX512SimdReal(
     _mm512_div_pd (a.v0,b.v0)
      );
}

inline AVX512SimdReal operator/(AVX512SimdReal a,Real b)
{
  return AVX512SimdReal(
    _mm512_div_pd (a.v0,_mm512_set1_pd(b))
    );
}

inline AVX512SimdReal operator/(Real b,AVX512SimdReal a)
{
  return AVX512SimdReal(
    _mm512_div_pd (_mm512_set1_pd(b), a.v0)
      );
}
namespace math {
// Binary operation min
inline AVX512SimdReal min (AVX512SimdReal a,AVX512SimdReal b)
{
  return AVX512SimdReal(
     _mm512_min_pd (a.v0,b.v0)
      );
}

inline AVX512SimdReal min(AVX512SimdReal a,Real b)
{
  return AVX512SimdReal(
    _mm512_min_pd (a.v0,_mm512_set1_pd(b))
    );
}

inline AVX512SimdReal min(Real b,AVX512SimdReal a)
{
  return AVX512SimdReal(
    _mm512_min_pd (_mm512_set1_pd(b), a.v0)
      );
}
}
namespace math {
// Binary operation max
inline AVX512SimdReal max (AVX512SimdReal a,AVX512SimdReal b)
{
  return AVX512SimdReal(
     _mm512_max_pd (a.v0,b.v0)
      );
}

inline AVX512SimdReal max(AVX512SimdReal a,Real b)
{
  return AVX512SimdReal(
    _mm512_max_pd (a.v0,_mm512_set1_pd(b))
    );
}

inline AVX512SimdReal max(Real b,AVX512SimdReal a)
{
  return AVX512SimdReal(
    _mm512_max_pd (_mm512_set1_pd(b), a.v0)
      );
}
}

 // IsEmulated
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace math {
// Unary operation sqrt
inline AVX512SimdReal sqrt (AVX512SimdReal a)
{
  return AVX512SimdReal(_mm512_sqrt_pd(a.v0));
}
}

namespace math {
// Unary operation exp
inline AVX512SimdReal exp (AVX512SimdReal a)
{
  Real* za = (Real*)(&a);
  return AVX512SimdReal::fromScalar(math::exp(za[0]),math::exp(za[1]),math::exp(za[2]),math::exp(za[3]),math::exp(za[4]),math::exp(za[5]),math::exp(za[6]),math::exp(za[7]));
}
}

namespace math {
// Unary operation log10
inline AVX512SimdReal log10 (AVX512SimdReal a)
{
  Real* za = (Real*)(&a);
  return AVX512SimdReal::fromScalar(math::log10(za[0]),math::log10(za[1]),math::log10(za[2]),math::log10(za[3]),math::log10(za[4]),math::log10(za[5]),math::log10(za[6]),math::log10(za[7]));
}
}

