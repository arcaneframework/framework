// WARNING: This file is generated. Do not edit.


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
 // IsEmulated

// Binary operation operator-
inline AVXSimdReal operator- (AVXSimdReal a,AVXSimdReal b)
{
  return AVXSimdReal(
     _mm256_sub_pd (a.v0,b.v0)
      );
}

inline AVXSimdReal operator-(AVXSimdReal a,Real b)
{
  return AVXSimdReal(
    _mm256_sub_pd (a.v0,_mm256_set1_pd(b))
    );
}

inline AVXSimdReal operator-(Real b,AVXSimdReal a)
{
  return AVXSimdReal(
    _mm256_sub_pd (_mm256_set1_pd(b), a.v0)
      );
}
// Binary operation operator+
inline AVXSimdReal operator+ (AVXSimdReal a,AVXSimdReal b)
{
  return AVXSimdReal(
     _mm256_add_pd (a.v0,b.v0)
      );
}

inline AVXSimdReal operator+(AVXSimdReal a,Real b)
{
  return AVXSimdReal(
    _mm256_add_pd (a.v0,_mm256_set1_pd(b))
    );
}

inline AVXSimdReal operator+(Real b,AVXSimdReal a)
{
  return AVXSimdReal(
    _mm256_add_pd (_mm256_set1_pd(b), a.v0)
      );
}
// Binary operation operator*
inline AVXSimdReal operator* (AVXSimdReal a,AVXSimdReal b)
{
  return AVXSimdReal(
     _mm256_mul_pd (a.v0,b.v0)
      );
}

inline AVXSimdReal operator*(AVXSimdReal a,Real b)
{
  return AVXSimdReal(
    _mm256_mul_pd (a.v0,_mm256_set1_pd(b))
    );
}

inline AVXSimdReal operator*(Real b,AVXSimdReal a)
{
  return AVXSimdReal(
    _mm256_mul_pd (_mm256_set1_pd(b), a.v0)
      );
}
// Binary operation operator/
inline AVXSimdReal operator/ (AVXSimdReal a,AVXSimdReal b)
{
  return AVXSimdReal(
     _mm256_div_pd (a.v0,b.v0)
      );
}

inline AVXSimdReal operator/(AVXSimdReal a,Real b)
{
  return AVXSimdReal(
    _mm256_div_pd (a.v0,_mm256_set1_pd(b))
    );
}

inline AVXSimdReal operator/(Real b,AVXSimdReal a)
{
  return AVXSimdReal(
    _mm256_div_pd (_mm256_set1_pd(b), a.v0)
      );
}
namespace math {
// Binary operation min
inline AVXSimdReal min (AVXSimdReal a,AVXSimdReal b)
{
  return AVXSimdReal(
     _mm256_min_pd (a.v0,b.v0)
      );
}

inline AVXSimdReal min(AVXSimdReal a,Real b)
{
  return AVXSimdReal(
    _mm256_min_pd (a.v0,_mm256_set1_pd(b))
    );
}

inline AVXSimdReal min(Real b,AVXSimdReal a)
{
  return AVXSimdReal(
    _mm256_min_pd (_mm256_set1_pd(b), a.v0)
      );
}
}
namespace math {
// Binary operation max
inline AVXSimdReal max (AVXSimdReal a,AVXSimdReal b)
{
  return AVXSimdReal(
     _mm256_max_pd (a.v0,b.v0)
      );
}

inline AVXSimdReal max(AVXSimdReal a,Real b)
{
  return AVXSimdReal(
    _mm256_max_pd (a.v0,_mm256_set1_pd(b))
    );
}

inline AVXSimdReal max(Real b,AVXSimdReal a)
{
  return AVXSimdReal(
    _mm256_max_pd (_mm256_set1_pd(b), a.v0)
      );
}
}

 // IsEmulated
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace math {
// Unary operation sqrt
inline AVXSimdReal sqrt (AVXSimdReal a)
{
  return AVXSimdReal(_mm256_sqrt_pd(a.v0));
}
}

namespace math {
// Unary operation exp
inline AVXSimdReal exp (AVXSimdReal a)
{
  Real* za = (Real*)(&a);
  return AVXSimdReal::fromScalar(math::exp(za[0]),math::exp(za[1]),math::exp(za[2]),math::exp(za[3]));
}
}

namespace math {
// Unary operation log10
inline AVXSimdReal log10 (AVXSimdReal a)
{
  Real* za = (Real*)(&a);
  return AVXSimdReal::fromScalar(math::log10(za[0]),math::log10(za[1]),math::log10(za[2]),math::log10(za[3]));
}
}

