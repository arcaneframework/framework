// WARNING: This file is generated. Do not edit.


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
 // IsEmulated

// Binary operation operator-
inline SSESimdReal operator- (SSESimdReal a,SSESimdReal b)
{
  return SSESimdReal(
     _mm_sub_pd (a.v0,b.v0)
    ,   _mm_sub_pd (a.v1,b.v1)
      );
}

inline SSESimdReal operator-(SSESimdReal a,Real b)
{
  return SSESimdReal(
    _mm_sub_pd (a.v0,_mm_set1_pd(b))
   ,  _mm_sub_pd (a.v1,_mm_set1_pd(b))
    );
}

inline SSESimdReal operator-(Real b,SSESimdReal a)
{
  return SSESimdReal(
    _mm_sub_pd (_mm_set1_pd(b), a.v0)
    ,  _mm_sub_pd (_mm_set1_pd(b), a.v1)
      );
}
// Binary operation operator+
inline SSESimdReal operator+ (SSESimdReal a,SSESimdReal b)
{
  return SSESimdReal(
     _mm_add_pd (a.v0,b.v0)
    ,   _mm_add_pd (a.v1,b.v1)
      );
}

inline SSESimdReal operator+(SSESimdReal a,Real b)
{
  return SSESimdReal(
    _mm_add_pd (a.v0,_mm_set1_pd(b))
   ,  _mm_add_pd (a.v1,_mm_set1_pd(b))
    );
}

inline SSESimdReal operator+(Real b,SSESimdReal a)
{
  return SSESimdReal(
    _mm_add_pd (_mm_set1_pd(b), a.v0)
    ,  _mm_add_pd (_mm_set1_pd(b), a.v1)
      );
}
// Binary operation operator*
inline SSESimdReal operator* (SSESimdReal a,SSESimdReal b)
{
  return SSESimdReal(
     _mm_mul_pd (a.v0,b.v0)
    ,   _mm_mul_pd (a.v1,b.v1)
      );
}

inline SSESimdReal operator*(SSESimdReal a,Real b)
{
  return SSESimdReal(
    _mm_mul_pd (a.v0,_mm_set1_pd(b))
   ,  _mm_mul_pd (a.v1,_mm_set1_pd(b))
    );
}

inline SSESimdReal operator*(Real b,SSESimdReal a)
{
  return SSESimdReal(
    _mm_mul_pd (_mm_set1_pd(b), a.v0)
    ,  _mm_mul_pd (_mm_set1_pd(b), a.v1)
      );
}
// Binary operation operator/
inline SSESimdReal operator/ (SSESimdReal a,SSESimdReal b)
{
  return SSESimdReal(
     _mm_div_pd (a.v0,b.v0)
    ,   _mm_div_pd (a.v1,b.v1)
      );
}

inline SSESimdReal operator/(SSESimdReal a,Real b)
{
  return SSESimdReal(
    _mm_div_pd (a.v0,_mm_set1_pd(b))
   ,  _mm_div_pd (a.v1,_mm_set1_pd(b))
    );
}

inline SSESimdReal operator/(Real b,SSESimdReal a)
{
  return SSESimdReal(
    _mm_div_pd (_mm_set1_pd(b), a.v0)
    ,  _mm_div_pd (_mm_set1_pd(b), a.v1)
      );
}
namespace math {
// Binary operation min
inline SSESimdReal min (SSESimdReal a,SSESimdReal b)
{
  return SSESimdReal(
     _mm_min_pd (a.v0,b.v0)
    ,   _mm_min_pd (a.v1,b.v1)
      );
}

inline SSESimdReal min(SSESimdReal a,Real b)
{
  return SSESimdReal(
    _mm_min_pd (a.v0,_mm_set1_pd(b))
   ,  _mm_min_pd (a.v1,_mm_set1_pd(b))
    );
}

inline SSESimdReal min(Real b,SSESimdReal a)
{
  return SSESimdReal(
    _mm_min_pd (_mm_set1_pd(b), a.v0)
    ,  _mm_min_pd (_mm_set1_pd(b), a.v1)
      );
}
}
namespace math {
// Binary operation max
inline SSESimdReal max (SSESimdReal a,SSESimdReal b)
{
  return SSESimdReal(
     _mm_max_pd (a.v0,b.v0)
    ,   _mm_max_pd (a.v1,b.v1)
      );
}

inline SSESimdReal max(SSESimdReal a,Real b)
{
  return SSESimdReal(
    _mm_max_pd (a.v0,_mm_set1_pd(b))
   ,  _mm_max_pd (a.v1,_mm_set1_pd(b))
    );
}

inline SSESimdReal max(Real b,SSESimdReal a)
{
  return SSESimdReal(
    _mm_max_pd (_mm_set1_pd(b), a.v0)
    ,  _mm_max_pd (_mm_set1_pd(b), a.v1)
      );
}
}

 // IsEmulated
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace math {
// Unary operation sqrt
inline SSESimdReal sqrt (SSESimdReal a)
{
  Real* za = (Real*)(&a);
  return SSESimdReal::fromScalar(math::sqrt(za[0]),math::sqrt(za[1]),math::sqrt(za[2]),math::sqrt(za[3]));
}
}

namespace math {
// Unary operation exp
inline SSESimdReal exp (SSESimdReal a)
{
  Real* za = (Real*)(&a);
  return SSESimdReal::fromScalar(math::exp(za[0]),math::exp(za[1]),math::exp(za[2]),math::exp(za[3]));
}
}

namespace math {
// Unary operation log10
inline SSESimdReal log10 (SSESimdReal a)
{
  Real* za = (Real*)(&a);
  return SSESimdReal::fromScalar(math::log10(za[0]),math::log10(za[1]),math::log10(za[2]),math::log10(za[3]));
}
}

