// WARNING: This file is generated. Do not edit.


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Emulated Binary operation operator-
inline EMULSimdReal operator- (EMULSimdReal a,EMULSimdReal b)
{
  return EMULSimdReal::fromScalar(a.v0 - b.v0,a.v1 - b.v1);
}

inline EMULSimdReal operator-(EMULSimdReal a,Real b)
{
  return EMULSimdReal::fromScalar(a.v0 - b,a.v1 - b);
}

inline EMULSimdReal operator-(Real a,EMULSimdReal b)
{
  return EMULSimdReal::fromScalar(a - b.v0,a - b.v1);
}
// Emulated Binary operation operator+
inline EMULSimdReal operator+ (EMULSimdReal a,EMULSimdReal b)
{
  return EMULSimdReal::fromScalar(a.v0 + b.v0,a.v1 + b.v1);
}

inline EMULSimdReal operator+(EMULSimdReal a,Real b)
{
  return EMULSimdReal::fromScalar(a.v0 + b,a.v1 + b);
}

inline EMULSimdReal operator+(Real a,EMULSimdReal b)
{
  return EMULSimdReal::fromScalar(a + b.v0,a + b.v1);
}
// Emulated Binary operation operator*
inline EMULSimdReal operator* (EMULSimdReal a,EMULSimdReal b)
{
  return EMULSimdReal::fromScalar(a.v0 * b.v0,a.v1 * b.v1);
}

inline EMULSimdReal operator*(EMULSimdReal a,Real b)
{
  return EMULSimdReal::fromScalar(a.v0 * b,a.v1 * b);
}

inline EMULSimdReal operator*(Real a,EMULSimdReal b)
{
  return EMULSimdReal::fromScalar(a * b.v0,a * b.v1);
}
// Emulated Binary operation operator/
inline EMULSimdReal operator/ (EMULSimdReal a,EMULSimdReal b)
{
  return EMULSimdReal::fromScalar(a.v0 / b.v0,a.v1 / b.v1);
}

inline EMULSimdReal operator/(EMULSimdReal a,Real b)
{
  return EMULSimdReal::fromScalar(a.v0 / b,a.v1 / b);
}

inline EMULSimdReal operator/(Real a,EMULSimdReal b)
{
  return EMULSimdReal::fromScalar(a / b.v0,a / b.v1);
}
namespace math {
// Emulated Binary operation min
inline EMULSimdReal min (EMULSimdReal a,EMULSimdReal b)
{
  return EMULSimdReal::fromScalar(min(a.v0,b.v0),min(a.v1,b.v1));
}

inline EMULSimdReal min(EMULSimdReal a,Real b)
{
  return EMULSimdReal::fromScalar(min(a.v0,b),min(a.v1,b));
}

inline EMULSimdReal min(Real a,EMULSimdReal b)
{
  return EMULSimdReal::fromScalar(min(a,b.v0),min(a,b.v1));
}
}
namespace math {
// Emulated Binary operation max
inline EMULSimdReal max (EMULSimdReal a,EMULSimdReal b)
{
  return EMULSimdReal::fromScalar(max(a.v0,b.v0),max(a.v1,b.v1));
}

inline EMULSimdReal max(EMULSimdReal a,Real b)
{
  return EMULSimdReal::fromScalar(max(a.v0,b),max(a.v1,b));
}

inline EMULSimdReal max(Real a,EMULSimdReal b)
{
  return EMULSimdReal::fromScalar(max(a,b.v0),max(a,b.v1));
}
}

 // IsEmulated
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace math {
// Unary operation sqrt
inline EMULSimdReal sqrt (EMULSimdReal a)
{
  Real* za = (Real*)(&a);
  return EMULSimdReal::fromScalar(math::sqrt(za[0]),math::sqrt(za[1]));
}
}

namespace math {
// Unary operation exp
inline EMULSimdReal exp (EMULSimdReal a)
{
  Real* za = (Real*)(&a);
  return EMULSimdReal::fromScalar(math::exp(za[0]),math::exp(za[1]));
}
}

namespace math {
// Unary operation log10
inline EMULSimdReal log10 (EMULSimdReal a)
{
  Real* za = (Real*)(&a);
  return EMULSimdReal::fromScalar(math::log10(za[0]),math::log10(za[1]));
}
}

