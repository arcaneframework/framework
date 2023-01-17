// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimdMathUtils.h                                             (C) 2000-2022 */
/*                                                                           */
/* Fonctions mathématiques diverses pour les classes SIMD.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_SIMDMATHUTILS_H
#define ARCANE_SIMDMATHUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/SimdOperation.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::math
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Produit vectoriel de \a u par \a v dans \f$R^3\f$.
 */
inline SimdReal3
cross(const SimdReal3& u,const SimdReal3& v)
{
  return SimdReal3(
                    u.y * v.z  - u.z * v.y,
                    u.z * v.x  - u.x * v.z,
                    u.x * v.y  - u.y * v.x
                    );
}

/*!
 * \brief Produit vectoriel de \a u par \a v dans \f$R^2\f$.
 */
inline SimdReal
cross2D(const SimdReal3& u,const SimdReal3& v)
{
  return (u.x * v.y  - u.y * v.x);
}

inline SimdReal
dot(const SimdReal3& u,const SimdReal3& v)
{
  return SimdReal(u.x*v.x + u.y*v.y + u.z*v.z);
}

inline SimdReal
dot(const SimdReal2& u,const SimdReal2& v)
{
  return SimdReal(u.x*v.x + u.y*v.y);
}

ARCCORE_DEPRECATED_2021("Use normL2() instead")
inline SimdReal
abs(const SimdReal3& sr)
{
  SimdReal vr;
  ENUMERATE_SIMD_REAL(si){
    Real3 r(sr.x[si],sr.y[si],sr.z[si]);
    vr[si] = r.normL2();
  }
  return vr;
}

ARCCORE_DEPRECATED_2021("Use normL2() instead")
inline SimdReal
abs(const SimdReal2& sr)
{
  SimdReal vr;
  ENUMERATE_SIMD_REAL(si){
    Real2 r(sr.x[si],sr.y[si]);
    vr[si] = r.normL2();
  }
  return vr;
}

inline SimdReal
normL2(const SimdReal3& sr)
{
  SimdReal vr;
  ENUMERATE_SIMD_REAL(si){
    Real3 r(sr.x[si],sr.y[si],sr.z[si]);
    vr[si] = r.normL2();
  }
  return vr;
}

inline SimdReal
normL2(const SimdReal2& sr)
{
  SimdReal vr;
  ENUMERATE_SIMD_REAL(si){
    Real2 r(sr.x[si],sr.y[si]);
    vr[si] = r.normL2();
  }
  return vr;
}

/*!
 * \brief Produit mixte de \a u, \a v et \a w
 */
inline SimdReal
mixteMul(const SimdReal3& u,const SimdReal3& v,const SimdReal3& w)
{
  return dot(u,cross(v,w));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline SimdReal3x3
prodTens(const SimdReal3& u,const SimdReal3& v)
{
  return SimdReal3x3(u.x*v,u.y*v,u.z*v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup GroupMathUtils
 * \brief Produit matrice matrice entre deux tenseurs.
 */
inline SimdReal3x3
matrixProduct(const SimdReal3x3& t,const SimdReal3x3& v)
{
  return SimdReal3x3(SimdReal3(t.x.x*v.x.x+t.x.y*v.y.x+t.x.z*v.z.x,
                               t.x.x*v.x.y+t.x.y*v.y.y+t.x.z*v.z.y,
                               t.x.x*v.x.z+t.x.y*v.y.z+t.x.z*v.z.z),
                     SimdReal3(t.y.x*v.x.x+t.y.y*v.y.x+t.y.z*v.z.x,
                               t.y.x*v.x.y+t.y.y*v.y.y+t.y.z*v.z.y,
                               t.y.x*v.x.z+t.y.y*v.y.z+t.y.z*v.z.z),
                     SimdReal3(t.z.x*v.x.x+t.z.y*v.y.x+t.z.z*v.z.x,
                               t.z.x*v.x.y+t.z.y*v.y.y+t.z.z*v.z.y,
                               t.z.x*v.x.z+t.z.y*v.y.z+t.z.z*v.z.z));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup GroupMathUtils
 * \brief Transpose la matrice.
 */
inline SimdReal3x3
matrixTranspose(const SimdReal3x3& t)
{
  return SimdReal3x3(SimdReal3(t.x.x, t.y.x, t.z.x),
                     SimdReal3(t.x.y, t.y.y, t.z.y),
                     SimdReal3(t.x.z, t.y.z, t.z.z));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Retourne le minimum de deux SimdReal2
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline SimdReal2
min(const SimdReal2& a,const SimdReal2& b)
{
  return SimdReal2( math::min(a.x,b.x), math::min(a.y,b.y) );
}
/*!
 * \brief Retourne le minimum de deux SimdReal3
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline SimdReal3
min(const SimdReal3& a,const SimdReal3& b)
{
  return SimdReal3( math::min(a.x,b.x), math::min(a.y,b.y), math::min(a.z,b.z) );
}
/*!
 * \brief Retourne le minimum de deux SimdReal2x2
 * \ingroup GroupMathUtils
 */
inline SimdReal2x2
min(const SimdReal2x2& a,const SimdReal2x2& b)
{
  return SimdReal2x2( math::min(a.x,b.x), math::min(a.y,b.y) );
}
/*!
 * \brief Retourne le minimum de deux SimdReal3x3
 * \ingroup GroupMathUtils
 */
inline SimdReal3x3
min(const SimdReal3x3& a,const SimdReal3x3& b)
{
  return SimdReal3x3( math::min(a.x,b.x), math::min(a.y,b.y), math::min(a.z,b.z) );
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Retourne le maximum de deux SimdReal2
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline SimdReal2
max(const SimdReal2& a,const SimdReal2& b)
{
  return SimdReal2( math::max(a.x,b.x), math::max(a.y,b.y) );
}
/*!
 * \brief Retourne le maximum de deux SimdReal3
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline SimdReal3
max(const SimdReal3& a,const SimdReal3& b)
{
  return SimdReal3( math::max(a.x,b.x), math::max(a.y,b.y), math::max(a.z,b.z) );
}
/*!
 * \brief Retourne le maximum de deux SimdReal2x2
 * \ingroup GroupMathUtils
 */
inline SimdReal2x2
max(const SimdReal2x2& a,const SimdReal2x2& b)
{
  return SimdReal2x2( math::max(a.x,b.x), math::max(a.y,b.y) );
}
/*!
 * \brief Retourne le maximum de deux SimdReal3x3
 * \ingroup GroupMathUtils
 */
inline SimdReal3x3
max(const SimdReal3x3& a,const SimdReal3x3& b)
{
  return SimdReal3x3( math::max(a.x,b.x), math::max(a.y,b.y), math::max(a.z,b.z) );
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::math

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
