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

/*!
 * \brief Retourne le minimum de deux SimdReal2
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline SimdReal2
min(SimdReal2 a,SimdReal2 b)
{
  return SimdReal2( math::min(a.x,b.x), math::min(a.y,b.y) );
}
/*!
 * \brief Retourne le minimum de deux SimdReal3
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline SimdReal3
min(SimdReal3 a,SimdReal3 b)
{
  return SimdReal3( math::min(a.x,b.x), math::min(a.y,b.y), math::min(a.z,b.z) );
}
/*!
 * \brief Retourne le minimum de deux SimdReal2x2
 * \ingroup GroupMathUtils
 */
inline SimdReal2x2
min(SimdReal2x2 a,SimdReal2x2 b)
{
  return SimdReal2x2( math::min(a.x,b.x), math::min(a.y,b.y) );
}
/*!
 * \brief Retourne le minimum de deux SimdReal3x3
 * \ingroup GroupMathUtils
 */
inline SimdReal3x3
min(SimdReal3x3 a,SimdReal3x3 b)
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
max(SimdReal2 a,SimdReal2 b)
{
  return SimdReal2( math::max(a.x,b.x), math::max(a.y,b.y) );
}
/*!
 * \brief Retourne le maximum de deux SimdReal3
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline SimdReal3
max(SimdReal3 a,SimdReal3 b)
{
  return SimdReal3( math::max(a.x,b.x), math::max(a.y,b.y), math::max(a.z,b.z) );
}
/*!
 * \brief Retourne le maximum de deux SimdReal2x2
 * \ingroup GroupMathUtils
 */
inline SimdReal2x2
max(SimdReal2x2 a,SimdReal2x2 b)
{
  return SimdReal2x2( math::max(a.x,b.x), math::max(a.y,b.y) );
}
/*!
 * \brief Retourne le maximum de deux SimdReal3x3
 * \ingroup GroupMathUtils
 */
inline SimdReal3x3
max(SimdReal3x3 a,SimdReal3x3 b)
{
  return SimdReal3x3( math::max(a.x,b.x), math::max(a.y,b.y), math::max(a.z,b.z) );
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::math

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
