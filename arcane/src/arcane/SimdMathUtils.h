// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimdMathUtils.h                                             (C) 2000-2017 */
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

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace math
{
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

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
