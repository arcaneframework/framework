// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimdOperation.h                                             (C) 2000-2017 */
/*                                                                           */
/* Operations sur les types Simd.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_SIMDOPERATION_H
#define ARCANE_UTILS_SIMDOPERATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Simd.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_HAS_AVX512)
#include "arcane/utils/SimdAVX512Generated.h"
#endif
#if defined(ARCANE_HAS_AVX)
#include "arcane/utils/SimdAVXGenerated.h"
#endif
#if defined(ARCANE_HAS_SSE)
#include "arcane/utils/SimdSSEGenerated.h"
#endif

#include "arcane/utils/SimdEMULGenerated.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline SimdReal3
operator+(const SimdReal3& a,const SimdReal3& b)
{
  return SimdReal3(a.x + b.x,a.y + b.y,a.z + b.z);
}

inline SimdReal3
operator-(const SimdReal3& a)
{
  return SimdReal3(-a.x,-a.y,-a.z);
}

inline SimdReal3
operator-(const SimdReal3& a,const SimdReal3& b)
{
  return SimdReal3(a.x - b.x,a.y - b.y,a.z - b.z);
}

inline SimdReal3
operator*(const SimdReal3& a,Real b)
{
  return SimdReal3(a.x * b,a.y * b,a.z * b);
}

inline SimdReal3
operator*(const SimdReal3& a,const SimdReal& b)
{
  return SimdReal3(a.x * b,a.y * b,a.z * b);
}

inline SimdReal3
operator*(Real b,const SimdReal3& a)
{
  return SimdReal3(b * a.x,b * a.y,b * a.z);
}

inline SimdReal3
operator*(const SimdReal& b,const SimdReal3& a)
{
  return SimdReal3(b * a.x,b * a.y,b * a.z);
}

inline SimdReal3
operator*(const SimdReal3& a,const SimdReal3& b)
{
  return SimdReal3(a.x * b.x,a.y * b.y,a.z * b.z);
}

inline SimdReal3
operator/(const SimdReal3& a,const SimdReal& b)
{
  return SimdReal3(a.x / b,a.y / b,a.z / b);
}

inline SimdReal3
operator/(const SimdReal3& a,Real b)
{
  return SimdReal3(a.x / b,a.y / b,a.z / b);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline SimdReal2
operator+(const SimdReal2& a,const SimdReal2& b)
{
  return SimdReal2(a.x + b.x,a.y + b.y);
}

inline SimdReal2
operator-(const SimdReal2& a,const SimdReal2& b)
{
  return SimdReal2(a.x - b.x,a.y - b.y);
}

inline SimdReal2
operator*(const SimdReal2& a,Real b)
{
  return SimdReal2(a.x * b,a.y * b);
}

inline SimdReal2
operator*(Real b,const SimdReal2& a)
{
  return SimdReal2(b * a.x,b * a.y);
}

inline SimdReal2
operator*(const SimdReal2& a,const SimdReal2& b)
{
  return SimdReal2(a.x * b.x,a.y * b.y);
}

inline SimdReal2
operator/(const SimdReal2& a,const SimdReal& b)
{
  return SimdReal2(a.x / b,a.y / b);
}

inline SimdReal2
operator/(const SimdReal2& a,Real b)
{
  return SimdReal2(a.x / b,a.y / b);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
