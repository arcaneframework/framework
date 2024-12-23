// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Float128.h                                                  (C) 2000-2024 */
/*                                                                           */
/* Type flottant 128bit.                                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_FLOAT128_H
#define ARCCORE_BASE_FLOAT128_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

#include <cstdalign>

// Tous les compilateurs Linux supportés par Arccore ont le type '__float128'
#if defined(ARCCORE_OS_LINUX)
#define ARCCORE_HAS_NATIVE_FLOAT128
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Type flottant demi-précision
 */
class alignas(16) Float128
{
 public:

  Float128() = default;
  Float128(long double v)
  {
    _setFromLongDouble(v);
  }
  Float128& operator=(long double v)
  {
    _setFromLongDouble(v);
    return (*this);
  }
  operator long double() const { return _toLongDouble(); }

 private:

#ifdef ARCCORE_HAS_NATIVE_FLOAT128
#if defined(__aarch64__)
  using NativeType = _Float128;
#else
  using NativeType = __float128;
#endif
  explicit Float128(NativeType x)
  : m_v(x)
  {}
#else
  using NativeType = long double;
#endif

  NativeType m_v;

  long double _toLongDouble() const
  {
    return static_cast<long double>(m_v);
  }
  void _setFromLongDouble(long double v)
  {
    m_v = static_cast<NativeType>(v);
  }
  friend Float128 operator+(Float128 a, Float128 b)
  {
    return Float128(a.m_v + b.m_v);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
