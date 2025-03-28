// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Float128.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Type flottant 128bit.                                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_FLOAT128_H
#define ARCCORE_BASE_FLOAT128_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

// Tous les compilateurs Linux supportés par Arccore ont le type '__float128'
#if defined(ARCCORE_OS_LINUX)
// Sur certaines plateformes (par exemple ARM64 Grace), le type 'float128'
// correspond à un type pré-défini (en général 'long double')
#  if !defined(__HAVE_DISTINCT_FLOAT128)
#    define ARCCORE_HAS_NATIVE_FLOAT128
#  endif
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Type flottant sur 128 bits.
 *
 * \warning Cette classe est en cours de définition et ne doit pas être
 * utilisée.
 */
class alignas(16) Float128
{
 public:

  Float128() = default;
  constexpr Float128(long double v)
  : m_v(_toNativeType(v))
  {
  }
  constexpr Float128(double v)
  : m_v(_toNativeType(v))
  {
  }
  constexpr Float128& operator=(long double v)
  {
    _setFromLongDouble(v);
    return (*this);
  }
  constexpr Float128& operator=(double v)
  {
    _setFromLongDouble(v);
    return (*this);
  }
  constexpr operator long double() const { return _toLongDouble(); }

 public:

#ifdef ARCCORE_HAS_NATIVE_FLOAT128
#if defined(__aarch64__)
  using NativeType = _Float128;
#else
  using NativeType = __float128;
#endif
  constexpr Float128(NativeType x)
  : m_v(x)
  {}
#else
  using NativeType = long double;
#endif

 private:

  NativeType m_v;

  constexpr NativeType _toNativeType(long double v)
  {
    return static_cast<NativeType>(v);
  }
  constexpr long double _toLongDouble() const
  {
    return static_cast<long double>(m_v);
  }
  constexpr void _setFromLongDouble(long double v)
  {
    m_v = static_cast<NativeType>(v);
  }
  constexpr friend Float128 operator+(Float128 a, Float128 b)
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
