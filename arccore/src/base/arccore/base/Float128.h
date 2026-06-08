// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Float128.h                                                  (C) 2000-2025 */
/*                                                                           */
/* 128-bit floating-point type.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_FLOAT128_H
#define ARCCORE_BASE_FLOAT128_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

// All Linux compilers supported by Arccore have the '__float128' type,
#if defined(ARCCORE_OS_LINUX)
// It seems that AdaptiveCPP 2510 does not support '__float128'.
#if defined(__x86_64__) && (!defined(__ACPP__))
#define ARCCORE_HAS_NATIVE_FLOAT128
// On certain platforms (for example ARM64 Grace), the 'float128' type
// corresponds to a predefined type (generally 'long double')
#elif !defined(__HAVE_DISTINCT_FLOAT128)
#define ARCCORE_HAS_NATIVE_FLOAT128
#endif
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
 * \brief 128-bit floating-point type.
 *
 * \warning This class is currently under definition and should not be
 * used.
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

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
