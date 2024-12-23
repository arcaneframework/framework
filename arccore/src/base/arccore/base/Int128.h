// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Int128.h                                                    (C) 2000-2024 */
/*                                                                           */
/* Type flottant 128bit.                                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_INT128_H
#define ARCCORE_BASE_INT128_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

#include <cstdalign>

// Tous les compilateurs Linux supportés par Arccore ont le type '__int128'
#if defined(ARCCORE_OS_LINUX)
#define ARCCORE_HAS_NATIVE_INT128
#endif

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
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
class alignas(16) Int128
{
 public:

  Int128() = default;
  Int128(Int64 v)
  {
    _setFromInt64(v);
  }
  Int128& operator=(Int64 v)
  {
    _setFromInt64(v);
    return (*this);
  }
  Int64 toInt64() const { return _toInt64(); }
  operator Int64() const { return _toInt64(); }

 private:

#ifdef ARCCORE_HAS_NATIVE_INT128
  using NativeType = __int128;
  NativeType m_v;
  explicit Int128(__int128 x)
  : m_v(x)
  {}
#else
  Int64 m_v;
  Int64 m_v2;
#endif

  Int64 _toInt64() const
  {
    return static_cast<Int64>(m_v);
  }
  void _setFromInt64(Int64 v)
  {
    m_v = v;
  }
  friend Int128 operator+(Int128 a, Int128 b)
  {
    return Int128(a.m_v + b.m_v);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
