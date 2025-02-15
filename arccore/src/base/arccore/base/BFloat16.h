// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BFloat16.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Type flottant 'Brain Float16'.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_BFLOAT16_H
#define ARCCORE_BASE_BFLOAT16_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/FloatConversion.h"

//#define ARCCORE_HAS_NATIVE_BFLOAT16

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Type 'Brain Float16'
 */
class ARCCORE_BASE_EXPORT BFloat16
{
 public:

  BFloat16() = default;
  explicit BFloat16(float v)
  {
    _setFromFloat(v);
  }
  BFloat16& operator=(float v)
  {
    _setFromFloat(v);
    return (*this);
  }
  operator float() const { return _toFloat(); }
  friend bool operator==(const BFloat16& a, const BFloat16& b)
  {
    // NOTE: Ne gère pas NaN.
    return a.m_v == b.m_v;
  }
  friend bool operator!=(const BFloat16& a, const BFloat16& b)
  {
    return !(operator==(a, b));
  }
  friend bool operator<(const BFloat16& a, const BFloat16& b)
  {
    float xa = a;
    float xb = b;
    return xa < xb;
  }

 private:

#ifdef ARCCORE_HAS_NATIVE_BFLOAT16
  using NativeType = __bf16;
  NativeType m_v;
  float _toFloat() const
  {
    return static_cast<float>(m_v);
  }
  void _setFromFloat(float v)
  {
    m_v = static_cast<NativeType>(v);
  }
#else
  uint16_t m_v;

  float _toFloat() const
  {
    return impl::convertToBFloat16Impl(m_v);
  }
  void _setFromFloat(float v)
  {
    m_v = impl::convertBFloat16ToUint16Impl(v);
  }
#endif
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
