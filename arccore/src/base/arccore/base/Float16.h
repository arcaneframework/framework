// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Float16.h                                                   (C) 2000-2025 */
/*                                                                           */
/* Type flottant demi-précision.                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_FLOAT16_H
#define ARCCORE_BASE_FLOAT16_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/FloatConversion.h"

// #define ARCCORE_HAS_NATIVE_FLOAT16

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Type flottant demi-précision
 */
class ARCCORE_BASE_EXPORT Float16
{
 public:

  Float16() = default;
  explicit Float16(float v)
  {
    _setFromFloat(v);
  }
  Float16& operator=(float v)
  {
    _setFromFloat(v);
    return (*this);
  }
  operator float() const { return _toFloat(); }

 private:

#ifdef ARCCORE_HAS_NATIVE_FLOAT16
  using NativeType = __fp16;

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
    return impl::convertToFloat16Impl(m_v);
  }
  void _setFromFloat(float v)
  {
    m_v = impl::convertFloat16ToUint16Impl(v);
  }
#endif
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
