// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FloatInfo.h                                                 (C) 2000-2026 */
/*                                                                           */
/* Informations about limits for floating point types.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_FLOATINFO_H
#define ARCCORE_BASE_FLOATINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

#include <cfloat>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Information about the floating-point type.
 * \note Mandatory specialization for floating-point types.
 */
template <typename T>
class FloatInfo
{
 public:

  //! Indicates that the instantiation is for a floating-point type.
  static constexpr bool isFloatType() { return false; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Specialization of the FloatInfo class for the \c float type.
 */
template <>
class FloatInfo<float>
{
 public:

  //! Indicates that the instantiation is for a floating-point type.
  static constexpr bool isFloatType() { return true; }

 public:

  static constexpr unsigned int precision() { return 1; }
  static constexpr unsigned int maxDigit() { return FLT_DIG; }
  static constexpr float epsilon() { return FLT_EPSILON; }
  static constexpr float nearlyEpsilon() { return FLT_EPSILON * 10.0f; }
  static constexpr float maxValue() { return FLT_MAX; }
  static constexpr float zero() { return 0.0f; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Specialization of the FloatInfo class for the <tt>double</tt> type.
 */
template <>
class FloatInfo<double>
{
 public:

  //! Indicates that the instantiation is for a floating-point type.
  static constexpr bool isFloatType() { return true; }

 public:

  static constexpr unsigned int precision() { return 2; }
  static constexpr unsigned int maxDigit() { return DBL_DIG; }
  static constexpr double epsilon() { return DBL_EPSILON; }
  static constexpr double nearlyEpsilon() { return DBL_EPSILON * 10.0; }
  static constexpr double maxValue() { return DBL_MAX; }
  static constexpr double zero() { return 0.0; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Specialization of the FloatInfo class for the type
 * <tt>long double</tt>.
 */
template <>
class FloatInfo<long double>
{
 public:

  //! Indicates that the instantiation is for a floating-point type.
  static constexpr bool isFloatType() { return true; }

 public:

  static constexpr unsigned int precision() { return 3; }
  static constexpr unsigned int maxDigit() { return LDBL_DIG; }
  static constexpr long double epsilon() { return LDBL_EPSILON; }
  static constexpr long double nearlyEpsilon() { return LDBL_EPSILON * 10.0; }
  static constexpr long double maxValue() { return LDBL_MAX; }
  static constexpr long double zero() { return 0.0l; }
};

#ifdef ARCCORE_REAL_USE_APFLOAT
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Specialization of the FloatInfo class for the type
 * <tt>long double</tt>.
 *
 * \todo Verify that this class is valid for all architectures.
 */
template <>
class FloatInfo<apfloat>
{
 public:

  //! Indicates that the instantiation is for a floating-point type.
  //typedef TrueType _IsFloatType;
  //! Indicates that the instantiation is for a floating-point type.
  static constexpr bool isFloatType() { return true; }

 public:

  static constexpr unsigned int precision() { return 3; }
  static constexpr unsigned int maxDigit() { return 35; }
  static constexpr apfloat epsilon() { return 1e-30; }
  static constexpr apfloat nearlyEpsilon() { return 1e-28; }
  static constexpr apfloat maxValue() { return apfloat("1e1000"); }
  static constexpr apfloat zero() { return apfloat("0.0"); }
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
