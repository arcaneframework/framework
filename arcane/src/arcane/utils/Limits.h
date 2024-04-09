// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Limits.h                                                    (C) 2000-2024 */
/*                                                                           */
/* Fichiers encapsulant <limits> et associés.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_LIMITS_H
#define ARCANE_UTILS_LIMITS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/StdHeader.h"

// Comme <limits> definit min, max, abs, ... et que certains logiciels
// en font des macros, on les supprime
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#ifdef abs
#undef abs
#endif
#include <limits>

#include <float.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations sur le type flottant.
 * \note Spécialisation obligatoire pour les flottants.
 */
template<typename T>
class FloatInfo
{
 public:
  //! Indique si l'instantiation est pour un type flottant.
  typedef FalseType  _IsFloatType;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Spécialisation de la classe FloatInfo pour le type \c float.
 */
template<>
class FloatInfo<float>
{
 public:
  //! Indique que l'instantiation est pour un type flottant.
  typedef TrueType  _IsFloatType;
 public:
  ARCCORE_HOST_DEVICE static constexpr unsigned int precision() { return 1; }
  ARCCORE_HOST_DEVICE static constexpr unsigned int maxDigit() { return FLT_DIG; }
  ARCCORE_HOST_DEVICE static constexpr float epsilon() { return FLT_EPSILON; }
  ARCCORE_HOST_DEVICE static constexpr float nearlyEpsilon() { return FLT_EPSILON*10.0f; }
  ARCCORE_HOST_DEVICE static constexpr float maxValue() { return FLT_MAX; }
  ARCCORE_HOST_DEVICE static constexpr float zero() { return 0.0f; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Spécialisation de la classe FloatInfo pour le type <tt>double</tt>.
 */
template<>
class FloatInfo<double>
{
 public:
  //! Indique que l'instantiation est pour un type flottant.
  typedef TrueType  _IsFloatType;
 public:
  ARCCORE_HOST_DEVICE static constexpr unsigned int precision() { return 2; }
  ARCCORE_HOST_DEVICE static constexpr unsigned int maxDigit() { return DBL_DIG; }
  ARCCORE_HOST_DEVICE static constexpr double epsilon() { return DBL_EPSILON; }
  ARCCORE_HOST_DEVICE static constexpr double nearlyEpsilon() { return DBL_EPSILON*10.0; }
  ARCCORE_HOST_DEVICE static constexpr double maxValue() { return DBL_MAX; }
  ARCCORE_HOST_DEVICE static constexpr double zero() { return 0.0; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Spécialisation de la classe FloatInfo pour le type
 * <tt>long double</tt>.
 *
 * \todo Vérifier que cette classe est valide pour toutes les architectures.
 */
template<>
class FloatInfo<long double>
{
 public:
  //! Indique que l'instantiation est pour un type flottant.
  typedef TrueType  _IsFloatType;
 public:
  ARCCORE_HOST_DEVICE static constexpr unsigned int precision() { return 3; }
  ARCCORE_HOST_DEVICE static constexpr unsigned int maxDigit() { return LDBL_DIG; }
  ARCCORE_HOST_DEVICE static constexpr long double epsilon() { return LDBL_EPSILON; }
  ARCCORE_HOST_DEVICE static constexpr long double nearlyEpsilon() { return LDBL_EPSILON*10.0; }
  ARCCORE_HOST_DEVICE static constexpr long double maxValue() { return LDBL_MAX; }
  ARCCORE_HOST_DEVICE static constexpr long double zero() { return 0.0l; }
};

#ifdef ARCANE_REAL_USE_APFLOAT
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Spécialisation de la classe FloatInfo pour le type
 * <tt>long double</tt>.
 *
 * \todo Vérifier que cette classe est valide pour toutes les architectures.
 */
template<>
class FloatInfo<apfloat>
{
 public:
  //! Indique que l'instantiation est pour un type flottant.
  typedef TrueType  _IsFloatType;
 public:
  ARCCORE_HOST_DEVICE static constexpr unsigned int precision() { return 3; }
  ARCCORE_HOST_DEVICE static constexpr unsigned int maxDigit() { return 35; }
  ARCCORE_HOST_DEVICE static constexpr apfloat epsilon() { return 1e-30; }
  ARCCORE_HOST_DEVICE static constexpr apfloat nearlyEpsilon() { return 1e-28; }
  ARCCORE_HOST_DEVICE static constexpr apfloat maxValue() { return apfloat("1e1000"); }
  ARCCORE_HOST_DEVICE static constexpr apfloat zero() { return apfloat("0.0"); }
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
