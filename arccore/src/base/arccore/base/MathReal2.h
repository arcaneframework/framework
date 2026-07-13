// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MathReal2.h                                                 (C) 2000-2026 */
/*                                                                           */
/* Mathematical operations on Real2.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_MATHREAL2_H
#define ARCCORE_BASE_MATHREAL2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Real2.h"
#include "arccore/base/MathBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::math
{
  /*!
   * \brief Indicates if the instance is close to the zero instance.
   *
   * \retval true if math::isNearlyZero() is true for every component.
   * \retval false otherwise.
   */
  inline constexpr ARCCORE_HOST_DEVICE bool isNearlyZero(const Real2& v)
  {
    return math::isNearlyZero(v.x) && math::isNearlyZero(v.y);
  }

  //! Returns the squared norm of the pair $\f$x^2+y^2+z^2$\f$
  inline constexpr ARCCORE_HOST_DEVICE Real squareNormL2(const Real2& v)
  {
    return v.x * v.x + v.y * v.y;
  }

  //! Returns the norm of the pair $\f$\sqrt{x^2+y^2+z^2}$\f$
  inline ARCCORE_HOST_DEVICE Real normL2(const Real2& v)
  {
    return math::sqrt(math::squareNormL2(v));
  }

  /*!
   * \brief Normalizes the pair.
   *
   * If the pair is non-zero, divides each component by the norm of the pair
   * (abs()), such that after calling this method, abs() equals 1.
   * If the pair is zero, does nothing.
   */
  inline Real2& mutableNormalize(Real2& v)
  {
    Real d = math::normL2(v);
    if (!math::isZero(d))
      v.divSame(d);
    return v;
  }

  /*!
    * \brief Returns the pair v normalized by the L2 norm.
    *
    * If `math::normL2(v)` is non-zero, returns the pair v divided by `math::normL2(v)`.
    * Otherwise, returns v.
    */
  inline Real2 normalizeL2(const Real2& v)
  {
    Real d = math::normL2(v);
    if (!math::isZero(d))
      return v / d;
    return v;
  }
} // namespace math

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline constexpr ARCCORE_HOST_DEVICE bool Real2::
isNearlyZero() const
{
  return math::isNearlyZero(*this);
}

inline ARCCORE_HOST_DEVICE Real Real2::
normL2() const
{
  return math::normL2(*this);
}

inline Real2& Real2::
normalize()
{
  return math::mutableNormalize(*this);
}

inline ARCCORE_HOST_DEVICE Real Real2::
abs() const
{
  return math::normL2(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
