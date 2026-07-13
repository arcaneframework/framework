// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MathReal3.h                                                 (C) 2000-2026 */
/*                                                                           */
/* Mathematical operations on Real3.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_MATHREAL3_H
#define ARCCORE_BASE_MATHREAL3_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Real3.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace math
{
  //! Returns the square of the L2 norm of the triplet $\f$x^2+y^2+z^2\f$
  inline constexpr ARCCORE_HOST_DEVICE Real squareNormL2(const Real3& v)
  {
    return v.x * v.x + v.y * v.y + v.z * v.z;
  }

  /*!
   * \brief Indicates if the instance is close to the zero instance.
   *
   * \retval true if math::isNearlyZero() is true for every component.
   * \retval false otherwise.
   */
  inline constexpr ARCCORE_HOST_DEVICE bool isNearlyZero(const Real3& v)
  {
    return math::isNearlyZero(v.x) && math::isNearlyZero(v.y) && math::isNearlyZero(v.z);
  }

  //! Returns the L2 norm of the triplet $\f$\sqrt{v.x^2+v.y^2+v.z^2}\f$
  inline ARCCORE_HOST_DEVICE Real normL2(const Real3& v)
  {
    return math::sqrt(math::squareNormL2(v));
  }

  /*!
    * \brief Normalizes the triplet v
    *
    * If the triplet is non-zero, divides each component by the norm of the triplet
    * (abs()), so that after calling this method, math::normL2() equals 1.
    * If the triplet is zero, does nothing.
    */
  inline Real3& mutableNormalize(Real3& v)
  {
    Real d = math::normL2(v);
    if (!math::isZero(d))
      v.divSame(d);
    return v;
  }

  /*!
    * \brief Returns the triplet v normalized with the L2 norm.
    *
    * If `math::normL2(v)` is non-zero, returns the triplet v divided by `math::normL2(v)`.
    * Otherwise, returns v.
    */
  inline Real3 normalizeL2(const Real3& v)
  {
    Real d = math::normL2(v);
    if (!math::isZero(d))
      return v / d;
    return v;
  }
} // namespace math

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Real3& Real3::
normalize()
{
  return math::mutableNormalize(*this);
}

inline constexpr ARCCORE_HOST_DEVICE bool Real3::
isNearlyZero() const
{
  return math::isNearlyZero(*this);
}

inline ARCCORE_HOST_DEVICE Real Real3::
normL2() const
{
  return math::normL2(*this);
}

inline ARCCORE_HOST_DEVICE Real Real3::
abs() const
{
  return math::normL2(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
