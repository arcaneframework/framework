// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeomElementView.h                                           (C) 2000-2026 */
/*                                                                           */
/* Views on geometric elements.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_GEOMETRIC_GEOMELEMENTVIEW_H
#define ARCANE_GEOMETRIC_GEOMELEMENTVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real3.h"

#include "arcane/geometry/GeometricGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::geometric
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneGeometric
 * \brief Base class for constant views on geometric elements.
 *
 * The views of this type are only valid as long as the instance
 * they originate from exists. Consequently, these views are primarily
 * used for argument passing and should not be retained.
 */
class GeomElementConstViewBase
{
 public:

  constexpr explicit GeomElementConstViewBase(ARCANE_RESTRICT const Real3POD* ptr)
  : m_s(ptr)
  {}

 public:

  //! Retrieves the value of the i-th node
  constexpr const Real3 operator[](Integer i) const
  {
    return Real3(m_s[i].x, m_s[i].y, m_s[i].z);
  }

  /*!
   * \brief Retrieves the value of the i-th node.
   * \deprecated Use operator[] instead.
   */
  //ARCANE_DEPRECATED inline const Real3 s(Integer i) const
  constexpr const Real3 s(Integer i) const
  {
    return Real3(m_s[i].x, m_s[i].y, m_s[i].z);
  }

 protected:

  ARCANE_RESTRICT const Real3POD* m_s = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneGeometric
 * \brief Base class for modifiable views on geometric elements.
 *
 * The views of this type are only valid as long as the instance
 * they originate from exists. Consequently, these views are primarily
 * used for argument passing and should not be retained.
 */
class GeomElementViewBase
{
 public:

  explicit constexpr GeomElementViewBase(ARCANE_RESTRICT Real3POD* ptr)
  : m_s(ptr)
  {}

 public:

  //! Retrieves the value of the i-th node
  constexpr const Real3 operator[](Integer i) const
  {
    return Real3(m_s[i].x, m_s[i].y, m_s[i].z);
  }

  //! Sets the value of the i-th node to v.
  constexpr void setValue(Integer i, Real3 v)
  {
    m_s[i].x = v.x;
    m_s[i].y = v.y;
    m_s[i].z = v.z;
  }

 protected:

  ARCANE_RESTRICT Real3POD* m_s = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/geometry/GeneratedGeomElementView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::geometric

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
