// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeomElement.h                                               (C) 2000-2026 */
/*                                                                           */
/* Geometric Elements.                                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_GEOMETRIC_GEOMELEMENT_H
#define ARCANE_GEOMETRIC_GEOMELEMENT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/Item.h"
#include "arcane/core/VariableTypes.h"

#include "arcane/geometry/GeomElementView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::geometric
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneGeometric
 * \brief Base class for geometric elements.
 *
 * A geometric element contains the coordinates of the \a NbNode nodes that
 * compose this element.
 *
 * For performance reasons, the default constructor
 * does not initialize the coordinates.
 *
 * To retrieve the coordinates of the i-th node of the geometric element,
 * simply use the [] operator. Modification is done via
 * setItem().
 */
template<int NbNode>
class GeomElementBase
{
 public:
  //! Empty constructor.
  GeomElementBase(){}
  //! Constructor from the coordinates \a coords of the nodes of the entity \a item
  GeomElementBase(const VariableNodeReal3& coords,ItemWithNodes item)
  {
    init(coords,item);
  }
  //! Constructor from the coordinates \a coords.
  GeomElementBase(Real3ConstArrayView coords)
  {
    for( Integer i=0; i<NbNode; ++i )
      setItem(i,coords[i]);
  }

  //! Retrieves the value of the i-th node
  inline const Real3 s(Integer i) const { return Real3(m_s[i].x,m_s[i].y,m_s[i].z); }
  //! Retrieves the value of the i-th node
  inline const Real3 operator[](Integer i) const { return Real3(m_s[i].x,m_s[i].y,m_s[i].z); }
  //! Positions the value of the i-th node at \a v
  inline void setItem(Integer i,const Real3& v) { m_s[i] = v; }
  //! Positions the value of the i-th node at Real3(\a x,\a y,\a z)
  inline void setItem(Integer i,Real x,Real y,Real z) { m_s[i] = Real3(x,y,z); }

  /*!
   * \brief Fills the view \a view with the coordinates of the instance.
   */
  void fillView(Real3ArrayView view) const
  {
    for( Integer i=0; i<NbNode; ++i )
      view[i] = s(i);
  }

  /*!
   * \brief Initializes the coordinates with those of the nodes of the entity \a item
   */
  void init(const VariableNodeReal3& coords,ItemWithNodes item)
  {
    for( Integer i=0; i<NbNode; ++i )
      m_s[i] = coords[item.node(i)];
  }

 protected:

  Real3POD m_s[NbNode];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/geometry/GeneratedGeomElement.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::geometric

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
