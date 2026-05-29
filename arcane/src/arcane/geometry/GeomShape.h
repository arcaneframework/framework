// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeomShape.h                                                 (C) 2000-2026 */
/*                                                                           */
/* Geometric shape.                                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_GEOMETRIC_GEOMSHAPE_H
#define ARCANE_GEOMETRIC_GEOMSHAPE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/geometry/GeomShapeView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::geometric
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneGeometric
 * \brief Geometric shape.
 *
 * This class is only used to create a temporary geometric shape.
 * For a geometric shape derived from a mesh, you must go through the
 * GeomShapeMng.
 *
 * An instance of this class allows storing the information
 * necessary for a geometric shape.
 *
 * It is possible to directly initialize
 * a geometric shape from a hexahedron (initFromHexaedron8())
 * or a quadrangle (initFromQuad4()). These methods initialize the shape
 * and return a view of it.
 * \code
 * GeomShape shape;
 * HexaElement hexa;
 * GeomShapeView shape_view = shape.initFromHexaedron8(hexa);
 * \endcode
 *
 * \todo set up a specific initialization. For this, it will be necessary
 * to use toMutableView() but it is also necessary to be able to specify the geomType().
 */
class ARCANE_GEOMETRY_EXPORT GeomShape
{
  // TEMPORARY: to be deleted when the initFromHexa() and initFromQuad()
  // of GeomShapeView are deleted
  friend class GeomShapeView;

 public:
  
  //! Modifiable view of this instance.
  GeomShapeMutableView toMutableView()
  {
    return GeomShapeMutableView((Real3*)m_node_ptr,(Real3*)m_face_ptr,(Real3*)&m_center);
  }

  //! Initializes the shape with a hexahedron \a hexa and returns a view of it.
  Hexaedron8ShapeView initFromHexaedron8(Hexaedron8ElementConstView hexa);

  //! Initializes the shape with a quadrangle \a quad and returns a view of it.
  Quad4ShapeView initFromQuad4(Quad4ElementConstView quad);

 protected:

  void _setArray(GeomShapeView& shape)
  {
    shape._setArray((Real3*)m_node_ptr,(Real3*)m_face_ptr,(Real3*)&m_center);
  }

 private:

  Real3POD m_node_ptr[ItemStaticInfo::MAX_CELL_NODE];
  Real3POD m_face_ptr[ItemStaticInfo::MAX_CELL_FACE];
  Real3POD m_center;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::geometric

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
