// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeomShapeView.h                                             (C) 2000-2026 */
/*                                                                           */
/* Handling of 2D and 3D geometric shapes.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_GEOMETRIC_GEOMSHAPEVIEW_H
#define ARCANE_GEOMETRIC_GEOMSHAPEVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ArrayView.h"
#include "arcane/utils/Real3.h"

#include "arcane/core/Item.h"

#include "arcane/geometry/GeometricConnectic.h"
#include "arcane/geometry/GeomElement.h"
#include "arcane/geometry/CellConnectivity.h"
#include "arcane/geometry/GeomShapeMutableView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::geometric
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class GeomShapeConnectivity;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneGeometric
 * \brief Constant view on a geometric shape GeomShape.
 *
 * A view on a geometric shape allows for the optimized retrieval of
 * the position of nodes, faces, and edges (in 3D)
 * of a geometric object.
 *
 * This class manages a view on a geometric shape. There are two
 * ways to initialize a view:
 * - by retrieving the view associated with a mesh cell via the call
 * to GeomShapeMng::initShape(). In this case, it is possible to
 * retrieve the associated cell via the cell() method.
 * - from a temporary instance of GeomShape
 * via one of the two methods initFromHexa() or initFromQuad().
 *
 */
class ARCANE_GEOMETRY_EXPORT GeomShapeView
{
  friend class GeomShapeMng;
  friend class GeomShape;
  friend class GeomShapeConnectivity;

 private:

  static CellConnectivity* global_cell_connectivity[NB_BASIC_ITEM_TYPE];
  static GeomShapeConnectivity* global_connectivity;

 public:

  static void initializeConnectivity();

 public:

  GeomShapeView()
  : m_node_ptr(0)
  , m_face_ptr(0)
  , m_center_ptr(0)
  , m_cell_connectivity(global_cell_connectivity[IT_NullType])
  , m_item_internal(ItemInternal::nullItem())
  {
  }

 public:

#include "arcane/geometry/GeneratedGeomShapeViewDeclarations.h"

 public:

  //! Fills \a hexa with the information of the \a i-th sub-control volume
  void fillSubZoneElement(HexaElementView hexa, Integer i);
  //! Fills \a quad with the information of the \a i-th sub-control volume
  void fillSubZoneElement(QuadElementView quad, Integer i);

  /*!
   * \deprecated Use GeomShape::initFromHexaedron8() instead.
   */
  ARCANE_DEPRECATED_122 void initFromHexa(HexaElementConstView hexa, GeomShape& geom_cell);
  /*!
   * \deprecated Use GeomShape::initFromQuad4() instead.
   */
  ARCANE_DEPRECATED_122 void initFromQuad(QuadElementConstView hexa, GeomShape& geom_cell);

 public:

  /*!
   * \name Coordinate Retrieval.
   */
  //@{
  //! Position of the \a i-th node of the shape
  const Real3 node(Integer i) const
  {
    return m_node_ptr[i];
  }

  //! Position of the center of the \a i-th face of the shape
  const Real3 face(Integer i) const
  {
    return m_face_ptr[i];
  }

  //! Position of the center of the shape
  const Real3 center() const
  {
    return *m_center_ptr;
  }

  //! Position of the center of the \a i-th edge of the shape
  inline const Real3 edge(Integer i) const
  {
    return 0.5 * (node(m_cell_connectivity->m_edge_direct_connectic[(i * 2)]) + node(m_cell_connectivity->m_edge_direct_connectic[(i * 2) + 1]));
  }
  //@}

  //! Associated entity (null if none)
  Item item() const { return Item(m_item_internal); }
  //! Associated cell (null if none)
  Cell cell() const { return Cell(m_item_internal); }
  //! Associated face (null if none)
  Face face() const { return Face(m_item_internal); }

 protected:

  void _setArray(const Real3* node_ptr, const Real3* face_ptr, const Real3* center_ptr)
  {
    m_node_ptr = node_ptr;
    m_face_ptr = face_ptr;
    m_center_ptr = center_ptr;
  }

 private:

  ARCANE_RESTRICT const Real3* m_node_ptr;
  ARCANE_RESTRICT const Real3* m_face_ptr;
  ARCANE_RESTRICT const Real3* m_center_ptr;
  //! Connectivity information
  CellConnectivity* m_cell_connectivity;
  //! Information about the original entity (ItemInternal::nullItem() if none)
  ItemInternal* m_item_internal;

 protected:

  //TODO: TO BE REMOVED
  const Real3POD* _nodeView() const { return (Real3POD*)m_node_ptr; }

 public:

  /*!
   * \name Connectivity Information.
   */
  //@{
  //! Node connectivity information.
  const NodeConnectic& nodeConnectic(Integer i) const
  {
    return m_cell_connectivity->nodeConnectic()[i];
  }

  //! Edge connectivity information.
  const EdgeConnectic& edgeConnectic(Integer i) const
  {
    return m_cell_connectivity->edgeConnectic()[i];
  }

  //! Face connectivity information
  const FaceConnectic& faceConnectic(Integer i) const
  {
    return m_cell_connectivity->faceConnectic()[i];
  }

  //! Number of sub-control volumes
  Integer nbSubZone() const
  {
    return m_cell_connectivity->nbSubZone();
  }

  //! Number of internal SVC faces
  Integer nbSvcFace() const
  {
    return m_cell_connectivity->nbSubZoneFace();
  }

  //! Local number of the vertex associated with the sub-control volume
  Integer nodeAssociation(Integer i) const
  {
    return m_cell_connectivity->nodeAssociation()[i];
  }

  const SVCFaceConnectic& svcFaceConnectic(Integer i) const
  {
    return m_cell_connectivity->SCVFaceConnectic()[i];
  }

  //! Local numbers within the sub-control volumes
  Integer edgeNodeSubZoneId(Integer i) const
  {
    return m_cell_connectivity->m_edge_node_sub_zone_id[i];
  }

  Integer faceNodeSubZoneId(Integer i) const
  {
    return m_cell_connectivity->m_face_node_sub_zone_id[i];
  }
  //@}

  /*!
   * \brief Geometric type of the shape.
   *
   * If the shape is associated with an entity (retrievable via item()),
   * it is also the type of the entity.
   *
   * Returns \a GeomType::NullType if the instance is not initialized.
   */
  GeomType geomType() const
  {
    return m_cell_connectivity->cellType();
  }

 protected:

  void _setItem(Item item)
  {
    m_cell_connectivity = global_cell_connectivity[item.type()];
    m_item_internal = ItemCompatibility::_itemInternal(item);
  }

  void _setNullItem(int item_type)
  {
    m_item_internal = ItemInternal::nullItem();
    m_cell_connectivity = global_cell_connectivity[item_type];
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//TODO: Use Traits for the number of nodes and the SVC name.
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneGeometric
 * \brief View on 2D geometric shape.
 */
class GeomShape2DView
: public GeomShapeView
{
 public:

  GeomShape2DView() {}
  explicit GeomShape2DView(const GeomShapeView& rhs)
  : GeomShapeView(rhs)
  {}
};

/*!
 * \ingroup ArcaneGeometric
 * \brief View on 3D geometric shape.
 */
class GeomShape3DView
: public GeomShapeView
{
 public:

  GeomShape3DView() {}
  explicit GeomShape3DView(const GeomShapeView& rhs)
  : GeomShapeView(rhs)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/geometry/GeneratedGeomShapeView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::geometric

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! To be removed eventually
#include "arcane/geometry/GeomShape.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
