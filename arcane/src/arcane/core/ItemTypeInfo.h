// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemTypeInfo.h                                              (C) 2000-2026 */
/*                                                                           */
/* Information about a mesh entity type.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMTYPEINFO_H
#define ARCANE_CORE_ITEMTYPEINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

#include "arcane/core/ItemTypeMng.h"
#include "arcane/core/ItemTypeId.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Mesh
 * \brief Info on a mesh entity type.
 *
 * An instance of this class describes a mesh entity type, such as
 * a hexahedral mesh, a quadrilateral mesh.
 *
 * \sa ItemTypeMng
 *
 * There should only be one instance per entity type. Creation of a
 * type is done by the derived class ItemTypeInfoBuilder. Types must
 * be created before any mesh creation (i.e. during
 * the initialization of the architecture).
 */
class ItemTypeInfo
{
 public:

  /*!
   * \brief Local information about a mesh face.
   */
  class LocalFace
  {
   public:

    explicit LocalFace(Integer* index)
    : m_index(index)
    {}

   public:

    //! Type of the face entity
    Integer typeId() const { return m_index[0]; }
    //! Number of nodes of the face
    Integer nbNode() const { return m_index[1]; }
    //! Local index in the mesh of the i-th node of the face
    Integer node(Integer i) const { return m_index[2 + i]; }
    //! Number of edges of the face
    Integer nbEdge() const { return m_index[2 + nbNode()]; }
    //! Edge of the face
    Integer edge(Integer i) const { return m_index[3 + nbNode() + i]; }

   private:

    Integer* m_index; //!< Indices in the face info buffer
  };

  /*!
   * \brief Local information about a mesh edge.
   *
   * \warning To be initialized as an array, this class must be
   * a POD and not have a constructor.
   */
  class LocalEdge
  {
   public:

    explicit LocalEdge(Integer* index)
    : m_index(index)
    {}

   public:

    //! Local index in the mesh of the starting node of the edge
    Integer beginNode() const { return m_index[0]; }
    //! Local index in the mesh of the ending node of the edge
    Integer endNode() const { return m_index[1]; }
    //! Local index in the mesh of the face to the left of the edge
    Integer leftFace() const { return m_index[2]; }
    //! Local index in the mesh of the face to the right of the edge
    Integer rightFace() const { return m_index[3]; }

   private:

    Integer* m_index; //!< Indices in the face info buffer
  };

 protected:

  //! Default constructor
  ItemTypeInfo() = default;

 public:

  //! Type number
  Int16 typeId() const { return m_type_id.typeId(); }
  //! Type number
  ItemTypeId itemTypeId() const { return m_type_id; }
  //! Number of nodes of the entity
  Integer nbLocalNode() const { return m_nb_node; }
  //! Number of faces of the entity
  Integer nbLocalFace() const { return m_nb_face; }
  //! Number of edges of the entity
  Integer nbLocalEdge() const { return m_nb_edge; }
  //! Type name
  String typeName() const { return m_type_name; }
  //! Dimension of the element (<0 if unknown)
  Int16 dimension() const { return m_dimension; }
  //! Indicates if the type is valid for creating a mesh (Cell)
  bool isValidForCell() const { return m_is_valid_for_cell; }
  //! Order of the type
  Int32 order() const { return m_order; }
  //! Type of the corresponding linear element
  Int16 linearTypeId() const { return m_linear_type_id.typeId(); }
  //! Type of the corresponding linear element
  ItemTypeId linearItemTypeId() const { return m_linear_type_id; }
  //! Type of the corresponding linear element
  const ItemTypeInfo* linearTypeInfo() const { return m_mng->typeFromId(m_linear_type_id); }
  //! Indicates if the type has a center node
  bool hasCenterNode() const { return m_has_center_node; }
  /*!
   * \brief Indicates if the type is a polygon.
   *
   * A polygon is a 2D element of order 1 containing at least 5 nodes.
   */
  bool isPolygon() const { return m_is_polygon; }

 public:

  //! Local connectivity of the i-th edge of the mesh
  LocalEdge localEdge(Integer id) const
  {
    Array<Integer>& buf = m_mng->m_ids_buffer;
    Integer fi = buf[m_first_item_index + id];
    return LocalEdge(&buf[fi]);
  }

  //! Local connectivity of the i-th face of the mesh
  LocalFace localFace(Integer id) const
  {
    Array<Integer>& buf = m_mng->m_ids_buffer;
    Integer fi = buf[m_first_item_index + m_nb_edge + id];
    return LocalFace(&buf[fi]);
  }

 protected:

  ItemTypeMng* m_mng = nullptr;
  ItemTypeId m_type_id{ IT_NullType };
  ItemTypeId m_linear_type_id{ IT_NullType };
  //! Dimension (-1) if not initialized.
  Int16 m_dimension = (-1);
  //! Indicates if the type is valid for a mesh.
  bool m_is_valid_for_cell = true;
  //! Indicates if the type has a center node (for faces or meshes)
  bool m_has_center_node = false;
  //! Indicates if the type is a polygon
  bool m_is_polygon = false;
  Integer m_nb_node = 0;
  Integer m_nb_edge = 0;
  Integer m_nb_face = 0;
  Int32 m_order = 1;
  //! Index of this type in the list of indices of \a m_mng.
  Integer m_first_item_index = 0;
  String m_type_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
