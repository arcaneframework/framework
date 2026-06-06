// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemTypeInfoBuilder.h                                       (C) 2000-2026 */
/*                                                                           */
/* Construction of a mesh entity type.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMTYPEINFOBUILDER_H
#define ARCANE_CORE_ITEMTYPEINFOBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypeInfo.h"

#include <array>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Construction of the information for a mesh entity type.
 *
 * For performance reasons, we try to store this information
 * contiguously in memory, as they will be accessed very often.
 * To do this, we use a Buffer in ItemTypeMng.
 *
 * The setInfos() method allows you to specify the type and the number of nodes, edges
 * and faces of the type. You must then call the addEdge() and addFace...() methods.
 *
 * Once the type is completely defined, it must no longer be modified.
 *
 * For a given type number, there is only one instance of ItemTypeInfo, and this
 * instance remains valid as long as the type manager (ItemTypeMng) is not destroyed.
 * Consequently, it is possible to store the pointer to the instance and
 * to compare two types solely by comparing the pointers.
 */
class ItemTypeInfoBuilder
: public ItemTypeInfo
{
 public:

  //! Dimension of the type
  enum class Dimension : Int16
  {
    DimUnknown = -1,
    Dim0 = 0,
    Dim1 = 1,
    Dim2 = 2,
    Dim3 = 3
  };

 public:

  //! Default constructor
  ItemTypeInfoBuilder() = default;

 public:

  ARCANE_DEPRECATED_REASON("Y2025: Use setInfo(...,Dimension dimension, ...) instead")
  void setInfos(ItemTypeMng* mng, Int16 type_id, String type_name,
                Int32 nb_node, Int32 nb_edge, Int32 nb_face);

  ARCANE_DEPRECATED_REASON("Y2025: Use setInfo(...,Dimension dimension, ...) instead")
  void setInfos(ItemTypeMng* mng, ItemTypeId type_id, String type_name,
                Int32 nb_node, Int32 nb_edge, Int32 nb_face);

  /*!
   * \brief Positions the information for a type.
   */
  ARCANE_DEPRECATED_REASON("Y2025: Use setInfo(...,Dimension dimension, ...) instead")
  void setInfos(ItemTypeMng* mng, ItemTypeId type_id, String type_name, Int16 dimension,
                Int32 nb_node, Int32 nb_edge, Int32 nb_face);

  /*!
   * \brief Positions the information for a type.
   */
  void setInfos(ItemTypeMng* mng, ItemTypeId type_id, String type_name, Dimension dimension,
                Int32 nb_node, Int32 nb_edge, Int32 nb_face);

  /*!
   * \brief Positions the information for a type.
   */
  void setInfos(ItemTypeMng* mng, Int16 type_id, String type_name, Dimension dimension,
                Int32 nb_node, Int32 nb_edge, Int32 nb_face);

  /*!
   * \brief Positions the order of the type.
   *
   * If not called, the type is considered to be for order 1 entities.
   * The first argument is the entity order and the second is the corresponding order 1 element.
   */
  void setOrder(Int16 order, ItemTypeId linear_type);

  /*!
   * \brief Adds an edge to the list of edges
   *
   * \a n0 origin node
   * \a n1 end node
   * \a f_left local number of the left face
   * \a f_right local number of the right face
   */
  void addEdge(Integer edge_index, Integer n0, Integer n1, Integer f_left, Integer f_right);

  /*!
   * \brief Adds an edge and a face
   *
   * \a edge_face_index local index of the edge and the face.
   * \a begin_end_node pair (origin node, end node) of the edge and the face to be added.
   * \a left_and_right_face pair local number (left face, right face) of the edge to be added
   */
  void addEdgeAndFaceLine(Int32 edge_face_index,
                          std::array<Int16, 2> begin_end_node,
                          std::array<Int16, 2> left_and_right_face);

  /*!
   * \brief Adds an edge for a 2D mesh in a 3D mesh.
   *
   * \a n0 origin node
   * \a n1 end node
   */
  void addEdge2D(Integer edge_index, Integer n0, Integer n1);

  //! Adds a vertex to the list of faces (for 1D elements)
  void addFaceVertex(Integer face_index, Integer n0);

  //! Adds a line to the list of faces (for 2D elements)
  void addFaceLine(Integer face_index, Integer n0, Integer n1);

  //! Adds a quadratic line to the list of faces (for 2D elements)
  void addFaceLine3(Integer face_index, Integer n0, Integer n1, Integer n2);

  //! Adds a third-order line to the list of faces (for 2D elements)
  void addFaceLine4(Integer face_index, Integer n0, Integer n1, Integer n2, Integer n3);

  //! Adds a triangle to the list of faces
  void addFaceTriangle(Integer face_index, Integer n0, Integer n1, Integer n2);

  //! Adds a quadratic triangle to the list of faces
  void addFaceTriangle6(Integer face_index, Integer n0, Integer n1, Integer n2,
                        Integer n3, Integer n4, Integer n5);

  //! Adds a quadrilateral to the list of faces
  void addFaceQuad(Integer face_index, Integer n0, Integer n1, Integer n2, Integer n3);

  //! Adds a quadratic quadrilateral to the list of faces
  void addFaceQuad8(Integer face_index, Integer n0, Integer n1, Integer n2, Integer n3,
                    Integer n4, Integer n5, Integer n6, Integer n7);

  //! Adds a quadratic quadrilateral to the list of faces
  void addFaceQuad9(Integer face_index, Integer n0, Integer n1, Integer n2, Integer n3,
                    Integer n4, Integer n5, Integer n6, Integer n7, Integer n8);

  //! Adds a pentagon to the list of faces
  void addFacePentagon(Integer face_index, Integer n0, Integer n1, Integer n2, Integer n3, Integer n4);

  //! Adds a hexagon to the list of faces
  void addFaceHexagon(Integer face_index, Integer n0, Integer n1, Integer n2, Integer n3,
                      Integer n4, Integer n5);

  //! Adds a heptagon to the list of faces
  void addFaceHeptagon(Integer face_index, Integer n0, Integer n1, Integer n2, Integer n3,
                       Integer n4, Integer n5, Integer n6);

  //! Adds an octagon to the list of faces
  void addFaceOctogon(Integer face_index, Integer n0, Integer n1, Integer n2, Integer n3,
                      Integer n4, Integer n5, Integer n6, Integer n7);

  //! Adds a generic face to the list of faces
  void addFaceGeneric(Integer face_index, Integer type_id, ConstArrayView<Integer> n);

  //! Computes the face->edge relations
  void computeFaceEdgeInfos();

  //! Positions the information indicating if the type is valid for a cell
  void setIsValidForCell(bool is_valid)
  {
    m_is_valid_for_cell = is_valid;
  }

  //! Positions the information indicating if the type has a center node
  void setHasCenterNode(bool has_center_node)
  {
    m_has_center_node = has_center_node;
  }

 private:

  void _setNbEdgeAndFace(Integer nb_edge, Integer nb_face);
  void _checkDimension(Int16 dim);
  void _checkSetIsPolygon();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
