// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemTypeId.h                                                (C) 2000-2026 */
/*                                                                           */
/* Type of an entity.                                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMTYPEID_H
#define ARCANE_CORE_ITEMTYPEID_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Mesh
 * \brief Type of an entity (Item).
 */
class ARCANE_CORE_EXPORT ItemTypeId
{
 public:

  ItemTypeId() = default;
  constexpr ARCCORE_HOST_DEVICE explicit ItemTypeId(Int16 id)
  : m_type_id(id)
  {}
  constexpr ARCCORE_HOST_DEVICE operator Int16() const { return m_type_id; }

 public:

  constexpr ARCCORE_HOST_DEVICE Int16 typeId() const { return m_type_id; }
  constexpr ARCCORE_HOST_DEVICE bool isNull() const { return m_type_id == IT_NullType; }
  /*!
   * \brief Creates an instance from an integer.
   *
   * This method throws an exception if \a v is greater than the maximum allowed value,
   * which is currently 2^15.
   */
  static ItemTypeId fromInteger(Int64 v);

 private:

  Int16 m_type_id = IT_NullType;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Unknown or null entity type number
static constexpr ItemTypeId ITI_NullType(IT_NullType);
//! Entity type number Node (1 vertex 1D, 2D and 3D)
static constexpr ItemTypeId ITI_Vertex(IT_Vertex);
//! Entity type number Edge (2 vertices, 1D, 2D and 3D)
static constexpr ItemTypeId ITI_Line2(IT_Line2);
//! Entity type number Triangle (3 vertices, 2D)
static constexpr ItemTypeId ITI_Triangle3(IT_Triangle3);
//! Entity type number Quadrilateral (4 vertices, 2D)
static constexpr ItemTypeId ITI_Quad4(IT_Quad4);
//! Entity type number Pentagon (5 vertices, 2D)
static constexpr ItemTypeId ITI_Pentagon5(IT_Pentagon5);
//! Entity type number Hexagon (6 vertices, 2D)
static constexpr ItemTypeId ITI_Hexagon6(IT_Hexagon6);
//! Entity type number Tetrahedron (4 vertices, 3D)
static constexpr ItemTypeId ITI_Tetraedron4(IT_Tetraedron4);
//! Entity type number Pyramid (5 vertices, 3D)
static constexpr ItemTypeId ITI_Pyramid5(IT_Pyramid5);
//! Entity type number Prism (6 vertices, 3D)
static constexpr ItemTypeId ITI_Pentaedron6(IT_Pentaedron6);
//! Entity type number Hexahedron (8 vertices, 3D)
static constexpr ItemTypeId ITI_Hexaedron8(IT_Hexaedron8);
//! Entity type number Heptahedron (prism with a pentagonal base)
static constexpr ItemTypeId ITI_Heptaedron10(IT_Heptaedron10);
//! Entity type number Octahedron (prism with a hexagonal base)
static constexpr ItemTypeId ITI_Octaedron12(IT_Octaedron12);
//! Entity type number HemiHexa7 (hexahedron with 1 degeneracy)
static constexpr ItemTypeId ITI_HemiHexa7(IT_HemiHexa7);
//! Entity type number HemiHexa6 (hexahedron with 2 non-contiguous degeneracies)
static constexpr ItemTypeId ITI_HemiHexa6(IT_HemiHexa6);
//! Entity type number HemiHexa5 (hexahedrons with 3 non-contiguous degeneracies)
static constexpr ItemTypeId ITI_HemiHexa5(IT_HemiHexa5);
//! Entity type number AntiWedgeLeft6 (hexahedron with 2 contiguous degeneracies)
static constexpr ItemTypeId ITI_AntiWedgeLeft6(IT_AntiWedgeLeft6);
//! Entity type number AntiWedgeRight6 (hexahedron with 2 contiguous degeneracies (second form))
static constexpr ItemTypeId ITI_AntiWedgeRight6(IT_AntiWedgeRight6);
//! Entity type number DiTetra5 (hexahedron with 3 orthogonal degeneracies)
static constexpr ItemTypeId ITI_DiTetra5(IT_DiTetra5);
//! Number of the dual node entity type of a vertex
static constexpr ItemTypeId ITI_DualNode(IT_DualNode);
//! Number of the dual node entity type of an edge
static constexpr ItemTypeId ITI_DualEdge(IT_DualEdge);
//! Number of the dual node entity type of a face
static constexpr ItemTypeId ITI_DualFace(IT_DualFace);
//! Number of the dual node entity type of a cell
static constexpr ItemTypeId ITI_DualCell(IT_DualCell);
//! Entity type number Link
static constexpr ItemTypeId ITI_Link(IT_Link);
//! Entity type number Face for 1D meshes.
static constexpr ItemTypeId ITI_FaceVertex(IT_FaceVertex);
//! Entity type number Cell for 1D meshes.
static constexpr ItemTypeId ITI_CellLine2(IT_CellLine2);
//! Number of the dual node entity type of a particle
static constexpr ItemTypeId ITI_DualParticle(IT_DualParticle);

//! Entity type number Ennehedron (prism with a heptagonal base)
static constexpr ItemTypeId ITI_Enneedron14(IT_Enneedron14);
//! Entity type number Decahedron (prism with an Octagonal base)
static constexpr ItemTypeId ITI_Decaedron16(IT_Decaedron16);

//! Entity type number Heptagon 2D (heptagonal)
static constexpr ItemTypeId ITI_Heptagon7(IT_Heptagon7);

//! Entity type number Octagon 2D (octagonal)
static constexpr ItemTypeId ITI_Octogon8(IT_Octogon8);

//! Quadratic elements
//@{
//! Order 2 Line
static constexpr ItemTypeId ITI_Line3(IT_Line3);
//! Order 2 Triangle
static constexpr ItemTypeId ITI_Triangle6(IT_Triangle6);
//! Order 2 Quadrangle (with 4 nodes on the faces)
static constexpr ItemTypeId ITI_Quad8(IT_Quad8);
//! Order 2 Tetrahedron
static constexpr ItemTypeId ITI_Tetraedron10(IT_Tetraedron10);
//! Order 2 Hexahedron
static constexpr ItemTypeId ITI_Hexaedron20(IT_Hexaedron20);
//! Order 2 Hexahedron
static constexpr ItemTypeId ITI_Pentaedron15(IT_Pentaedron15);
//! Order 2 Pyramid
static constexpr ItemTypeId ITI_Pyramid13(IT_Pyramid13);
//@}

//! Line3 Mesh. EXPERIMENTAL !
static constexpr ItemTypeId ITI_CellLine3(IT_CellLine3);

/*!
 * \brief 2D meshes in a 3D mesh.
 * \warning These types are experimental and should not be used outside of %Arcane.
 */
//@{
//! Line2 Mesh in a 3D mesh. EXPERIMENTAL !
static constexpr ItemTypeId ITI_Cell3D_Line2(IT_Cell3D_Line2);
//! Triangular Mesh with 3 nodes in a 3D mesh. EXPERIMENTAL !
static constexpr ItemTypeId ITI_Cell3D_Triangle3(IT_Cell3D_Triangle3);
//! Quadrangular Mesh with 4 nodes in a 3D mesh. EXPERIMENTAL !
static constexpr ItemTypeId ITI_Cell3D_Quad4(IT_Cell3D_Quad4);
//! Line3 Mesh in a 3D mesh. EXPERIMENTAL !
static constexpr ItemTypeId ITI_Cell3D_Line3(IT_Cell3D_Line3);
//! Triangular Mesh with 6 nodes in a 3D mesh. EXPERIMENTAL !
static constexpr ItemTypeId ITI_Cell3D_Triangle6(IT_Cell3D_Triangle6);
//! Quadrangular Mesh with 8 nodes in a 3D mesh. EXPERIMENTAL !
static constexpr ItemTypeId ITI_Cell3D_Quad8(IT_Cell3D_Quad8);
//! Quadrangular Mesh with 9 nodes in a 3D mesh. EXPERIMENTAL !
static constexpr ItemTypeId ITI_Cell3D_Quad9(IT_Cell3D_Quad9);
//@}

//! Order 2 Quadrangle (with 4 nodes on the faces and 1 node in the center). EXPERIMENTAL !
static constexpr ItemTypeId ITI_Quad9(IT_Quad9);
//! Order 2 Hexahedron (with 12 nodes on the edges, 6 on the faces and one center node. EXPERIMENTAL !
static constexpr ItemTypeId ITI_Hexaedron27(IT_Hexaedron27);

//! Order 3 Line. EXPERIMENTAL !
static constexpr ItemTypeId ITI_Line4(IT_Line4);
//! Order 3 Triangle. EXPERIMENTAL !
static constexpr ItemTypeId ITI_Triangle10(IT_Triangle10);
//! Order 3 Line. EXPERIMENTAL !
static constexpr ItemTypeId ITI_CellLine4(IT_CellLine4);
//! Order 3 Line. EXPERIMENTAL !
static constexpr ItemTypeId ITI_Cell3D_Line4(IT_Cell3D_Line4);

//! Order 3 Triangle in a 3D mesh. EXPERIMENTAL !
static constexpr ItemTypeId ITI_Cell3D_Triangle10(IT_Cell3D_Triangle10);

//! First value for generic polygon types (EXPERIMENTAL)
static constexpr ItemTypeId ITI_GenericPolygon(IT_GenericPolygon);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
