// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BarycentricGeomShapeComputer.h                              (C) 2000-2026 */
/*                                                                           */
/* Calculation of GeomShapes using barycenters.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_GEOMETRIC_BARYCENTRICGEOMSHAPECOMPUTER_H
#define ARCANE_GEOMETRIC_BARYCENTRICGEOMSHAPECOMPUTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/VariableTypes.h"

#include "arcane/geometry/GeomShapeView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::geometric
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneGeometric
 * \brief Calculates GeomShapes using barycenters.
 *
 * This class allows updating the coordinates of the nodes
 * of a GeomShape and calculating its center and the center of its faces
 * using the barycenter formula. These coordinates must
 * be updated as soon as one of the mesh nodes moves.
 *
 * All methods of this class are static and it is therefore
 * not useful to create instances.
 *
 * There are several ways to update:
 * - via computeAll(GeomShapeMng& shape_mng,VariableNodeReal3& coords,const CellGroup& cells),
 * in which case all GeomShapes of the meshes in \a cells are updated. This is
 * the most performant method if a large number of
 * meshes must be updated.
 * - via computeAll(GeomShapeMutableView elem,const VariableNodeReal3& coords,Cell cell)
 * if you wish to update mesh by mesh.
 */
class ARCANE_GEOMETRY_EXPORT BarycentricGeomShapeComputer
{
 public:

  //! Calculates the information for the mesh \a cell
  static void computeAll(GeomShapeMutableView elem,const VariableNodeReal3& coords,Cell cell);

  //! Calculates the information for the meshes in the group \a cells
  static void computeAll(GeomShapeMng& shape_mng,VariableNodeReal3& coords,const CellGroup& cells);

  /*!
   * \name Calculation of the center and face centers by mesh type
   *
   * The coordinates of the nodes of \a elem must already have been positioned.
   */
  ///@{
  static void computeTriangle3(GeomShapeMutableView elem);
  static void computeQuad4(GeomShapeMutableView elem);
  static void computeTetraedron4(GeomShapeMutableView elem);
  static void computePyramid5(GeomShapeMutableView elem);
  static void computePentaedron6(GeomShapeMutableView elem);
  static void computeHexaedron8(GeomShapeMutableView elem);
  static void computeHeptaedron10(GeomShapeMutableView elem);
  static void computeOctaedron12(GeomShapeMutableView elem);
  ///@}

  /*!
   * \brief Template method.
   *
   * The template parameter \a ItemType must correspond to one of the following types: GeomType::Triangle3, GeomType::Quad4, GeomType::Tetraedron4, GeomType::Pyramid5,
   * GeomType::Pentaedron6, GeomType::Hexaedron8, GeomType::Heptaedron10, GeomType::Octaedron12.
   *
   * The coordinates of the nodes of \a elem must already have been positioned,
   * for example via the call to setNodes().
   *
   * The call is made by specifying the mesh type as defined in ArcaneTypes.h.
   * For example, for a Quad4:
   \code
   Cell cell = ...;
   GeomShapeMng shape_mng = ...;
   GeomShapeMutableView shape_view(shape_mng.mutableShapeView(cell));
   BarycentricGeomShapeComputer::compute<GeomType::Quad4>(shape_view);
   \endcode
   */
  template<GeomType ItemType> static
  void compute(GeomShapeMutableView elem);

  //! Fills the node information of the mesh \a cell with the coordinates of \a node_coord.
  static void setNodes(GeomShapeMutableView elem,const VariableNodeReal3& node_coord,Cell cell)
  {
    Integer nb_node = cell.nbNode();
    for( Integer node_id=0; node_id<nb_node; ++node_id){
      elem.setNode(node_id,node_coord[cell.node(node_id)]);
    }
  }

 private:

  inline static void
  _setFace3D(Integer fid,GeomShapeMutableView& elem,Integer id1,Integer id2,Integer id3,Integer id4)
  {
    elem.setFace(fid, 0.25 * ( elem.node(id1) + elem.node(id2) + elem.node(id3) + elem.node(id4) ));
  }

  inline static void
  _setFace3D(Integer fid,GeomShapeMutableView& elem,Integer id1,Integer id2,Integer id3)
  {
    elem.setFace(fid, (1.0/3.0) * ( elem.node(id1) + elem.node(id2) + elem.node(id3) ));
  }

  inline static void
  _setFace2D(Integer fid,GeomShapeMutableView& elem,Integer id1,Integer id2)
  {
    elem.setFace(fid,
                 Real3( 0.5 * ( elem.node(id1).x + elem.node(id2).x ),
                        0.5 * ( elem.node(id1).y + elem.node(id2).y ),
                        0.0));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::geometric

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
