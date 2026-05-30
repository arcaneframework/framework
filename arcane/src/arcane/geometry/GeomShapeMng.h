// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeomShapeMng.h                                              (C) 2000-2026 */
/*                                                                           */
/* Class managing the GeomShapes of a mesh.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_GEOMETRIC_GEOMSHAPEMNG_H
#define ARCANE_GEOMETRIC_GEOMSHAPEMNG_H
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
 * \brief Class managing the GeomShapes of a mesh.
 *
 * This class stores the information of the geometric shapes (GeomShape) associated
 * to the mesh cells. For a cell, retrieving a view is done
 * via the initShape() method:
 \code
 GeomShapeMng shape_mng;
 Cell cell;
 GeomShapeView shape_view;
 // Initializes the view \a shape_view on the cell \a cell
 shape_mng.initShape(shape_view,cell);
 \endcode
 *
 * A view can be used multiple times. For example, if you want
 * to iterate over multiple cells:
 \code
 * GeomShapeMng shape_mng;
 * GeomShapeView shape_view;
 * ENUMERATE_CELL(icell,allCells()){
 *   Cell cell = *icell;
 *   // Initializes the view \a shape_view on the cell \a cell
 *   shape_mng.initShape(shape_view,cell);
 *   info() << "Node0=" << shape_view.node(0);
 * }
 \endcode
 
 * The view retrieved by GeomShapeView is constant. To retrieve a
 * mutable view, you must use mutableShapeView(). The mutable view
 * is only used to update the different
 * coordinates (nodes, face centers, ...).
 *
 * Before being able to use one of the initShape() or mutableShapeView() methods,
 * you must initialize one of the instances by calling initialize().
 * Initialization only performs memory allocation but does not update
 * the coordinates.
 * \warning The initialize() method must also be called when the topology
 * of the mesh changes, for example after adding or deleting a cell.
 *
 * This class only manages the data on the geometric shapes and
 * these are independent of other variables. This means
 * that if the coordinates of a mesh node change, you must explicitly
 * update the geometric shape information. %Arcane provides
 * the BarycentricGeomShapeComputer class for this, but the user
 * can calculate this information in another way than using the barycenter. 
 *
 * All instances of this class whose name name() is identical
 * are implicitly shared and therefore provide the same GeomShapeView.
 * For example:
 \code
 IMesh* mesh;
 GeomShapeMng shape_mng(mesh,"GenericElement");
 GeomShapeMng shape_mng2(shape_mng);
 // shape_mng and shape_mng2 share the same GeomShapeView

 GeomShapeMng shape_mng3(mesh,"AleGenericElement");
 // shape_mng and shape_mng3 use different values.
 \endcode
 *
 */
class ARCANE_GEOMETRY_EXPORT GeomShapeMng
{
  // NOTE:
  // Since this class can be used by copy or created directly
  // via variable names, it should not contain fields
  // other than Arcane variables so that there are no inconsistencies
  // between the different instances.

 public:

  //! Initializes for the mesh \a mesh with the name \a name
  GeomShapeMng(IMesh* mesh, const String& name);
  //! Initializes for the mesh \a mesh with the default name \a GenericElement
  GeomShapeMng(IMesh* mesh);
  //! Copy constructor.
  GeomShapeMng(const GeomShapeMng& rhs);

 public:

  //! Indicates if the instance is initialized.
  bool isInitialized() const { return m_cell_shape_nodes.arraySize() != 0; }

  /*!
   * \brief Initializes the instance.
   *
   * Only instances with the same name need to be initialized once.
   */
  void initialize();

  //! Initializes the view \a ge with the information of the cell \a cell
  void initShape(GeomShapeView& ge, Cell cell) const
  {
    ge._setArray(m_cell_shape_nodes[cell].data(), m_cell_shape_faces[cell].data(), &m_cell_shape_centers[cell]);
    ge._setItem(cell);
  }

  //! Returns a mutable view on the GeomShape of the cell \a cell
  GeomShapeMutableView mutableShapeView(Cell cell)
  {
    return GeomShapeMutableView(m_cell_shape_nodes[cell].data(), m_cell_shape_faces[cell].data(), &m_cell_shape_centers[cell]);
  }

  //! Manager name.
  const String& name() const { return m_name; }

 private:

  String m_name;
  VariableCellArrayReal3 m_cell_shape_nodes; //!< Generic elements nodes
  VariableCellArrayReal3 m_cell_shape_faces; //!< Generic elements face
  VariableCellReal3 m_cell_shape_centers; //!< Generic elements center
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef GeomShapeMng GeomCellMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::geometric

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
