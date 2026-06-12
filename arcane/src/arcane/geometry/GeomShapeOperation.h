// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeomShapeOperation.h                                        (C) 2000-2026 */
/*                                                                           */
/* Operation on a geometric shape.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_GEOMETRIC_GEOMETRICOPERATION_H
#define ARCANE_GEOMETRIC_GEOMETRICOPERATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/AbstractItemOperationByBasicType.h"

#include "arcane/geometry/GeomShapeView.h"
#include "arcane/geometry/GeomShapeMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::geometric
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneGeometric
 * \brief Template class to apply specific operations to a geometric shape view.
 *
 * This class allows providing an operator implementing IItemOperationByBasicType
 * from an instance of OperationFunction that uses views
 * specific to geometric shapes (the derived classes of GeomShapeView).
 *
 * The class \a OperationFunction must provide an apply() method for each shape
 * geometric type (Hexaedron8ShapeView, Quad4ShapeView, ...)
 *
 * The call is then made with a group of cells (CellGroup) by calling
 * the ItemGroup::applyOperation() method with this instance as an argument:
 *
 \code
 * // Definition of the operation
 * class MyFunc
 * {
 *  public:
 *   void apply(Hexaedron8ShapeView view)
 *   {
 *     // Applies the operation for a hexahedron.
 *   }
 * };
 *
 * GeomShapeOperation<MyFunc> op;
 * CellGroup cells;
 * // Applies \a op on the group \a cells
 * cells.applyOperation(&op);
 \endcode
 */
template <typename OperationFunction>
class GeomShapeOperation
: public AbstractItemOperationByBasicType
{
 public:

  /*!
   * \brief Constructs the operator.
   *
   * The first argument is of type \a GeomShapeMng and is used to initialize
   * the operator. Subsequent optional arguments are passed directly to the
   * OperationFunction constructor.
   *
   * \a shape_mng must have been initialized before operations can be applied.
   */
  template <typename... BuildArgs>
  GeomShapeOperation(GeomShapeMng& shape_mng, BuildArgs... compute_function_args)
  : m_shape_mng(shape_mng)
  , m_operation_function(compute_function_args...)
  {
  }

  template <typename ShapeType>
  void apply(ItemVectorView cells)
  {
    ShapeType generic;
    ENUMERATE_CELL (i_cell, cells) {
      Cell cell = *i_cell;
      m_shape_mng.initShape(generic, cell);
      m_operation_function.apply(generic);
    }
  }

  void applyTriangle3(ItemVectorView cells)
  {
    apply<TriangleShapeView>(cells);
  }
  void applyQuad4(ItemVectorView cells)
  {
    apply<QuadShapeView>(cells);
  }
  void applyPentagon5(ItemVectorView cells)
  {
    apply<PentagonShapeView>(cells);
  }
  void applyHexagon6(ItemVectorView cells)
  {
    apply<HexagonShapeView>(cells);
  }

  void applyTetraedron4(ItemVectorView cells)
  {
    apply<TetraShapeView>(cells);
  }
  void applyPyramid5(ItemVectorView cells)
  {
    apply<PyramidShapeView>(cells);
  }
  void applyPentaedron6(ItemVectorView cells)
  {
    apply<PentaShapeView>(cells);
  }
  void applyHexaedron8(ItemVectorView cells)
  {
    apply<HexaShapeView>(cells);
  }
  void applyHeptaedron10(ItemVectorView cells)
  {
    apply<Wedge7ShapeView>(cells);
  }
  void applyOctaedron12(ItemVectorView cells)
  {
    apply<Wedge8ShapeView>(cells);
  }

 public:

  //! Operator instance
  OperationFunction& operation() { return m_operation_function; }
  //! Associated manager
  GeomShapeMng& cellShapeMng() { return m_shape_mng; }

 private:

  GeomShapeMng m_shape_mng;
  OperationFunction m_operation_function;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::geometric

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
