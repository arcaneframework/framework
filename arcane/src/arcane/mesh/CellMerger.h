// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CellMerger.h                                                (C) 2000-2023 */
/*                                                                           */
/* Merges two cells.                                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_CELLMERGER_H
#define ARCANE_MESH_CELLMERGER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/MeshGlobal.h"

#include "arcane/utils/String.h"
#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Helper class for merging two cells.
 */
class ARCANE_MESH_EXPORT CellMerger
{
 public:

  //! Constructor
  CellMerger(ITraceMng*) {}

  //! Destructor
  ~CellMerger() = default;

 public:

  /*!
   * \brief Merges the two cells \a i_cell_1 and \a i_cell_2.
   *
   * \param i_cell_1 a pointer to the first cell
   * \param i_cell_2 a pointer to the second cell
   *
   * \note the merge is \b always performed in \a i_cell_1, \a i_cell_2
   * becomes a flattened cell that will be destroyed later.
   */
  void merge(Cell i_cell_1, Cell i_cell_2);

  /*!
   * \brief Returns the ItemInternal used by the cell after merging
   *
   * \param i_cell_1 a pointer to the first cell
   * \param i_cell_2 a pointer to the second cell
   *
   * \return a pointer to the new cell.
   *
   * \note the new pointer is always either \a i_cell_1 or
   * \a i_cell_2. No memory allocation is performed.
   */
  ARCANE_DEPRECATED_REASON("Y2022: Use getCell() instead")
  ItemInternal* getItemInternal(ItemInternal* i_cell_1, ItemInternal* i_cell_2);

  /*!
   * \brief Returns the cell used by the cell after merging
   *
   * \param i_cell_1 a pointer to the first cell
   * \param i_cell_2 a pointer to the second cell
   *
   * \return the new cell.
   *
   * \note the new cell is always either \a i_cell_1 or
   * \a i_cell_2. No memory allocation is performed.
   */
  Cell getCell(Cell i_cell_1, Cell i_cell_2);

 private:

  /*!
   * We define a local enumerated type in order to perform
   * arithmetic operations (see \see _promoteType)
   */
  enum _Type
  {
    NotMergeable = 0,
    Hexahedron = 1,
    Pyramid = 2,
    Pentahedron = 3,
    Quadrilateral = 10,
    Triangle = 11
  };

  /*!
   * \brief Returns the name associated with the cell type
   *
   * \param t the type
   *
   * \return the string containing the name.
   */
  String _typeName(const _Type& t) const;

  /*!
   * \brief Determines the cell _Type based on its "ItemInternal" type
   *
   * \param internal_cell_type the "ItemInternal" type
   *
   * \return the cell _Type
   */
  _Type _getCellType(const Integer& internal_cell_type) const;

  /*!
   * \brief Determines the cell type resulting from the merging of two given types.
   * \note at this stage, nothing guarantees that the merge will succeed
   *
   * \param t1 the first type
   * \param t2 the second type
   *
   * \return the type of the merged cell
   */
  _Type _promoteType(const _Type& t1, const _Type& t2) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
