// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshCoarsening.h                                   (C) 2000-2024 */
/*                                                                           */
/* Coarsening of a Cartesian mesh.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_CARTESIANMESHCOARSENING_H
#define ARCANE_CARTESIANMESH_CARTESIANMESHCOARSENING_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array2.h"

#include "arcane/core/ItemTypes.h"
#include "arcane/core/ICartesianMeshGenerationInfo.h"

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneCartesianMesh
 *
 * \brief Coarsens a Cartesian mesh by 2.
 *
 * \deprecated This class is obsolete. The version 2 implementation (CartesianMeshCoarsening2) must be used.
 *
 * This class allows coarsening a Cartesian mesh. Instances
 * of this class are created via ICartesianMesh::createCartesianMeshCoarsening().
 *
 * The initial mesh must be Cartesian and must not have patches.
 *
 * The mesh must be an AMR mesh (IMesh::isAmrActivated()==true).
 *
 * The number of cells in each dimension must be a multiple of 2
 * as well as the number of local cells in each subdomain.
 *
 * The coarsening is done in two phases:
 *
 * - createCoarseCells(), which creates the coarse cells. After calling
 *   this method, it is possible to use coarseCells() to get
 *   the list of coarse cells and refinedCells() to get for
 *   each coarse cell the list of corresponding refined cells.
 * - removeRefinedCells(), which removes cells other than the coarse cells. After this call, there is only a Cartesian mesh
 *   with half the number of cells in each direction. It will then
 *   be possible to call the refinement methods to create additional levels.
 *
 * \code
 * ICartesianMesh* cartesian_mesh = ...;
 * Ref<CartesianMeshCoarsening> coarser = m_cartesian_mesh->createCartesianMeshCoarsening();
 * IMesh* mesh = cartesian_mesh->mesh();
 * CellInfoListView cells(mesh->cellFamily());
 * coarser->createCoarseCells();
 * Int32 index = 0;
 * for( Int32 cell_lid : coarser->coarseCells()){
 *   Cell cell = cells[cell_lid];
 *   info() << "Test: CoarseCell= " << ItemPrinter(cell);
 *   ConstArrayView<Int32> sub_cells(coarser->refinedCells(index));
 *   ++index;
 *   for( Int32 sub_lid : sub_cells )
 *     info() << "SubCell=" << ItemPrinter(cells[sub_lid]);
 * }
 * coarser->removeRefinedCells();
 * \endcode
 */
class ARCANE_CARTESIANMESH_EXPORT CartesianMeshCoarsening
: public TraceAccessor
{
  friend CartesianMeshImpl;

 private:

  ARCANE_DEPRECATED_REASON("Y2024: Use Arcane::CartesianMeshUtils::createCartesianMeshCoarsening2() instead")
  explicit CartesianMeshCoarsening(ICartesianMesh* m);

 public:

  /*!
   * \brief Coarsens the initial mesh by 2.
   *
   * This method is collective.
   */
  void createCoarseCells();

  /*!
   * \brief List of localIds() of refined cells for the parent cell at index \a index.
   *
   * This method is only valid after calling createCoarseCells().
   *
   * In 2D, there are 4 refined cells per coarse cell. In 3D, there are 8.
   */
  ConstArrayView<Int32> refinedCells(Int32 index) const
  {
    return m_refined_cells[index];
  }
  /*!
   * \brief List of localIds() of coarse cells.
   *
   * This method is only valid after calling createCoarseCells().
   */
  ConstArrayView<Int32> coarseCells() const { return m_coarse_cells; }

  /*!
   * \brief Removes refined cells.
   *
   * createCoarseCells() must be called beforehand.
   */
  void removeRefinedCells();

 private:

  ICartesianMesh* m_cartesian_mesh = nullptr;
  Int32 m_verbosity_level = false;
  UniqueArray2<Int32> m_refined_cells;
  UniqueArray<Int32> m_coarse_cells;
  bool m_is_create_coarse_called = false;
  bool m_is_remove_refined_called = false;
  Int64 m_first_own_cell_unique_id_offset = NULL_ITEM_UNIQUE_ID;

 private:

  Int64 _getMaxUniqueId(const ItemGroup& group);
  void _recomputeMeshGenerationInfo();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
