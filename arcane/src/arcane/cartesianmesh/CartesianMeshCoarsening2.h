// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshCoarsening2.h                                  (C) 2000-2024 */
/*                                                                           */
/* Coarsening of a Cartesian mesh.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_CARTESIANMESHCOARSENING2_H
#define ARCANE_CARTESIANMESH_CARTESIANMESHCOARSENING2_H
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
 * \warning This method is experimental.
 *
 * This class allows coarsening a Cartesian mesh. Instances
 * of this class are created via ICartesianMesh::createCartesianMeshCoarsening().
 *
 * The initial mesh must be Cartesian and must not have patches.
 *
 * The mesh must be an AMR mesh (IMesh::isAmrActivated()==true).
 *
 * The number of cells in each dimension must be a multiple of 2
 * as must the number of local cells in each subdomain.
 *
 * The coarsening occurs upon calling createCoarseCell(). After this
 * call, the mesh has the following structure:
 * - coarse cells are created for every quadruplet of existing cells
 *   which thus become refined cells.
 * - each coarse cell is at level 0 (Cell::level()) and each initial cell
 *   is at level 1.
 * - these child cells can be accessed via the Cell::nbHChildren() and
 *   Cell::hChild() methods.
 * - there will be two patches in the mesh. The first will contain the level zero cells
 *   and the second will contain the level 1 cells, which are the
 *   original cells before coarsening.
 *
 * It is then possible to keep only the coarse cells and
 * remove the refined cells by calling the method.
 *
 * - removeRefinedCells() which deletes cells other than the coarse cells. After this call, there is only a Cartesian mesh
 *   with half the number of cells in each direction. It will then
 *   be possible to call the refinement methods to create additional levels.
 * Here is an example of user code:
 *
 * \code
 * ICartesianMesh* cartesian_mesh = ...;
 * Ref<CartesianMeshCoarsening> coarser = CartesianMeshUtils::createCartesianMeshCoarsening(cartesian_mesh);
 * coarser->createCoarseCells();
 * \endcode
 */
class ARCANE_CARTESIANMESH_EXPORT CartesianMeshCoarsening2
: public TraceAccessor
{
  friend CartesianMeshImpl;

 private:

  explicit CartesianMeshCoarsening2(ICartesianMesh* m);

 public:

  /*!
   * \brief Coarsens the initial mesh by 2.
   *
   * This method is collective.
   */
  void createCoarseCells();

  void removeRefinedCells();

 private:

  ICartesianMesh* m_cartesian_mesh = nullptr;
  Int32 m_verbosity_level = false;
  //! uniqueId() of the coarse cells
  UniqueArray<Int64> m_coarse_cells_uid;
  bool m_is_create_coarse_called = false;
  bool m_is_remove_refined_called = false;
  Int64 m_first_own_cell_unique_id_offset = NULL_ITEM_UNIQUE_ID;

 private:

  Int64 _getMaxUniqueId(const ItemGroup& group);
  void _recomputeMeshGenerationInfo();
  void _writeMeshSVG(const String& name);
  void _doDoubleGhostLayers();
  void _createCoarseCells2D();
  void _createCoarseCells3D();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
