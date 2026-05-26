// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshAMRMng.h                                       (C) 2000-2026 */
/*                                                                           */
/* AMR Manager for a Cartesian Mesh.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CARTESIANMESH_CARTESIANMESHAMRMNG_H
#define ARCANE_CARTESIANMESH_CARTESIANMESHAMRMNG_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Class allowing access to the specific AMR methods of the Cartesian mesh.
 *
 * An instance of this class is valid as long as the ICartesianMesh passed in
 * the constructor is valid.
 */
class ARCANE_CARTESIANMESH_EXPORT CartesianMeshAMRMng
{
 public:

  /*!
   * \brief Constructor.
   */
  explicit CartesianMeshAMRMng(ICartesianMesh* cmesh);

 public:

  /*!
   * \brief Number of mesh patches.
   *
   * There is always at least one patch representing the Cartesian mesh.
   */
  Int32 nbPatch() const;

  /*!
   * \brief Returns the \a index-th patch of the mesh.
   *
   * If the mesh is Cartesian, there is only one patch.
   *
   * The returned instance remains valid as long as this instance is not destroyed.
   */
  CartesianPatch amrPatch(Int32 index) const;

  /*!
   * \brief View of the list of patches.
   */
  CartesianMeshPatchListView patches() const;

  /*!
   * \brief Refines a block of the Cartesian mesh.
   *
   * This method can only be called if the mesh is an
   * AMR mesh (IMesh::isAmrActivated()==true).
   *
   * The cells whose center positions are between
   * \a position and \a (position+length) are refined, and the corresponding
   * connectivity information is updated.
   *
   * This operation is collective.
   */
  void refineZone(const AMRZonePosition& position) const;

  /*!
   * \brief Coarsens a block of the Cartesian mesh.
   *
   * This method can only be called if the mesh is an
   * AMR mesh (IMesh::isAmrActivated()==true).
   *
   * The cells whose center positions are between
   * \a position and \a (position+length) are coarsened, and the corresponding
   * connectivity information is updated.
   *
   * All cells in the coarsening zone must be of the same
   * level.
   *
   * Patches that no longer contain cells after calling this method
   * will be deleted.
   *
   * This operation is collective.
   */
  void coarseZone(const AMRZonePosition& position) const;

  /*!
   * \brief Method to start mesh refinement.
   *
   * \warning Experimental method.
   *
   * This method can only be called if the mesh is an
   * AMR mesh (IMesh::isAmrActivated()==true) and the AMR type is 3
   * (PatchCartesianMeshOnly).
   *
   * This method is the first of a trio of methods necessary to
   * refine the mesh:
   * - \a void beginAdaptMesh(Int32 max_nb_levels, Int32 level_to_refine_first)
   * - \a void adaptLevel(Int32 level_to_adapt)
   * - \a void endAdaptMesh()
   *
   * This first method will prepare the mesh for
   * refinement.
   *
   * It is necessary to pass the number of
   * refinement levels that will occur during this refinement phase (\a max_nb_levels).
   *
   * It is recommended to specify the exact number of levels to avoid
   * adjustment of the number of overlap layers during
   * the call to the third method, which is computationally expensive.
   *
   *
   * It is also necessary to pass the first level to be
   * refined.
   * If two levels are already present on the mesh (0 and 1) and you
   * only want to create a third level (level 2) from
   * the second level (level 1), you can set 1 for the parameter
   * \a level_to_refine_first.
   *
   *
   * If two levels are already present on the mesh (0 and 1) and you
   * want to start from scratch, you can set 0 for the parameter
   * \a level_to_refine_first.
   * In this case, the level 1 patches will be deleted, but not the
   * cells/faces/nodes. Once the new level 1 patches are created using the second method, the third method will handle
   * deleting the excess items.
   * This allows the variable values for the
   * cells/faces/nodes that were in a patch before and are
   * preserved in a new patch.
   *
   *
   * Execution example:
   *    * CartesianMeshAMRMng amr_mng(cmesh());
   * amr_mng.clearRefineRelatedFlags();
   *
   * amr_mng.beginAdaptMesh(2, 0);
   * for (Integer level = 0; level < 2; ++level){
   *   // Will perform its calculations and set II_Refine flags on the cells
   *   // of level level.
   *   computeInLevel(level);
   *   amr_mng.adaptLevel(level);
   * }
   * amr_mng.endAdaptMesh();
   *    *
   * This operation is collective.
   *
   * \param max_nb_levels The desired number of refinement levels.
   * \param level_to_refine_first The level that will be refined first.
   */
  void beginAdaptMesh(Int32 max_nb_levels, Int32 level_to_refine_first);

  /*!
   * \brief Method to create a level of mesh refinement.
   *
   * \warning Experimental method.
   *
   * This method can only be called if the mesh is an
   * AMR mesh (IMesh::isAmrActivated()==true) and the AMR type is 3
   * (PatchCartesianMeshOnly).
   *
   * This second method will allow the mesh to be refined level by
   * level.
   *
   * Note that the parameter \a level_to_adapt designates the level to
   * refine, meaning the creation of level \a level_to_adapt +1 (if we want
   * to refine level 0, then level 1 will be created).
   *
   * Before calling this method, you must add the "II_Refine" flag to the
   * cells that must be refined, only on level \a level_to_adapt.
   * To ensure no flags are already present on the mesh, it is
   * possible to call the method \a clearRefineRelatedFlags().
   *
   * For the refinement of cells outside of level 0 (this "ground" level having
   * a special status), the cells that can be refined must
   * possess the "II_InPatch" flag. Cells without the "II_InPatch" flag
   * cannot be refined.
   * \todo Add the "II_InPatch" flag to all level 0 cells?
   *
   * Cells on level \a level_to_adapt that are already refined, but do not have
   * the "II_Refine" flag, may be deleted when calling the
   * third method.
   * This method redraws the patches and creates the new child cells
   * if necessary, but does not delete any cells. The third method will
   * handle the deletion of all cells that do not belong to any patch.
   *
   * Once this method is called, level \a level_to_adapt +1 is ready to
   * be used, notably to mark cells "II_Refine", and call
   * this method again to create another level, etc.
   *
   * This method is intended to be called iteratively, level by level
   * (from the lowest level to the highest level). If patches of levels
   * higher than \a level_to_adapt are detected, they will be deleted.
   * It is therefore possible to call this method for level n, and then call it
   * again for level n-1, for example (however, pay attention to the number of
   * new cells created).
   *
   * This operation is collective.
   *
   * \param level_to_adapt The level to adapt.
   * \param do_fatal_if_useless Triggers an exception if no cells are
   *                            to refine or if level_to_adapt designates a
   *                            level too high compared to the previous
   *                            call.
   */
  void adaptLevel(Int32 level_to_adapt, bool do_fatal_if_useless = false) const;

  /*!
   * \brief Method to finish mesh refinement.
   *
   * \warning Experimental method.
   *
   * This method can only be called if the mesh is an
   * AMR mesh (IMesh::isAmrActivated()==true) and the AMR type is 3
   * (PatchCartesianMeshOnly).
   *
   * This third method will allow the mesh refinement to finish, specifically
   * by deleting cells that no longer belong to any patch.
   *
   * If the highest level refined with the second method does not correspond
   * to the \a max_nb_levels parameter of the first method, there will be
   * an adjustment of the number of overlap layers.
   *
   * This operation is collective.
   */
  void endAdaptMesh();

  /*!
   * \brief Method to delete flags related to mesh refinement
   * for all cells.
   *
   * The flags concerned are:
   * - ItemFlags::II_Coarsen
   * - ItemFlags::II_Refine
   * - ItemFlags::II_JustCoarsened
   * - ItemFlags::II_JustRefined
   * - ItemFlags::II_JustAdded
   * - ItemFlags::II_CoarsenInactive
   */
  void clearRefineRelatedFlags() const;

  /*!
   * \brief Method to modify the number of overlap layers on the highest refinement level.
   *
   * A call to this method will trigger the adjustment of the number of layers
   * for all existing patches.
   *
   * The parameter \a new_size must be an even number (otherwise, it will be modified
   * to the next highest even number).
   *
   * \param new_size The new number of overlap layers.
   */
  void setOverlapLayerSizeTopLevel(Int32 new_size) const;

  /*!
   * \brief Method to disable overlap layers (and destroy them if present).
   *
   * \warning Without this layer, there may be more than one level of
   * refinement between two cells. It is up to the user to manage
   * this constraint.
   *
   * \note To reactivate these layers, a call to
   * \a setOverlapLayerSizeTopLevel() is sufficient.
   */
  void disableOverlapLayer();

  /*!
   * \brief Method to delete one or more layers
   * of ghost cells on a defined refinement level.
   *
   * The desired number of ghost cell layers may be increased
   * by the method. It is necessary to retrieve the returned value
   * to get the final number of ghost cell layers.
   *
   * \param level The refinement level concerned by the deletion
   * of ghost cells.
   *
   * \param target_nb_ghost_layers The desired number of layers after
   * calling this method. ATTENTION: It may be adjusted by the method.
   *
   * \return The final number of ghost cell layers.
   */
  Integer reduceNbGhostLayers(Integer level, Integer target_nb_ghost_layers) const;

  /*!
   * \brief Method to merge patches that can be merged.
   *
   * This method can only be called if the mesh is an
   * AMR mesh (IMesh::isAmrActivated()==true).
   * If the AMR type is not 3 (PatchCartesianMeshOnly), the method does nothing.
   *
   * This method can be useful after several calls to \a refineZone() and \a coarseZone(). However, calling this method is useless after a call to \a adaptLevel() because \a adaptLevel() handles it.
   */
  void mergePatches() const;

  /*!
   * \brief Method to create a sub-level ("level -1").
   *
   * This method can only be called if the mesh is an
   * AMR mesh (IMesh::isAmrActivated()==true).
   *
   * In the case of using AMR type 3 (PatchCartesianMeshOnly), it is
   * possible to call this method during calculation and as many times as
   * necessary (as long as it is possible to divide the size of level 0 by
   * 2).
   * Once level -1 is created, all levels are "upgraded" (meaning level -1 becomes the
   * level 0 "ground").
   */
  void createSubLevel() const;

 private:

  ICartesianMesh* m_cmesh;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif //ARCANE_CARTESIANMESH_CARTESIANMESHAMRMNG_H
