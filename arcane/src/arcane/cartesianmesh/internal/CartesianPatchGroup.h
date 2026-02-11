// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianPatchGroup.h                                       (C) 2000-2026 */
/*                                                                           */
/* Gestion du groupe de patchs du maillage cartésien.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_CARTESIANPATCHGROUP_H
#define ARCANE_CARTESIANMESH_CARTESIANPATCHGROUP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "AMRPatchPositionLevelGroup.h"
#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

#include "arcane/core/ItemGroup.h"

#include "arcane/utils/UniqueArray.h"
#include "arcane/utils/Ref.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/cartesianmesh/CartesianMeshPatchListView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CartesianMeshPatch;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CARTESIANMESH_EXPORT CartesianPatchGroup
: public TraceAccessor
{
 public:

  explicit CartesianPatchGroup(ICartesianMesh* cmesh);

 public:

  void build();
  void saveInfosInProperties();
  void recreateFromDump();

  /*!
   * \brief Méthode permettant de récupérer le patch de niveau 0.
   *
   * Création si non existant.
   */
  Ref<CartesianMeshPatch> groundPatch();

  /*!
   * \brief Méthode permettant de créer un patch contenant les mailles données
   * en paramètre.
   *
   * Cette méthode est compatible uniquement avec l'AMR type 1 (historique).
   */
  void addPatch(ConstArrayView<Int32> cells_local_id);

  /*!
   * \brief Méthode permettant de créer un patch à partir de la zone passée en
   * paramètre.
   *
   * Cette méthode utilisant les mêmes méthodes que l'adaptation de maillage,
   * elle peut être utilisée en cours de calcul.
   *
   * Cette méthode est compatible uniquement avec l'AMR type 3
   * (AMR PatchCartesianMeshOnly).
   */
  void addPatch(const AMRZonePosition& zone_position);

  Integer nbPatch() const;
  Ref<CartesianMeshPatch> patch(Integer index) const;
  CartesianMeshPatchListView patchListView() const;

  /*!
   * \brief Méthode permettant de récupérer le groupe de toutes les mailles du
   * patch demandé.
   *
   * \param index Index (tableau) du patch.
   */
  CellGroup allCells(Integer index);

  /*!
   * \brief Méthode permettant de récupérer le groupe des mailles du
   * patch demandé.
   *
   * \param index Index (tableau) du patch.
   */
  CellGroup inPatchCells(Integer index);

  /*!
   * \brief Méthode permettant de récupérer le groupe des mailles de
   * recouvrement du patch demandé.
   *
   * \param index Index (tableau) du patch.
   */
  CellGroup overlapCells(Integer index);

  /*!
   * \brief Méthode permettant de supprimer tous les patchs. Attention, les
   * mailles des patchs ne seront pas supprimées.
   */
  void clear();

  /*!
   * \brief Méthode permettant de supprimer un patch.
   *
   * \note Cette méthode est en deux temps. Pour déclencher la suppression du
   * ou des patchs, il est nécessaire d'appeler la méthode \a applyPatchEdit().
   *
   * \param index Index (tableau) du patch.
   */
  void removePatch(Integer index);

  /*!
   * \brief Méthode permettant de supprimer des mailles de tous les patchs.
   *
   * Les patchs vides seront supprimés.
   *
   * Cette méthode est compatible uniquement avec l'AMR type 1 (historique).
   */
  void removeCellsInAllPatches(ConstArrayView<Int32> cells_local_id);

  /*!
   * \brief Méthode permettant de supprimer une zone de mailles.
   *
   * Cette zone peut être sur plusieurs patchs à la fois, mais doit désigner
   * des mailles d'un même niveau.
   *
   * Une reconstruction des patchs sera réalisée afin qu'ils restent conforme.
   * Les patchs devenus vides seront supprimés.
   */
  void removeCellsInZone(const AMRZonePosition& zone_to_delete);

  /*!
   * \brief Méthode permettant de supprimer les patchs en attente de suppression.
   *
   * \param remove_empty_patches (AMR-Type=1 uniquement) Suppression des
   * patchs vides.
   * \param update_higher_level (AMR-Type=3 uniquement) Après une suppression,
   * le niveau le plus haut peut devenir vide. Il est alors nécessaire de
   * mettre ce paramètre à true pour mettre à jour les couches de mailles de
   * recouvrement. Les mailles devenues inutiles seront supprimées.
   */
  void applyPatchEdit(bool remove_empty_patches, bool update_higher_level);

  /*!
   * \brief Méthode permettant de remonter tous les patchs d'un niveau, de
   * mettre à jour le niveau ground et de créer le nouveau niveau 1.
   *
   * À appeler une fois les mailles de niveau 0 créées.
   */
  void updateLevelsAndAddGroundPatch();

  void mergePatches();

  void beginAdaptMesh(Int32 nb_levels, Int32 level_to_refine_first);
  void endAdaptMesh();
  void adaptLevel(Int32 level_to_adapt, bool do_fatal_if_useless);

  void clearRefineRelatedFlags() const;

  void rebuildAvailableGroupIndex(ConstArrayView<Integer> available_group_index);

  ConstArrayView<Int32> availableGroupIndex();

  void setOverlapLayerSizeTopLevel(Integer size_of_overlap_layer_top_level);
  Int32 overlapLayerSize(Int32 level);

 private:

  Integer _addPatchAfterRestore(CellGroup cell_group);
  void _addPatch(CellGroup cell_group, Integer group_index);

  void _increaseOverlapSizeLevel(Int32 level_to_increate, Int32 new_size);
  void _reduceOverlapSizeLevel(Int32 level_to_reduce, Int32 new_size);

  void _updateHigherLevel();

  void _changeOverlapSizeLevel(Int32 level, Int32 previous_higher_level, Int32 new_higher_level);

  void _coarsenUselessCells(bool use_cells_level);
  void _coarsenUselessCellsInLevel(Int32 level);

  void _updatePatchFlagsOfItemsLevel(Int32 level, bool use_cell_groups);
  void _updatePatchFlagsOfItemsGroundLevel();
  void _removePatchFlagsOfItemsLevel(Int32 level);

  void _checkPatchesAndMesh();

  void _removeCellsInAllPatches(const AMRPatchPosition& zone_to_delete);

  Integer _nextIndexForNewPatch();

  void _addPatchInstance(Ref<CartesianMeshPatch> v);

  void _removeOnePatch(Integer index);
  void _removeMultiplePatches(ConstArrayView<Integer> indexes);
  void _removeAllPatches();
  void _createGroundPatch();

  Integer _addCellGroup(CellGroup cell_group, CartesianMeshPatch* patch, bool add_flags);
  void _updateCellGroups(Integer index, bool update_flags);

  void _removePartOfPatch(Integer index_patch_to_edit, const AMRPatchPosition& patch_position);
  void _addCutPatch(const AMRPatchPosition& new_patch_position, CellGroup parent_patch_cell_group);
  Integer _addPatch(const AMRPatchPosition& new_patch_position);

 private:

  UniqueArray<CellGroup> m_amr_patch_cell_groups_all;
  UniqueArray<CellGroup> m_amr_patch_cell_groups_inpatch;
  UniqueArray<CellGroup> m_amr_patch_cell_groups_overlap;
  UniqueArray<ICartesianMeshPatch*> m_amr_patches_pointer;
  UniqueArray<Ref<CartesianMeshPatch>> m_amr_patches;
  ICartesianMesh* m_cmesh;
  UniqueArray<Integer> m_patches_to_delete;
  Int32 m_index_new_patches;
  UniqueArray<Integer> m_available_group_index;
  Integer m_size_of_overlap_layer_top_level;
  Int32 m_higher_level;
  Int32 m_target_nb_levels;
  Int32 m_latest_call_level;
  Ref<Properties> m_properties;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
