// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianPatchGroup.cc                                      (C) 2000-2025 */
/*                                                                           */
/* Gestion du groupe de patchs du maillage cartésien.                        */
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/internal/CartesianPatchGroup.h"

#include "arcane/utils/FixedArray.h"
#include "arcane/utils/Vector3.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/MeshKind.h"
#include "arcane/core/Properties.h"

#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/AMRZonePosition.h"

#include "arcane/cartesianmesh/internal/CartesianMeshPatch.h"
#include "arcane/cartesianmesh/internal/ICartesianMeshInternal.h"
#include "arcane/cartesianmesh/internal/AMRPatchPositionLevelGroup.h"
#include "arcane/cartesianmesh/internal/AMRPatchPositionSignature.h"
#include "arcane/cartesianmesh/internal/AMRPatchPositionSignatureCut.h"
#include "arcane/cartesianmesh/internal/ICartesianMeshNumberingMngInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Le patch 0 est un patch spécial "ground". Il ne possède pas de cell_group
// dans le tableau "m_amr_patch_cell_groups".
// Pour les index, on utilise toujours celui des tableaux m_amr_patches_pointer
// et m_amr_patches.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianPatchGroup::
CartesianPatchGroup(ICartesianMesh* cmesh)
: TraceAccessor(cmesh->traceMng())
, m_cmesh(cmesh)
, m_index_new_patches(1)
, m_size_of_overlap_layer_top_level(0)
, m_higher_level(0)
, m_target_nb_levels(0)
, m_latest_call_level(-1)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
build()
{
  m_properties = makeRef(new Properties(*(m_cmesh->mesh()->properties()), "CartesianPatchGroup"));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
saveInfosInProperties()
{
  m_properties->set("Version", 1);

  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    UniqueArray<String> patch_group_names;
    for (Integer i = 1; i < m_amr_patches_pointer.size(); ++i) {
      patch_group_names.add(allCells(i).name());
    }
    m_properties->set("PatchGroupNames", patch_group_names);
  }

  else {
    UniqueArray<String> patch_group_names(m_amr_patches_pointer.size() - 1);
    UniqueArray<Int32> level(m_amr_patches_pointer.size());
    UniqueArray<Int32> overlap(m_amr_patches_pointer.size());
    UniqueArray<Int32> index(m_amr_patches_pointer.size());
    UniqueArray<CartCoord> min_point(m_amr_patches_pointer.size() * 3);
    UniqueArray<CartCoord> max_point(m_amr_patches_pointer.size() * 3);

    for (Integer i = 0; i < m_amr_patches_pointer.size(); ++i) {
      const AMRPatchPosition& position = m_amr_patches_pointer[i]->_internalApi()->positionRef();
      level[i] = position.level();
      overlap[i] = position.overlapLayerSize();
      index[i] = m_amr_patches_pointer[i]->index();

      const Integer pos = i * 3;
      min_point[pos + 0] = position.minPoint().x;
      min_point[pos + 1] = position.minPoint().y;
      min_point[pos + 2] = position.minPoint().z;
      max_point[pos + 0] = position.maxPoint().x;
      max_point[pos + 1] = position.maxPoint().y;
      max_point[pos + 2] = position.maxPoint().z;

      if (i != 0) {
        patch_group_names[i - 1] = allCells(i).name();
      }
    }
    m_properties->set("LevelPatches", level);
    m_properties->set("OverlapSizePatches", overlap);
    m_properties->set("IndexPatches", index);
    m_properties->set("MinPointPatches", min_point);
    m_properties->set("MaxPointPatches", max_point);

    // TODO : Trouver une autre façon de gérer ça.
    //        Dans le cas d'une protection reprise, le tableau m_available_index
    //        ne peut pas être correctement recalculé à cause des éléments après
    //        le "index max" des "index actif". Ces éléments "en trop" ne
    //        peuvent pas être retrouvés sans plus d'infos.
    m_properties->set("PatchGroupNamesAvailable", m_available_group_index);
    m_properties->set("PatchGroupNames", patch_group_names);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
recreateFromDump()
{
  Trace::Setter mci(traceMng(), "CartesianPatchGroup");

  // Sauve le numéro de version pour être sur que c'est OK en reprise
  Int32 v = m_properties->getInt32("Version");
  if (v != 1) {
    ARCANE_FATAL("Bad serializer version: trying to read from incompatible checkpoint v={0} expected={1}", v, 1);
  }

  clear();

  // Récupère les noms des groupes des patchs
  UniqueArray<String> patch_group_names;
  m_properties->get("PatchGroupNames", patch_group_names);

  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    info(4) << "Found n=" << patch_group_names.size() << " patchs";

    IItemFamily* cell_family = m_cmesh->mesh()->cellFamily();
    for (const String& x : patch_group_names) {
      CellGroup group = cell_family->findGroup(x);
      if (group.null())
        ARCANE_FATAL("Can not find cell group '{0}'", x);
      addPatchAfterRestore(group);
    }
  }
  else {
    UniqueArray<Int32> level;
    UniqueArray<Int32> overlap;
    UniqueArray<Int32> index;
    UniqueArray<CartCoord> min_point;
    UniqueArray<CartCoord> max_point;

    m_properties->get("LevelPatches", level);
    m_properties->get("OverlapSizePatches", overlap);
    m_properties->get("IndexPatches", index);
    m_properties->get("MinPointPatches", min_point);
    m_properties->get("MaxPointPatches", max_point);

    if (index.size() < 1) {
      ARCANE_FATAL("Le ground est forcement save");
    }

    {
      ConstArrayView min(min_point.subConstView(0, 3));
      ConstArrayView max(max_point.subConstView(0, 3));

      AMRPatchPosition position(
      level[0],
      { min[MD_DirX], min[MD_DirY], min[MD_DirZ] },
      { max[MD_DirX], max[MD_DirY], max[MD_DirZ] },
      overlap[0]);

      m_amr_patches_pointer[0]->_internalApi()->setPosition(position);
    }

    IItemFamily* cell_family = m_cmesh->mesh()->cellFamily();

    for (Integer i = 1; i < index.size(); ++i) {
      ConstArrayView min(min_point.subConstView(i * 3, 3));
      ConstArrayView max(max_point.subConstView(i * 3, 3));

      AMRPatchPosition position(
      level[i],
      { min[MD_DirX], min[MD_DirY], min[MD_DirZ] },
      { max[MD_DirX], max[MD_DirY], max[MD_DirZ] },
      overlap[i]);

      const String& x = patch_group_names[i - 1];
      CellGroup cell_group = cell_family->findGroup(x);
      if (cell_group.null())
        ARCANE_FATAL("Can not find cell group '{0}'", x);

      auto* cdi = new CartesianMeshPatch(m_cmesh, index[i], position);
      _addPatchInstance(makeRef(cdi));
      _addCellGroup(cell_group, cdi);
    }

    UniqueArray<Int32> available_index;
    m_properties->get("PatchGroupNamesAvailable", available_index);
    rebuildAvailableGroupIndex(available_index);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<CartesianMeshPatch> CartesianPatchGroup::
groundPatch()
{
  _createGroundPatch();
  return patch(0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
addPatch(ConstArrayView<Int32> cells_local_id)
{
  if (m_cmesh->mesh()->meshKind().meshAMRKind() == eMeshAMRKind::PatchCartesianMeshOnly) {
    ARCANE_FATAL("Do not use this method with AMR type 3");
  }

  Integer index = _nextIndexForNewPatch();
  String children_group_name = String("CartesianMeshPatchCells") + index;
  IItemFamily* cell_family = m_cmesh->mesh()->cellFamily();
  CellGroup children_cells = cell_family->createGroup(children_group_name, cells_local_id, true);
  addPatch(children_cells, index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Il faut appeler rebuildAvailableIndex() après les appels à cette méthode.
Integer CartesianPatchGroup::
addPatchAfterRestore(CellGroup cell_group)
{
  const String& name = cell_group.name();
  Integer group_index = -1;
  if (name.startsWith("CartesianMeshPatchCells")) {
    String index_str = name.substring(23);
    group_index = std::stoi(index_str.localstr());
  }
  else {
    ARCANE_FATAL("Invalid group");
  }

  addPatch(cell_group, group_index);
  return group_index;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
addPatch(CellGroup cell_group, Integer group_index)
{
  _createGroundPatch();
  if (cell_group.null())
    ARCANE_FATAL("Null cell group");

  AMRPatchPosition position;

  auto* cdi = new CartesianMeshPatch(m_cmesh, group_index, position);
  _addPatchInstance(makeRef(cdi));
  _addCellGroup(cell_group, cdi);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
addPatch(const AMRZonePosition& zone_position)
{
  clearRefineRelatedFlags();

  auto amr = m_cmesh->_internalApi()->cartesianMeshAMRPatchMng();
  auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();

  auto position = zone_position.toAMRPatchPosition(m_cmesh);
  Int32 level = position.level();

  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level)) {
    if (!icell->hasHChildren()) {
      const CartCoord3 pos = numbering->cellUniqueIdToCoord(*icell);
      if (position.isInWithOverlap(pos)) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
      }
    }
  }

  amr->refine();

  Int32 higher_level = m_higher_level;
  if (level > higher_level) {
    higher_level = level;
  }
  _addPatch(position.patchUp(m_cmesh->mesh()->dimension(), higher_level, m_size_of_overlap_layer_top_level));
  _updateHigherLevel();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Attention : avant _createGroundPatch() = 0, après _createGroundPatch(); = 1
Integer CartesianPatchGroup::
nbPatch() const
{
  return m_amr_patches.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<CartesianMeshPatch> CartesianPatchGroup::
patch(const Integer index) const
{
  return m_amr_patches[index];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshPatchListView CartesianPatchGroup::
patchListView() const
{
  return CartesianMeshPatchListView{ m_amr_patches_pointer };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellGroup CartesianPatchGroup::
allCells(const Integer index)
{
  if (index == 0) {
    ARCANE_FATAL("You cannot get cells of ground patch with this method");
  }
  return m_amr_patch_cell_groups_all[index - 1];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellGroup CartesianPatchGroup::
inPatchCells(Integer index)
{
  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    ARCANE_FATAL("Method available only with AMR PatchCartesianMeshOnly");
  }
  if (index == 0) {
    ARCANE_FATAL("You cannot get cells of ground patch with this method");
  }
  return m_amr_patch_cell_groups_inpatch[index - 1];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellGroup CartesianPatchGroup::
overlapCells(Integer index)
{
  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    ARCANE_FATAL("Method available only with AMR PatchCartesianMeshOnly");
  }
  if (index == 0) {
    ARCANE_FATAL("You cannot get cells of ground patch with this method");
  }
  return m_amr_patch_cell_groups_overlap[index - 1];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Attention : efface aussi le ground patch. Nécessaire de le récupérer après coup.
void CartesianPatchGroup::
clear()
{
  _removeAllPatches();
  _createGroundPatch();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
removePatch(const Integer index)
{
  if (m_patches_to_delete.contains(index)) {
    return;
  }
  if (index == 0) {
    ARCANE_FATAL("You cannot remove ground patch");
  }
  if (index < 1 || index >= m_amr_patches.size()) {
    ARCANE_FATAL("Invalid index");
  }

  m_patches_to_delete.add(index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
removeCellsInAllPatches(ConstArrayView<Int32> cells_local_id)
{
  if (m_cmesh->mesh()->meshKind().meshAMRKind() == eMeshAMRKind::PatchCartesianMeshOnly) {
    ARCANE_FATAL("Method available only with AMR Cell");
  }
  for (CellGroup cells : m_amr_patch_cell_groups_all) {
    cells.removeItems(cells_local_id);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_removeCellsInAllPatches(const AMRPatchPosition& zone_to_delete)
{
  // Attention si suppression de la suppression en deux étapes : _splitPatch() supprime aussi des patchs.
  // i = 1 car on ne peut pas déraffjner le patch ground.
  const Integer nb_patchs = m_amr_patches_pointer.size();
  for (Integer i = 1; i < nb_patchs; ++i) {
    ICartesianMeshPatch* patch = m_amr_patches_pointer[i];
    // info() << "I : " << i
    //                                     << " -- Compare Patch (min : " << patch->position().minPoint()
    //                                     << ", max : " << patch->position().maxPoint()
    //                                     << ", level : " << patch->position().level()
    //                                     << ") and Zone (min : " << zone_to_delete.minPoint()
    //                                     << ", max : " << zone_to_delete.maxPoint()
    //                                     << ", level : " << zone_to_delete.level() << ")";

    if (zone_to_delete.haveIntersection(patch->position())) {
      _removePartOfPatch(i, zone_to_delete);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
removeCellsInZone(const AMRZonePosition& zone_to_delete)
{
  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    ARCANE_FATAL("Method available only with AMR PatchCartesianMeshOnly");
  }
  clearRefineRelatedFlags();

  UniqueArray<Int32> cells_local_id;

  AMRPatchPosition patch_position;
  zone_to_delete.cellsInPatch(m_cmesh, cells_local_id, patch_position);

  _removeCellsInAllPatches(patch_position);
  applyPatchEdit(false);
  auto amr = m_cmesh->_internalApi()->cartesianMeshAMRPatchMng();
  auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();

  Int32 level = patch_position.level();

  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level)) {
    if (!icell->hasHChildren()) {
      const CartCoord3 pos = numbering->cellUniqueIdToCoord(*icell);
      if (patch_position.isIn(pos)) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Coarsen);
      }
    }
  }

  amr->coarsen(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
applyPatchEdit(bool remove_empty_patches)
{
  // m_cmesh->mesh()->traceMng()->info() << "applyPatchEdit() -- Remove nb patch : " << m_patches_to_delete.size();

  std::stable_sort(m_patches_to_delete.begin(), m_patches_to_delete.end(),
                   [](const Integer a, const Integer b) {
                     return a < b;
                   });

  _removeMultiplePatches(m_patches_to_delete);
  m_patches_to_delete.clear();

  if (remove_empty_patches) {
    if (m_cmesh->mesh()->meshKind().meshAMRKind() == eMeshAMRKind::PatchCartesianMeshOnly) {
      ARCANE_FATAL("remove_empty_patches=true available only with AMR Cell");
    }
    UniqueArray<Integer> size_of_patches(m_amr_patch_cell_groups_all.size());
    for (Integer i = 0; i < m_amr_patch_cell_groups_all.size(); ++i) {
      size_of_patches[i] = m_amr_patch_cell_groups_all[i].size();
    }
    m_cmesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMax, size_of_patches);
    for (Integer i = 0; i < size_of_patches.size(); ++i) {
      if (size_of_patches[i] == 0) {
        m_patches_to_delete.add(i + 1);
      }
    }
    _removeMultiplePatches(m_patches_to_delete);
    m_patches_to_delete.clear();
  }

  if (m_cmesh->mesh()->meshKind().meshAMRKind() == eMeshAMRKind::PatchCartesianMeshOnly) {
    _updateHigherLevel();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
updateLevelsAndAddGroundPatch()
{
  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    return;
  }
  auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();

  // Attention : on suppose que numbering->updateFirstLevel(); a déjà été appelé !

  for (ICartesianMeshPatch* patch : m_amr_patches_pointer) {
    const Int32 level = patch->position().level();
    // Si le niveau est 0, c'est le patch spécial 0 donc on ne modifie que le max, le niveau reste à 0.
    if (level == 0) {
      const CartCoord3 max_point = patch->position().maxPoint();
      if (m_cmesh->mesh()->dimension() == 2) {
        patch->_internalApi()->positionRef().setMaxPoint({
        numbering->offsetLevelToLevel(max_point.x, level, level - 1),
        numbering->offsetLevelToLevel(max_point.y, level, level - 1),
        1,
        });
      }
      else {
        patch->_internalApi()->positionRef().setMaxPoint({
        numbering->offsetLevelToLevel(max_point.x, level, level - 1),
        numbering->offsetLevelToLevel(max_point.y, level, level - 1),
        numbering->offsetLevelToLevel(max_point.z, level, level - 1),
        });
      }
    }
    // Sinon, on "surélève" le niveau des patchs vu qu'il va y avoir le patch "-1"
    else {
      patch->_internalApi()->positionRef().setLevel(level + 1);
    }
  }

  AMRPatchPosition old_ground;
  old_ground.setLevel(1);
  old_ground.setMinPoint({ 0, 0, 0 });
  old_ground.setMaxPoint({ numbering->globalNbCellsX(1), numbering->globalNbCellsY(1), numbering->globalNbCellsZ(1) });
  old_ground.computeOverlapLayerSize(m_higher_level + 1, m_size_of_overlap_layer_top_level);

  _addPatch(old_ground);
  _updateHigherLevel();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer CartesianPatchGroup::
_nextIndexForNewPatch()
{
  if (!m_available_group_index.empty()) {
    const Integer elem = m_available_group_index.back();
    m_available_group_index.popBack();
    return elem;
  }
  return m_index_new_patches++;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
mergePatches()
{
  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    return;
  }
  // info() << "Global fusion";
  UniqueArray<std::pair<Integer, Int64>> index_n_nb_cells;
  {
    Integer index = 0;
    for (auto patch : m_amr_patches_pointer) {
      index_n_nb_cells.add({ index++, patch->position().nbCells() });
    }
  }

  // Algo de fusion.
  // D'abord, on trie les patchs du plus petit nb de mailles au plus grand nb de mailles (optionnel).
  // Ensuite, pour chaque patch, on regarde si l'on peut le fusionner avec un autre.
  // Si on arrive à faire une fusion, on recommence l'algo jusqu'à ne plus pouvoir fusionner.
  bool fusion = true;
  while (fusion) {
    fusion = false;

    std::stable_sort(index_n_nb_cells.begin(), index_n_nb_cells.end(),
                     [](const std::pair<Integer, Int64>& a, const std::pair<Integer, Int64>& b) {
                       return a.second < b.second;
                     });

    for (Integer p0 = 0; p0 < index_n_nb_cells.size(); ++p0) {
      auto [index_p0, nb_cells_p0] = index_n_nb_cells[p0];

      AMRPatchPosition& patch_fusion_0 = m_amr_patches_pointer[index_p0]->_internalApi()->positionRef();
      if (patch_fusion_0.isNull())
        continue;

      // Si une fusion a déjà eu lieu, on doit alors regarder les patchs avant "p0"
      // (vu qu'il y en a au moins un qui a été modifié).
      // (une "optimisation" pourrait être de récupérer la position du premier
      // patch fusionné mais bon, moins lisible + pas beaucoup de patchs).
      for (Integer p1 = p0 + 1; p1 < m_amr_patches_pointer.size(); ++p1) {
        auto [index_p1, nb_cells_p1] = index_n_nb_cells[p1];

        AMRPatchPosition& patch_fusion_1 = m_amr_patches_pointer[index_p1]->_internalApi()->positionRef();
        if (patch_fusion_1.isNull())
          continue;

        // info() << "\tCheck fusion"
        //                                     << " -- 0 Min point : " << patch_fusion_0.minPoint()
        //                                     << " -- 0 Max point : " << patch_fusion_0.maxPoint()
        //                                     << " -- 0 Level : " << patch_fusion_0.level()
        //                                     << " -- 1 Min point : " << patch_fusion_1.minPoint()
        //                                     << " -- 1 Max point : " << patch_fusion_1.maxPoint()
        //                                     << " -- 1 Level : " << patch_fusion_1.level();

        if (patch_fusion_0.fusion(patch_fusion_1)) {
          // info() << "Fusion OK";
          index_n_nb_cells[p0].second = patch_fusion_0.nbCells();

          UniqueArray<Int32> local_ids;
          allCells(index_p1).view().fillLocalIds(local_ids);
          allCells(index_p0).addItems(local_ids, false);

          // info() << "Remove patch : " << index_p1;
          removePatch(index_p1);

          fusion = true;
          break;
        }
      }
      if (fusion) {
        break;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
beginAdaptMesh(Int32 nb_levels, Int32 level_to_refine_first)
{
  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    ARCANE_FATAL("Method available only with AMR PatchCartesianMeshOnly");
  }
  if (m_latest_call_level != -1) {
    ARCANE_FATAL("Call endAdaptMesh() before restart mesh adaptation");
  }

  Trace::Setter mci(traceMng(), "CartesianPatchGroup");
  info() << "Begin adapting mesh with higher level = " << (nb_levels - 1);

  // On doit adapter tous les niveaux sous le niveau à adapter.
  // Les patchs du niveau "level_to_refine_first" (exclus) et plus seront supprimés.
  if (nb_levels - 1 != m_higher_level) {
    debug() << "beginAdaptMesh() -- First call -- Change overlap layer size -- Old higher level : " << m_higher_level
            << " -- Asked higher level : " << (nb_levels - 1)
            << " -- Adapt level lower than : " << level_to_refine_first;

    for (Int32 level = 1; level <= level_to_refine_first; ++level) {
      _changeOverlapSizeLevel(level, m_higher_level, nb_levels - 1);
    }
  }

  // On supprime tous les patchs au-dessus du premier niveau à raffiner.
  Int32 max_level = 0;
  for (Integer p = 1; p < m_amr_patches_pointer.size(); ++p) {
    Int32 level = m_amr_patches_pointer[p]->_internalApi()->positionRef().level();
    if (level > level_to_refine_first) {
      removePatch(p);
      max_level = level;
    }
  }
  applyPatchEdit(false);

  // On enlève aussi les flags II_InPatch et II_Overlap des mailles pour que
  // celles qui ne sont plus utilisées par la suite dans un des nouveaux
  // patchs soit supprimées dans la méthode finalizeAdaptMesh().
  for (Integer l = level_to_refine_first + 1; l <= max_level; ++l) {
    ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(l)) {
      icell->mutableItemBase().removeFlags(ItemFlags::II_Overlap | ItemFlags::II_InPatch);
    }
  }

  m_target_nb_levels = nb_levels;
  m_latest_call_level = level_to_refine_first;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
endAdaptMesh()
{
  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    ARCANE_FATAL("Method available only with AMR PatchCartesianMeshOnly");
  }
  if (m_latest_call_level == -1) {
    ARCANE_FATAL("Call beginAdaptMesh() before");
  }
  Trace::Setter mci(traceMng(), "CartesianPatchGroup");
  info() << "Finalizing adapting mesh with higher level = " << (m_target_nb_levels - 1);

  auto amr = m_cmesh->_internalApi()->cartesianMeshAMRPatchMng();

  // Le plus haut niveau devient le dernier niveau adapté (+1 pour avoir le
  // niveau raffiné).
  // On est sûr que c'est le niveau le plus haut étant donné que l'on supprime
  // systématiquement les patchs au-dessus dans adaptLevel().
  m_higher_level = m_latest_call_level + 1;

  // Si m_latest_call_level == 0, alors adaptLevel() a créée le niveau 1 donc
  // il y a 2 niveaux.
  // Si le niveau le plus haut créé est inférieur au niveau le plus haut donné
  // par l'utilisateur dans la méthode beginAdaptMesh(), on est obligé de
  // réadapter le nombre de couche de mailles de recouvrement pour chaque
  // patch.
  if (m_higher_level + 1 < m_target_nb_levels) {
    info() << "Reduce higher level from " << (m_target_nb_levels - 1) << " to " << m_higher_level;

    for (Int32 level = 1; level <= m_higher_level; ++level) {
      _changeOverlapSizeLevel(level, m_target_nb_levels - 1, m_higher_level);
    }
  }

  // On doit utiliser le niveau des mailles et non des patchs car il peut y
  // avoir des mailles qui ne sont plus dans des patchs.
  Int32 max_level = 0;
  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allCells()) {
    if (icell->level() > max_level) {
      max_level = icell->level();
    }
  }
  max_level = m_cmesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMax, max_level);

  debug() << "Max level of cells : " << max_level;

  // On supprime les mailles qui ne sont pas/plus dans un patch.
  for (Integer level = max_level; level > 0; --level) {
    Integer nb_cells_to_coarse = 0;
    ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level)) {
      if (!icell->hasFlags(ItemFlags::II_InPatch) && !icell->hasFlags(ItemFlags::II_Overlap)) {
        //debug() << "Coarse CellUID : " << icell->uniqueId();
        icell->mutableItemBase().addFlags(ItemFlags::II_Coarsen);
        nb_cells_to_coarse++;
      }
    }
    debug() << "Remove " << nb_cells_to_coarse << " refined cells without flag in level " << level;
    nb_cells_to_coarse = m_cmesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMax, nb_cells_to_coarse);
    if (nb_cells_to_coarse != 0) {
      amr->coarsen(true);
    }
  }

  m_target_nb_levels = 0;
  m_latest_call_level = -1;
  clearRefineRelatedFlags();

  info() << "Patch list:";

  for (Integer i = 0; i <= m_higher_level; ++i) {
    for (auto p : m_amr_patches_pointer) {
      auto& position = p->_internalApi()->positionRef();
      if (position.level() == i) {
        info() << "\tPatch #" << p->index()
               << " -- Level : " << position.level()
               << " -- Min point : " << position.minPoint()
               << " -- Max point : " << position.maxPoint()
               << " -- Overlap layer size : " << position.overlapLayerSize();
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
adaptLevel(Int32 level_to_adapt)
{
  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    ARCANE_FATAL("Method available only with AMR PatchCartesianMeshOnly");
  }

  if (m_latest_call_level == -1) {
    ARCANE_FATAL("Call beginAdaptMesh() before to begin a mesh adaptation");
  }

  Trace::Setter mci(traceMng(), "CartesianPatchGroup");

  if (level_to_adapt + 1 >= m_target_nb_levels || level_to_adapt < 0) {
    ARCANE_FATAL("Bad level to adapt -- Level to adapt : {0} (creating level {1}) -- Max nb levels : {2}", level_to_adapt, level_to_adapt + 1, m_target_nb_levels);
  }

  // On supprime tous les patchs au-dessus du niveau que l'on souhaite adapter.
  // On le fait aussi ici dans le cas où l'utilisateur appelle cette méthode
  // avec un niveau inférieur à son précedent appel (ce qui n'est pas
  // forcément optimal vu qu'on supprime ce qui a été calculé
  // précédemment...).
  if (level_to_adapt < m_latest_call_level) {
    Int32 max_level = 0;
    for (Integer p = 1; p < m_amr_patches_pointer.size(); ++p) {
      Int32 level = m_amr_patches_pointer[p]->_internalApi()->positionRef().level();
      if (level > level_to_adapt) {
        removePatch(p);
        max_level = level;
      }
    }
    applyPatchEdit(false);

    for (Integer l = level_to_adapt + 1; l <= max_level; ++l) {
      ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(l)) {
        icell->mutableItemBase().removeFlags(ItemFlags::II_Overlap | ItemFlags::II_InPatch);
      }
    }
  }

  m_latest_call_level = level_to_adapt;
  auto amr = m_cmesh->_internalApi()->cartesianMeshAMRPatchMng();
  auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();

  // Le nombre de couches de mailles de recouvrements.
  // +1 car les patchs créés seront de niveau level_to_adapt + 1.
  // /pattern car les mailles à raffiner sont sur le niveau level_to_adapt.
  //
  // Explication :
  //  level_to_adapt=0,
  //  les futurs patchs seront de niveau 1, donc le nombre de couches de
  //  recouvrement doit être celui correspondant au niveau 1
  //  (donc level_to_adapt+1),
  //  or, les mailles à raffiner sont de niveau 0, donc on doit diviser le
  //  nombre de couches par le nombre de mailles enfants qui seront créées
  //  (pour une dimension) (donc numbering->pattern()).
  Int32 nb_overlap_cells = overlapLayerSize(level_to_adapt + 1) / numbering->pattern();

  info() << "adaptLevel()"
         << " -- level_to_adapt : " << level_to_adapt
         << " -- nb_overlap_cells : " << nb_overlap_cells;

  // Deux vérifications :
  // - on ne peut pas raffiner plusieurs niveaux d'un coup,
  // - on ne peut pas raffiner des mailles qui ne sont pas dans un patch (les
  // mailles de recouvrements ne sont pas forcément dans un patch).
  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allCells()) {
    if (icell->hasFlags(ItemFlags::II_Refine)) {
      if (icell->level() != level_to_adapt) {
        ARCANE_FATAL("Flag II_Refine found on Cell (UID={0} - Level={1}) not in level to refine (={2})", icell->uniqueId(), icell->level(), level_to_adapt);
      }
      if (level_to_adapt != 0 && !icell->hasFlags(ItemFlags::II_InPatch)) {
        const CartCoord3 pos = numbering->cellUniqueIdToCoord(*icell);
        ARCANE_FATAL("Cannot refine cell not in patch -- Pos : {0} -- CellUID : {1} -- CellLevel : {2}", pos, icell->uniqueId(), icell->level());
      }
    }
  }

  UniqueArray<AMRPatchPositionSignature> sig_array;

  // On doit donner un ou plusieurs patchs initiaux, pour être réduit et
  // découpé.
  // Si le niveau à adapter est le niveau 0, on peut créer un patch initial
  // qui fait la taille du patch ground.
  // On n'a pas besoin de le réduire, AMRPatchPositionSignature::fillSig()
  // s'en occupera.
  if (level_to_adapt == 0) {
    AMRPatchPosition all_level;
    all_level.setLevel(level_to_adapt);
    all_level.setMinPoint({ 0, 0, 0 });
    all_level.setMaxPoint({ numbering->globalNbCellsX(level_to_adapt), numbering->globalNbCellsY(level_to_adapt), numbering->globalNbCellsZ(level_to_adapt) });
    // Pour ce setOverlapLayerSize(), voir l'explication au-dessus.
    all_level.setOverlapLayerSize(nb_overlap_cells);
    AMRPatchPositionSignature sig(all_level, m_cmesh);
    sig_array.add(sig);
  }

  // Pour les autres niveaux, on crée les patchs initiaux en copiant les
  // patchs du niveau level_to_adapt.
  // On ne peut pas créer un patch qui fait la taille du niveau level_to_adapt
  // car il est impératif que le ou les patchs générés par
  // AMRPatchPositionSignatureCut (futurs patchs du niveau level_to_adapt+1)
  // soit inclus dans le ou les patchs du niveau level_to_adapt ! (sinon on
  // aurait des mailles orphelines).
  // On peut prendre ces patchs comme patchs initiaux car on sait que les
  // seules mailles que l'on aura à raffiner sont dans le ou les patchs du
  // niveau level_to_adapt (les mailles ayant le flag II_InPatch).
  // On sait aussi que les méthodes de AMRPatchPositionSignatureCut ne peuvent
  // pas agrandir les patchs initiaux (uniquement réduire ou couper).
  // (et oui, le if(level_to_adapt == 0) n'est pas indispensable, mais comme
  // on sait qu'il y a qu'un seul patch de niveau 0, c'est plus rapide).
  else {
    for (auto patch : m_amr_patches_pointer) {
      Integer level = patch->_internalApi()->positionRef().level();
      if (level == level_to_adapt) {
        auto position = patch->position();
        position.setOverlapLayerSize(nb_overlap_cells);
        AMRPatchPositionSignature sig(position, m_cmesh);
        sig_array.add(sig);
      }
    }
  }

  AMRPatchPositionSignatureCut::cut(sig_array);

  // Une fois les patchs découpés, on ajoute le flag II_Refine aux mailles de
  // ces patchs.
  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level_to_adapt)) {
    if (!icell->hasHChildren()) {
      const CartCoord3 pos = numbering->cellUniqueIdToCoord(*icell);
      for (const AMRPatchPositionSignature& patch_signature : sig_array) {
        if (patch_signature.patch().isInWithOverlap(pos)) {
          // TODO Quand le patch ground sera un patch classique, retirer level_to_adapt!=0.
          if (level_to_adapt != 0 && !icell->hasFlags(ItemFlags::II_InPatch) && !icell->hasFlags(ItemFlags::II_Overlap)) {
            ARCANE_FATAL("Internal error -- Refine algo error -- Pos : {0}", pos);
          }
          icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
        }
      }
    }
  }

  {
    //   UniqueArray<CartCoord> out(numbering->globalNbCellsY(level_to_adapt) * numbering->globalNbCellsX(level_to_adapt), -1);
    //   Array2View av_out(out.data(), numbering->globalNbCellsY(level_to_adapt), numbering->globalNbCellsX(level_to_adapt));
    //   ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level_to_adapt)) {
    //     CartCoord3 pos = numbering->cellUniqueIdToCoord(*icell);
    //     if (icell->hasHChildren()) {
    //       av_out(pos.y, pos.x) = 0;
    //     }
    //     if (icell->hasFlags(ItemFlags::II_Refine)) {
    //       av_out(pos.y, pos.x) = 1;
    //     }
    //   }
    //
    //   StringBuilder str = "";
    //   for (CartCoord i = 0; i < numbering->globalNbCellsX(level_to_adapt); ++i) {
    //     str += "\n";
    //     for (CartCoord j = 0; j < numbering->globalNbCellsY(level_to_adapt); ++j) {
    //       CartCoord c = av_out(i, j);
    //       if (c == 1)
    //         str += "[++]";
    //       else if (c == 0)
    //         str += "[XX]";
    //       else
    //         str += "[  ]";
    //     }
    //   }
    //   info() << str;
  }

  // On raffine.
  amr->refine();

  // TODO : Normalement, il n'y a pas besoin de faire ça, à corriger dans amr->refine().
  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level_to_adapt)) {
    icell->mutableItemBase().removeFlags(ItemFlags::II_Refine);
  }

  // Les patchs de sig_array sont des patchs "intermédiaires". Ce sont des
  // patchs de niveau level_to_adapt représentant des patchs de niveau
  // level_to_adapt+1.
  // Il est maintenant nécessaire de les convertir en patch de niveau
  // level_to_adapt+1.
  UniqueArray<AMRPatchPosition> all_patches;
  for (const auto& elem : sig_array) {
    all_patches.add(elem.patch().patchUp(m_cmesh->mesh()->dimension(), m_target_nb_levels - 1, m_size_of_overlap_layer_top_level));
  }

  // On fusionne les patchs qui peuvent l'être avant de les "ajouter" dans le maillage.
  AMRPatchPositionLevelGroup::fusionPatches(all_patches, true);

  for (const AMRPatchPosition& patch : all_patches) {
    debug() << "\tPatch AAA"
            << " -- Level : " << patch.level()
            << " -- Min point : " << patch.minPoint()
            << " -- Max point : " << patch.maxPoint()
            << " -- overlapLayerSize : " << patch.overlapLayerSize();
  }

  // On ajoute les flags sur les mailles des patchs.
  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level_to_adapt + 1)) {
    bool in_overlap = false;
    bool in_patch = false;

    // Si une maille est dans un patch, elle prend le flag II_InPatch.
    // Si une maille est une maille de recouvrement pour un patch, elle prend
    // le flag II_Overlap.
    // Comme son nom l'indique, une maille de recouvrement peut recouvrir un
    // autre patch. Donc une maille peut être à la fois II_InPatch et
    // II_Overlap.
    const CartCoord3 pos = numbering->cellUniqueIdToCoord(*icell);
    for (const AMRPatchPosition& patch : all_patches) {
      if (patch.isIn(pos)) {
        in_patch = true;
      }
      else if (patch.isInWithOverlap(pos)) {
        in_overlap = true;
      }
      if (in_patch && in_overlap) {
        break;
      }
    }
    if (in_patch && in_overlap) {
      icell->mutableItemBase().addFlags(ItemFlags::II_Overlap);
      icell->mutableItemBase().addFlags(ItemFlags::II_InPatch);
    }
    else if (in_overlap) {
      icell->mutableItemBase().addFlags(ItemFlags::II_Overlap);
      icell->mutableItemBase().removeFlags(ItemFlags::II_InPatch); //Au cas où.
    }
    else if (in_patch) {
      icell->mutableItemBase().addFlags(ItemFlags::II_InPatch);
      icell->mutableItemBase().removeFlags(ItemFlags::II_Overlap); //Au cas où.
    }
    else {
      icell->mutableItemBase().removeFlags(ItemFlags::II_InPatch); //Au cas où.
      icell->mutableItemBase().removeFlags(ItemFlags::II_Overlap); //Au cas où.
    }
  }

  {
    //   UniqueArray<CartCoord> out(numbering->globalNbCellsY(level_to_adapt + 1) * numbering->globalNbCellsX(level_to_adapt + 1), -1);
    //   Array2View av_out(out.data(), numbering->globalNbCellsY(level_to_adapt + 1), numbering->globalNbCellsX(level_to_adapt + 1));
    //   ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level_to_adapt + 1)) {
    //
    //     CartCoord3 pos = numbering->cellUniqueIdToCoord(*icell);
    //     Integer patch = -1;
    //
    //     if (icell->hasFlags(ItemFlags::II_Overlap) && icell->hasFlags(ItemFlags::II_InPatch)) {
    //       patch = -2;
    //     }
    //     else if (icell->hasFlags(ItemFlags::II_Overlap)) {
    //       patch = -3;
    //     }
    //     else if (icell->hasFlags(ItemFlags::II_InPatch)) {
    //       patch = -4;
    //     }
    //     else {
    //       patch = -5;
    //     }
    //     av_out(pos.y, pos.x) = patch;
    //     if (icell->uniqueId() == 3310) {
    //       info() << "Maille présente ! -- Coord : " << pos << " -- Flags : " << patch;
    //     }
    //   }
    //
    //   StringBuilder str = "";
    //   for (CartCoord i = 0; i < numbering->globalNbCellsX(level_to_adapt + 1); ++i) {
    //     str += "\n";
    //     for (CartCoord j = 0; j < numbering->globalNbCellsY(level_to_adapt + 1); ++j) {
    //       CartCoord c = av_out(i, j);
    //       if (c >= 0) {
    //         str += "[";
    //         if (c < 10)
    //           str += " ";
    //         str += c;
    //         str += "]";
    //       }
    //       else if (c == -2) {
    //         str += "[OI]";
    //       }
    //       else if (c == -3) {
    //         str += "[OO]";
    //       }
    //       else if (c == -4) {
    //         str += "[II]";
    //       }
    //       else if (c == -5) {
    //         str += "[XX]";
    //       }
    //       else
    //         str += "[  ]";
    //     }
    //   }
    //   info() << str;
  }

  // On ajoute les patchs au maillage (on crée les groupes) et on calcule les
  // directions pour chacun d'eux (pour que l'utilisateur puisse les utiliser
  // directement).
  for (const AMRPatchPosition& patch : all_patches) {
    Integer index = _addPatch(patch);
    // TODO : Mais alors pas une bonne idée du tout !
    m_cmesh->computeDirectionsPatchV2(index + 1);
  }

  debug() << "adaptLevel() -- End call -- Actual patch list:";

  for (Integer i = 0; i <= m_target_nb_levels; ++i) {
    for (auto p : m_amr_patches_pointer) {
      auto& position = p->_internalApi()->positionRef();
      if (position.level() == i) {
        debug() << "\tPatch #" << p->index()
                << " -- Level : " << position.level()
                << " -- Min point : " << position.minPoint()
                << " -- Max point : " << position.maxPoint()
                << " -- Overlap layer size : " << position.overlapLayerSize();
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_increaseOverlapSizeLevel(Int32 level_to_increate, Int32 new_size)
{
  if (level_to_increate == 0) {
    ARCANE_FATAL("Level 0 has not overlap layer");
  }

  auto amr = m_cmesh->_internalApi()->cartesianMeshAMRPatchMng();
  auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();

  bool has_cell_to_refine = false;

  // Trois grandes étapes :
  // - d'abord, on agrandit le nombre de couches dans les structures position,
  //   puis on ajoute le flag II_Refine aux mailles parentes qui n'ont pas
  //   d'enfant,
  // - on raffine les mailles,
  // - on ajoute les flags aux nouvelles mailles et on les ajoute aux groupes
  //   de mailles des patchs.
  for (Integer p = 1; p < m_amr_patches_pointer.size(); ++p) {
    Int32 level = m_amr_patches_pointer[p]->_internalApi()->positionRef().level();
    if (level == level_to_increate) {
      AMRPatchPosition& position = m_amr_patches_pointer[p]->_internalApi()->positionRef();

      Int32 size_layer = position.overlapLayerSize();
      if (size_layer > new_size) {
        ARCANE_FATAL("Cannot reduce layer with _increaseOverlapSizeLevel method");
      }

      // On pourrait vérifier que le nombre de couches de tous les patchs d'un
      // niveau est identique.

      if (size_layer == new_size) {
        continue;
      }

      has_cell_to_refine = true;
      position.setOverlapLayerSize(new_size);

      // Les mailles à raffiner sont sur le niveau inférieur.
      // Pour chaque maille, pour savoir si l'on doit la raffiner ou non, on
      // la monte d'un niveau et on regarde si elle est dans les couches de
      // recouvrement.
      ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level_to_increate - 1)) {
        const CartCoord3 pos = numbering->offsetLevelToLevel(numbering->cellUniqueIdToCoord(*icell), level_to_increate - 1, level_to_increate);
        if (position.isInWithOverlap(pos) && !icell->hasHChildren()) {
          icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
        }
      }
    }
  }
  has_cell_to_refine = m_cmesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMax, has_cell_to_refine);
  if (!has_cell_to_refine) {
    return;
  }

  // On raffine les mailles.
  amr->refine();

  // TODO : Normalement, il n'y a pas besoin de faire ça, à corriger dans amr->refine().
  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level_to_increate - 1)) {
    icell->mutableItemBase().removeFlags(ItemFlags::II_Refine);
  }

  UniqueArray<Int32> cell_to_add;

  // Ajoute les flags et on actualise les groupes de mailles des patchs.
  for (Integer p = 1; p < m_amr_patches_pointer.size(); ++p) {
    Int32 level = m_amr_patches_pointer[p]->_internalApi()->positionRef().level();
    if (level == level_to_increate) {
      AMRPatchPosition& position = m_amr_patches_pointer[p]->_internalApi()->positionRef();

      ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level_to_increate)) {
        if (!icell->hasFlags(ItemFlags::II_JustAdded))
          continue;

        const CartCoord3 pos = numbering->cellUniqueIdToCoord(*icell);

        if (position.isIn(pos)) {
          icell->mutableItemBase().addFlags(ItemFlags::II_InPatch);
        }
        else if (position.isInWithOverlap(pos)) {
          cell_to_add.add(icell.localId());
          icell->mutableItemBase().addFlags(ItemFlags::II_Overlap);
        }
      }

      m_amr_patch_cell_groups_all[p - 1].addItems(cell_to_add, true); //TODO Normalement, mettre check = false
      m_amr_patch_cell_groups_overlap[p - 1].addItems(cell_to_add, true);
      cell_to_add.clear();

      // On calcule les directions pour que le patch soit utilisable.
      m_cmesh->computeDirectionsPatchV2(p);
    }
  }

  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level_to_increate)) {
    icell->mutableItemBase().removeFlags(ItemFlags::II_JustAdded);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_reduceOverlapSizeLevel(Int32 level_to_reduce, Int32 new_size)
{
  // Attention : Le reduce est possible car on ne supprime pas de mailles, on
  // leur enlève leurs flags InPatch/Overlap pour pouvoir les supprimer
  // ensuite.
  // On les enlève uniquement des groupes de mailles des patchs.
  // Il est donc nécessaire d'avoir une autre méthode après celle-ci pour
  // supprimer les mailles sans flags.

  if (level_to_reduce == 0) {
    ARCANE_FATAL("Level 0 has not overlap layer");
  }

  auto amr = m_cmesh->_internalApi()->cartesianMeshAMRPatchMng();
  auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();

  // Deux étapes :
  // - d'abord, on actualise les structures position des patchs puis on
  //   supprime des groupes de mailles des patchs les mailles qui ne sont plus
  //   dans les couches de recouvrement,
  // - enfin, on recalcule les flags de tout le niveau.
  bool has_cell_to_mark = false;
  UniqueArray<Int32> cell_to_remove;

  for (Integer p = 1; p < m_amr_patches_pointer.size(); ++p) {
    Int32 level = m_amr_patches_pointer[p]->_internalApi()->positionRef().level();
    if (level == level_to_reduce) {
      AMRPatchPosition& position = m_amr_patches_pointer[p]->_internalApi()->positionRef();

      Int32 size_layer = position.overlapLayerSize();
      if (size_layer < new_size) {
        ARCANE_FATAL("Cannot add layer with _reduceOverlapSizeLevel method");
      }
      if (size_layer == new_size) {
        continue;
      }

      has_cell_to_mark = true;
      position.setOverlapLayerSize(new_size);

      ENUMERATE_ (Cell, icell, m_amr_patch_cell_groups_overlap[p - 1]) {
        const CartCoord3 pos = numbering->cellUniqueIdToCoord(*icell);
        if (!position.isInWithOverlap(pos)) {
          cell_to_remove.add(icell.localId());
        }
      }

      m_amr_patch_cell_groups_all[p - 1].removeItems(cell_to_remove, true); //TODO Normalement, mettre check = false
      m_amr_patch_cell_groups_overlap[p - 1].removeItems(cell_to_remove, true);
      cell_to_remove.clear();
    }
  }
  has_cell_to_mark = m_cmesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMax, has_cell_to_mark);
  if (!has_cell_to_mark) {
    return;
  }

  // À cause du mélange des deux flags, on doit recalculer les flags.
  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level_to_reduce)) {
    icell->mutableItemBase().removeFlags(ItemFlags::II_InPatch | ItemFlags::II_Overlap);
  }

  for (Integer p = 1; p < m_amr_patches_pointer.size(); ++p) {
    Int32 level = m_amr_patches_pointer[p]->_internalApi()->positionRef().level();
    if (level == level_to_reduce) {
      ENUMERATE_ (Cell, icell, m_amr_patch_cell_groups_inpatch[p - 1]) {
        icell->mutableItemBase().addFlags(ItemFlags::II_InPatch);
      }
      ENUMERATE_ (Cell, icell, m_amr_patch_cell_groups_overlap[p - 1]) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Overlap);
      }
      m_cmesh->computeDirectionsPatchV2(p);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_updateHigherLevel()
{
  // On regarde quel est le patch le plus haut.
  Int32 new_higher_level = 0;
  for (const auto patch : m_amr_patches_pointer) {
    const Int32 level = patch->_internalApi()->positionRef().level();
    if (level > new_higher_level) {
      new_higher_level = level;
    }
  }

  if (new_higher_level == m_higher_level) {
    return;
  }

  auto amr = m_cmesh->_internalApi()->cartesianMeshAMRPatchMng();

  for (Int32 level = 1; level <= new_higher_level; ++level) {
    _changeOverlapSizeLevel(level, m_higher_level, new_higher_level);
  }
  m_higher_level = new_higher_level;

  // Attention : Par rapport à endAdaptMesh(), on ne regarde pas si des
  // mailles sont au-dessus de m_higher_level !

  // On supprime les mailles qui ne sont pas/plus dans un patch.
  for (Integer level = m_higher_level; level > 0; --level) {
    Integer nb_cells_to_coarse = 0;
    ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level)) {
      if (!icell->hasFlags(ItemFlags::II_InPatch) && !icell->hasFlags(ItemFlags::II_Overlap)) {
        //debug() << "Coarse CellUID : " << icell->uniqueId();
        icell->mutableItemBase().addFlags(ItemFlags::II_Coarsen);
        nb_cells_to_coarse++;
      }
    }
    debug() << "Remove " << nb_cells_to_coarse << " refined cells without flag in level " << level;
    nb_cells_to_coarse = m_cmesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMax, nb_cells_to_coarse);
    if (nb_cells_to_coarse != 0) {
      amr->coarsen(true);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_changeOverlapSizeLevel(Int32 level, Int32 previous_higher_level, Int32 new_higher_level)
{
  if (previous_higher_level == new_higher_level) {
    return;
  }

  Int32 old_overlap_size = AMRPatchPosition::computeOverlapLayerSize(level, previous_higher_level, m_size_of_overlap_layer_top_level);
  Int32 new_overlap_size = AMRPatchPosition::computeOverlapLayerSize(level, new_higher_level, m_size_of_overlap_layer_top_level);

  if (old_overlap_size == new_overlap_size) {
    return;
  }
  if (old_overlap_size < new_overlap_size) {
    _increaseOverlapSizeLevel(level, new_overlap_size);
  }
  else {
    _reduceOverlapSizeLevel(level, new_overlap_size);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
clearRefineRelatedFlags() const
{
  constexpr ItemFlags::FlagType flags_to_remove = (ItemFlags::II_Coarsen | ItemFlags::II_Refine |
                                                   ItemFlags::II_JustCoarsened | ItemFlags::II_JustRefined |
                                                   ItemFlags::II_JustAdded | ItemFlags::II_CoarsenInactive);
  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allCells()) {
    icell->mutableItemBase().removeFlags(flags_to_remove);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
rebuildAvailableGroupIndex(ConstArrayView<Integer> available_group_index)
{
  m_available_group_index = available_group_index;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> CartesianPatchGroup::
availableGroupIndex()
{
  return m_available_group_index;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
setOverlapLayerSizeTopLevel(Int32 size_of_overlap_layer_top_level)
{
  // On s'assure que la taille fournie par l'utilisateur est un multiple de
  // pattern (2 aujourd'hui).
  auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();
  Int32 new_size_of_overlap_layer_top_level = size_of_overlap_layer_top_level + (size_of_overlap_layer_top_level % numbering->pattern());

  if (new_size_of_overlap_layer_top_level == m_size_of_overlap_layer_top_level) {
    return;
  }

  // S'il y a changement de la taille de la couche du niveau le plus haut, il
  // y aura un changement de taille sur les autres niveaux.
  for (Int32 level = 1; level <= m_higher_level; ++level) {
    Int32 old_overlap_size = AMRPatchPosition::computeOverlapLayerSize(level, m_higher_level, m_size_of_overlap_layer_top_level);
    Int32 new_overlap_size = AMRPatchPosition::computeOverlapLayerSize(level, m_higher_level, new_size_of_overlap_layer_top_level);

    if (old_overlap_size == new_overlap_size) {
      continue;
    }
    if (old_overlap_size < new_overlap_size) {
      _increaseOverlapSizeLevel(level, new_overlap_size);
    }
    else {
      _reduceOverlapSizeLevel(level, new_overlap_size);
    }
  }
  m_size_of_overlap_layer_top_level = new_size_of_overlap_layer_top_level;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 CartesianPatchGroup::
overlapLayerSize(Int32 level)
{
  // Deux cas :
  // - on est dans une phase de raffinement (beginAdaptMesh()), on doit donc
  //   considérer que le niveau le plus haut est m_target_nb_levels-1,
  // - sinon, on prend le niveau le plus haut actuel.
  Int32 higher_level = m_higher_level;
  if (m_target_nb_levels != 0) {
    higher_level = m_target_nb_levels - 1;
  }
  return AMRPatchPosition::computeOverlapLayerSize(level, higher_level, m_size_of_overlap_layer_top_level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_addPatchInstance(Ref<CartesianMeshPatch> v)
{
  m_amr_patches.add(v);
  m_amr_patches_pointer.add(v.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_removeOnePatch(Integer index)
{
  m_available_group_index.add(m_amr_patches[index]->index());
  // info() << "_removeOnePatch() -- Save group_index : " << m_available_group_index.back();

  m_amr_patch_cell_groups_all[index - 1].clear();
  m_amr_patch_cell_groups_all.remove(index - 1);

  if (m_cmesh->mesh()->meshKind().meshAMRKind() == eMeshAMRKind::PatchCartesianMeshOnly) {
    m_amr_patch_cell_groups_inpatch[index - 1].clear();
    m_amr_patch_cell_groups_inpatch.remove(index - 1);
    m_amr_patch_cell_groups_overlap[index - 1].clear();
    m_amr_patch_cell_groups_overlap.remove(index - 1);
  }

  m_amr_patches_pointer.remove(index);
  m_amr_patches.remove(index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Le tableau doit être trié.
void CartesianPatchGroup::
_removeMultiplePatches(ConstArrayView<Integer> indexes)
{
  Integer count = 0;
  for (const Integer index : indexes) {
    _removeOnePatch(index - count);
    count++;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_removeAllPatches()
{
  Ref<CartesianMeshPatch> ground_patch = m_amr_patches.front();

  for (CellGroup cell_group : m_amr_patch_cell_groups_all) {
    cell_group.clear();
  }
  m_amr_patch_cell_groups_all.clear();

  if (m_cmesh->mesh()->meshKind().meshAMRKind() == eMeshAMRKind::PatchCartesianMeshOnly) {
    for (CellGroup cell_group : m_amr_patch_cell_groups_inpatch) {
      cell_group.clear();
    }
    for (CellGroup cell_group : m_amr_patch_cell_groups_overlap) {
      cell_group.clear();
    }
    m_amr_patch_cell_groups_inpatch.clear();
    m_amr_patch_cell_groups_overlap.clear();
  }

  m_amr_patches_pointer.clear();
  m_amr_patches.clear();
  m_available_group_index.clear();
  m_patches_to_delete.clear();
  m_index_new_patches = 1;

  m_amr_patches.add(ground_patch);
  m_amr_patches_pointer.add(ground_patch.get());
  m_higher_level = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_createGroundPatch()
{
  if (!m_amr_patches.empty())
    return;
  auto patch = makeRef(new CartesianMeshPatch(m_cmesh, -1));

  if (m_cmesh->mesh()->meshKind().meshAMRKind() == eMeshAMRKind::PatchCartesianMeshOnly) {
    auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();
    patch->_internalApi()->positionRef().setMinPoint({ 0, 0, 0 });
    patch->_internalApi()->positionRef().setMaxPoint({ numbering->globalNbCellsX(0), numbering->globalNbCellsY(0), numbering->globalNbCellsZ(0) });
    patch->_internalApi()->positionRef().setLevel(0);
  }

  _addPatchInstance(patch);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer CartesianPatchGroup::
_addCellGroup(CellGroup cell_group, CartesianMeshPatch* patch)
{
  m_amr_patch_cell_groups_all.add(cell_group);

  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    // Patch non-régulier.
    // m_amr_patch_cell_groups_inpatch.add(cell_group);
    // m_amr_patch_cell_groups_overlap.add(CellGroup());
    return m_amr_patch_cell_groups_all.size() - 1;
  }

  AMRPatchPosition patch_position = patch->position();
  Ref<ICartesianMeshNumberingMngInternal> numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();

  UniqueArray<Int32> inpatch_items_lid;
  UniqueArray<Int32> overlap_items_lid;

  ENUMERATE_ (Cell, icell, cell_group) {
    Cell cell = *icell;
    const CartCoord3 pos = numbering->cellUniqueIdToCoord(cell);

    if (patch_position.isIn(pos)) {
      inpatch_items_lid.add(cell.localId());
    }
    else {
      overlap_items_lid.add(cell.localId());
    }
  }

  CellGroup own = m_cmesh->mesh()->cellFamily()->createGroup(cell_group.name().clone() + "_InPatch", inpatch_items_lid, true);
  m_amr_patch_cell_groups_inpatch.add(own);

  CellGroup overlap = m_cmesh->mesh()->cellFamily()->createGroup(cell_group.name().clone() + "_Overlap", overlap_items_lid, true);
  m_amr_patch_cell_groups_overlap.add(overlap);

  return m_amr_patch_cell_groups_all.size() - 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Il est nécessaire que le patch source et le patch part_to_remove soient
// en contact pour que cette méthode fonctionne.
void CartesianPatchGroup::
_removePartOfPatch(Integer index_patch_to_edit, const AMRPatchPosition& part_to_remove)
{
  // info() << "Coarse Zone"
  //                             << " -- Min point : " << part_to_remove.minPoint()
  //                             << " -- Max point : " << part_to_remove.maxPoint()
  //                             << " -- Level : " << part_to_remove.level();

  // p1 est le bout de patch qu'il faut retirer de p0.
  // On a donc uniquement quatre cas à traiter (sachant que p0 et p1 sont
  // forcément en contact en x et/ou y et/ou z).
  //
  // Cas 1 :
  // p0   |-----|
  // p1 |---------|
  // r = {-1, -1}
  //
  // Cas 2 :
  // p0   |-----|
  // p1       |-----|
  // r = {p1_min, -1}
  //
  // Cas 3 :
  // p0   |-----|
  // p1 |-----|
  // r = {-1, p1_max}
  //
  // Cas 4 :
  // p0   |-----|
  // p1    |---|
  // r = {p1_min, p1_max}
  auto cut_points_p0 = [](CartCoord p0_min, CartCoord p0_max, CartCoord p1_min, CartCoord p1_max) -> std::pair<CartCoord, CartCoord> {
    std::pair to_return{ -1, -1 };
    if (p1_min > p0_min && p1_min < p0_max) {
      to_return.first = p1_min;
    }
    if (p1_max > p0_min && p1_max < p0_max) {
      to_return.second = p1_max;
    }
    return to_return;
  };

  ICartesianMeshPatch* patch = m_amr_patches_pointer[index_patch_to_edit];
  AMRPatchPosition patch_position = patch->position();

  UniqueArray<AMRPatchPosition> new_patch_out;

  CartCoord3 min_point_of_patch_to_exclude(-1, -1, -1);

  // Partie découpe du patch autour de la zone à exclure.
  {
    UniqueArray<AMRPatchPosition> new_patch_in;

    // On coupe le patch en x.
    {
      auto cut_point_x = cut_points_p0(patch_position.minPoint().x, patch_position.maxPoint().x, part_to_remove.minPoint().x, part_to_remove.maxPoint().x);

      // p0   |-----|
      // p1 |---------|
      if (cut_point_x.first == -1 && cut_point_x.second == -1) {
        min_point_of_patch_to_exclude.x = patch_position.minPoint().x;
        new_patch_out.add(patch_position);
      }
      // p0   |-----|
      // p1       |-----|
      else if (cut_point_x.second == -1) {
        min_point_of_patch_to_exclude.x = cut_point_x.first;
        auto [fst, snd] = patch_position.cut(cut_point_x.first, MD_DirX);
        new_patch_out.add(fst);
        new_patch_out.add(snd);
      }
      // p0   |-----|
      // p1 |-----|
      else if (cut_point_x.first == -1) {
        min_point_of_patch_to_exclude.x = patch_position.minPoint().x;
        auto [fst, snd] = patch_position.cut(cut_point_x.second, MD_DirX);
        new_patch_out.add(fst);
        new_patch_out.add(snd);
      }
      // p0   |-----|
      // p1    |---|
      else {
        min_point_of_patch_to_exclude.x = cut_point_x.first;
        auto [fst, snd_thr] = patch_position.cut(cut_point_x.first, MD_DirX);
        new_patch_out.add(fst);
        auto [snd, thr] = snd_thr.cut(cut_point_x.second, MD_DirX);
        new_patch_out.add(snd);
        new_patch_out.add(thr);
      }
    }

    // On coupe le patch en y.
    {
      std::swap(new_patch_out, new_patch_in);

      auto cut_point_y = cut_points_p0(patch_position.minPoint().y, patch_position.maxPoint().y, part_to_remove.minPoint().y, part_to_remove.maxPoint().y);

      // p0   |-----|
      // p1 |---------|
      if (cut_point_y.first == -1 && cut_point_y.second == -1) {
        for (const AMRPatchPosition& patch_x : new_patch_in) {
          min_point_of_patch_to_exclude.y = patch_x.minPoint().y;
          new_patch_out.add(patch_x);
        }
      }
      // p0   |-----|
      // p1       |-----|
      else if (cut_point_y.second == -1) {
        min_point_of_patch_to_exclude.y = cut_point_y.first;
        for (const AMRPatchPosition& patch_x : new_patch_in) {
          auto [fst, snd] = patch_x.cut(cut_point_y.first, MD_DirY);
          new_patch_out.add(fst);
          new_patch_out.add(snd);
        }
      }
      // p0   |-----|
      // p1 |-----|
      else if (cut_point_y.first == -1) {
        for (const AMRPatchPosition& patch_x : new_patch_in) {
          min_point_of_patch_to_exclude.y = patch_x.minPoint().y;
          auto [fst, snd] = patch_x.cut(cut_point_y.second, MD_DirY);
          new_patch_out.add(fst);
          new_patch_out.add(snd);
        }
      }
      // p0   |-----|
      // p1    |---|
      else {
        min_point_of_patch_to_exclude.y = cut_point_y.first;
        for (const AMRPatchPosition& patch_x : new_patch_in) {
          auto [fst, snd_thr] = patch_x.cut(cut_point_y.first, MD_DirY);
          new_patch_out.add(fst);
          auto [snd, thr] = snd_thr.cut(cut_point_y.second, MD_DirY);
          new_patch_out.add(snd);
          new_patch_out.add(thr);
        }
      }
    }

    // On coupe le patch en z.
    if (m_cmesh->mesh()->dimension() == 3) {
      std::swap(new_patch_out, new_patch_in);
      new_patch_out.clear();

      auto cut_point_z = cut_points_p0(patch_position.minPoint().z, patch_position.maxPoint().z, part_to_remove.minPoint().z, part_to_remove.maxPoint().z);

      // p0   |-----|
      // p1 |---------|
      if (cut_point_z.first == -1 && cut_point_z.second == -1) {
        for (const AMRPatchPosition& patch_y : new_patch_in) {
          min_point_of_patch_to_exclude.z = patch_y.minPoint().z;
          new_patch_out.add(patch_y);
        }
      }
      // p0   |-----|
      // p1       |-----|
      else if (cut_point_z.second == -1) {
        for (const AMRPatchPosition& patch_y : new_patch_in) {
          min_point_of_patch_to_exclude.z = cut_point_z.first;
          auto [fst, snd] = patch_y.cut(cut_point_z.first, MD_DirZ);
          new_patch_out.add(fst);
          new_patch_out.add(snd);
        }
      }
      // p0   |-----|
      // p1 |-----|
      else if (cut_point_z.first == -1) {
        for (const AMRPatchPosition& patch_y : new_patch_in) {
          min_point_of_patch_to_exclude.z = patch_y.minPoint().z;
          auto [fst, snd] = patch_y.cut(cut_point_z.second, MD_DirZ);
          new_patch_out.add(fst);
          new_patch_out.add(snd);
        }
      }
      // p0   |-----|
      // p1    |---|
      else {
        for (const AMRPatchPosition& patch_y : new_patch_in) {
          min_point_of_patch_to_exclude.z = cut_point_z.first;
          auto [fst, snd_thr] = patch_y.cut(cut_point_z.first, MD_DirZ);
          new_patch_out.add(fst);
          auto [snd, thr] = snd_thr.cut(cut_point_z.second, MD_DirZ);
          new_patch_out.add(snd);
          new_patch_out.add(thr);
        }
      }
    }
  }

  // Partie fusion et ajout.
  {
    if (m_cmesh->mesh()->dimension() == 2) {
      min_point_of_patch_to_exclude.z = 0;
    }
    // info() << "Nb of new patch before fusion : " << new_patch_out.size();
    // info() << "min_point_of_patch_to_exclude : " << min_point_of_patch_to_exclude;

    // On met à null le patch représentant le bout de patch à retirer.
    for (AMRPatchPosition& new_patch : new_patch_out) {
      if (new_patch.minPoint() == min_point_of_patch_to_exclude) {
        new_patch.setLevel(-2); // Devient null.
      }
      // else {
      //   info() << "\tPatch before fusion"
      //                               << " -- Min point : " << new_patch.minPoint()
      //                               << " -- Max point : " << new_patch.maxPoint()
      //                               << " -- Level : " << new_patch.level();
      // }
    }

    AMRPatchPositionLevelGroup::fusionPatches(new_patch_out, false);

    // On ajoute les nouveaux patchs dans la liste des patchs.
    Integer d_nb_patch_final = 0;
    for (const auto& new_patch : new_patch_out) {
      if (!new_patch.isNull()) {
        // info() << "\tNew cut patch"
        //                                     << " -- Min point : " << new_patch.minPoint()
        //                                     << " -- Max point : " << new_patch.maxPoint()
        //                                     << " -- Level : " << new_patch.level();
        _addCutPatch(new_patch, m_amr_patch_cell_groups_all[index_patch_to_edit - 1]);
        d_nb_patch_final++;
      }
    }
    // info() << "Nb of new patch after fusion : " << d_nb_patch_final;
  }

  removePatch(index_patch_to_edit);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_addCutPatch(const AMRPatchPosition& new_patch_position, CellGroup parent_patch_cell_group)
{
  // Si cette méthode est utilisé par une autre méthode que _removePartOfPatch(),
  // voir si la mise à jour de m_higher_level est nécessaire.
  // (jusque-là, ce n'est pas utile vu qu'il y aura appel à applyPatchEdit()).
  if (parent_patch_cell_group.null())
    ARCANE_FATAL("Null cell group");

  IItemFamily* cell_family = m_cmesh->mesh()->cellFamily();
  Integer group_index = _nextIndexForNewPatch();
  String patch_group_name = String("CartesianMeshPatchCells") + group_index;

  auto* cdi = new CartesianMeshPatch(m_cmesh, group_index, new_patch_position);
  _addPatchInstance(makeRef(cdi));

  UniqueArray<Int32> cells_local_id;

  auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();
  ENUMERATE_ (Cell, icell, parent_patch_cell_group) {
    const CartCoord3 pos = numbering->cellUniqueIdToCoord(*icell);
    if (new_patch_position.isIn(pos)) {
      cells_local_id.add(icell.localId());
    }
  }

  CellGroup parent_cells = cell_family->createGroup(patch_group_name, cells_local_id, true);
  _addCellGroup(parent_cells, cdi);

  // info() << "_addCutPatch()"
  //                             << " -- m_amr_patch_cell_groups : " << m_amr_patch_cell_groups_all.size()
  //                             << " -- m_amr_patches : " << m_amr_patches.size()
  //                             << " -- group_index : " << group_index
  //                             << " -- cell_group name : " << m_amr_patch_cell_groups_all.back().name();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer CartesianPatchGroup::
_addPatch(const AMRPatchPosition& new_patch_position)
{
  UniqueArray<Int32> cells_local_id;

  auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();
  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(new_patch_position.level())) {
    const CartCoord3 pos = numbering->cellUniqueIdToCoord(*icell);
    if (new_patch_position.isInWithOverlap(pos)) {
      cells_local_id.add(icell.localId());
    }
  }

  IItemFamily* cell_family = m_cmesh->mesh()->cellFamily();
  Integer group_index = _nextIndexForNewPatch();
  String patch_group_name = String("CartesianMeshPatchCells") + group_index;

  auto* cdi = new CartesianMeshPatch(m_cmesh, group_index, new_patch_position);

  _addPatchInstance(makeRef(cdi));
  CellGroup parent_cells = cell_family->createGroup(patch_group_name, cells_local_id, true);
  Integer array_index = _addCellGroup(parent_cells, cdi);

  // TODO : Ces deux index, c'est vraiment pas une bonne idée...
  return array_index;

  // info() << "_addPatch()"
  //                             << " -- m_amr_patch_cell_groups : " << m_amr_patch_cell_groups_all.size()
  //                             << " -- m_amr_patches : " << m_amr_patches.size()
  //                             << " -- group_index : " << group_index
  //                             << " -- cell_group name : " << m_amr_patch_cell_groups_all.back().name();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
