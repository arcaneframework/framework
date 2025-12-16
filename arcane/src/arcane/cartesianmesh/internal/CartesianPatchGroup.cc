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
: m_cmesh(cmesh)
, m_index_new_patches(1)
, m_size_of_overlap_layer_sub_top_level(0)
, m_higher_level(0)
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
    UniqueArray<CartCoordType> min_point(m_amr_patches_pointer.size() * 3);
    UniqueArray<CartCoordType> max_point(m_amr_patches_pointer.size() * 3);

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
    m_cmesh->traceMng()->info(4) << "Found n=" << patch_group_names.size() << " patchs";

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
    UniqueArray<CartCoordType> min_point;
    UniqueArray<CartCoordType> max_point;

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
      const CartCoord3Type pos = numbering->cellUniqueIdToCoord(*icell);
      if (position.isInWithOverlap(pos)) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
      }
    }
  }

  amr->refine();

  _addPatch(position.patchUp(m_cmesh->mesh()->dimension()));
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
overallCells(Integer index)
{
  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    ARCANE_FATAL("Method available only with AMR PatchCartesianMeshOnly");
  }
  if (index == 0) {
    ARCANE_FATAL("You cannot get cells of ground patch with this method");
  }
  return m_amr_patch_cell_groups_overall[index - 1];
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
    // m_cmesh->traceMng()->info() << "I : " << i
    //                                     << " -- Compare Patch (min : " << patch->position().minPoint()
    //                                     << ", max : " << patch->position().maxPoint()
    //                                     << ", level : " << patch->position().level()
    //                                     << ") and Zone (min : " << zone_to_delete.minPoint()
    //                                     << ", max : " << zone_to_delete.maxPoint()
    //                                     << ", level : " << zone_to_delete.level() << ")";

    if (zone_to_delete.haveIntersection(patch->position())) {
      _splitPatch(i, zone_to_delete);
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
      const CartCoord3Type pos = numbering->cellUniqueIdToCoord(*icell);
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

  m_higher_level = 0;
  for (const auto patch : m_amr_patches_pointer) {
    const Int32 level = patch->_internalApi()->positionRef().level();
    if (level > m_higher_level) {
      m_higher_level = level;
    }
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

  // TODO : Mettre à jour la taille des couches de recouvrement !

  for (ICartesianMeshPatch* patch : m_amr_patches_pointer) {
    const Int32 level = patch->position().level();
    // Si le niveau est 0, c'est le patch spécial 0 donc on ne modifie que le max, le niveau reste à 0.
    if (level == 0) {
      const CartCoord3Type max_point = patch->position().maxPoint();
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
      if (level + 1 > m_higher_level) {
        m_higher_level = level + 1;
      }
    }
  }

  AMRPatchPosition old_ground;
  old_ground.setLevel(1);
  old_ground.setMinPoint({ 0, 0, 0 });
  old_ground.setMaxPoint({ numbering->globalNbCellsX(1), numbering->globalNbCellsY(1), numbering->globalNbCellsZ(1) });
  old_ground.setOverlapLayerSize(0);

  _addPatch(old_ground);
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
  // m_cmesh->traceMng()->info() << "Global fusion";
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

        // m_cmesh->traceMng()->info() << "\tCheck fusion"
        //                                     << " -- 0 Min point : " << patch_fusion_0.minPoint()
        //                                     << " -- 0 Max point : " << patch_fusion_0.maxPoint()
        //                                     << " -- 0 Level : " << patch_fusion_0.level()
        //                                     << " -- 1 Min point : " << patch_fusion_1.minPoint()
        //                                     << " -- 1 Max point : " << patch_fusion_1.maxPoint()
        //                                     << " -- 1 Level : " << patch_fusion_1.level();

        if (patch_fusion_0.fusion(patch_fusion_1)) {
          // m_cmesh->traceMng()->info() << "Fusion OK";
          patch_fusion_1.setLevel(-2); // Devient null.
          index_n_nb_cells[p0].second = patch_fusion_0.nbCells();

          UniqueArray<Int32> local_ids;
          allCells(index_p1).view().fillLocalIds(local_ids);
          allCells(index_p0).addItems(local_ids, false);

          // m_cmesh->traceMng()->info() << "Remove patch : " << index_p1;
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
refine(bool clear_refine_flag)
{
  // TODO : Le paramètre clear_refine_flag doit être à true pour l'instant.
  //        À cause des mailles de recouvrements, on doit regénérer les patchs
  //        de tous les niveaux à chaque fois. Pour que ça fonctionne, il
  //        faudrait demander le nombre de niveaux qui sera généré en tout,
  //        pour cette itération, pour calculer en avance la taille de la
  //        couche de recouvrement de chaque niveau.
  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    ARCANE_FATAL("Method available only with AMR PatchCartesianMeshOnly");
  }
  Integer dimension = m_cmesh->mesh()->dimension();
  Int32 nb_overlap_cells = m_size_of_overlap_layer_sub_top_level;
  Int32 min_level = 0;
  Int32 future_max_level = -1; // Désigne le niveau max qui aura des enfants, donc le futur level max +1.
  Int32 old_max_level = -1; // Mais s'il reste des mailles à des niveaux plus haut, il faut les retirer.
  auto amr = m_cmesh->_internalApi()->cartesianMeshAMRPatchMng();

  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allCells()) {
    Integer level = icell->level();
    if (icell->hasFlags(ItemFlags::II_Refine)) {
      if (level > future_max_level)
        future_max_level = level;
    }
    if (level > old_max_level)
      old_max_level = level;
  }
  future_max_level = m_cmesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMax, future_max_level);
  old_max_level = m_cmesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMax, old_max_level);

  // m_cmesh->traceMng()->info() << "Min level : " << min_level << " -- Max level : " << future_max_level;
  auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();

  AMRPatchPositionLevelGroup all_patches(future_max_level);

  for (Int32 level = future_max_level; level >= min_level; --level) {
    // m_cmesh->traceMng()->info() << "Refine Level " << level << " with " << nb_overlap_cells << " layers of overlap cells";
    if (level != future_max_level) {
      ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allCells()) {
        if (icell->level() == level && icell->hasFlags(ItemFlags::II_Refine)) {
          const CartCoord3Type pos = numbering->offsetLevelToLevel(numbering->cellUniqueIdToCoord(*icell), level, level + 1);
          for (const auto& patch : all_patches.patches(level)) {
            if (patch.isInWithOverlap(pos, patch.overlapLayerSize() + 1)) {
              icell->mutableItemBase().removeFlags(ItemFlags::II_Refine);
            }
          }
        }
      }
      // m_cmesh->traceMng()->info() << "All patch level+1 with margin (can be overlap) : ";
      // for (auto& elem : all_patches.patches(level + 1)) {
      //   m_cmesh->traceMng()->info() << "\tPatch -- min = " << elem.minPointWithOverlap() << " -- max = " << elem.maxPointWithOverlap();
      // }
    }

    AMRPatchPosition all_level;
    all_level.setLevel(level);
    all_level.setMinPoint({ 0, 0, 0 });
    all_level.setMaxPoint({ numbering->globalNbCellsX(level), numbering->globalNbCellsY(level), numbering->globalNbCellsZ(level) });
    all_level.setOverlapLayerSize(nb_overlap_cells);

    AMRPatchPositionSignature sig(all_level, m_cmesh, &all_patches);
    UniqueArray<AMRPatchPositionSignature> sig_array;
    sig_array.add(sig);

    AMRPatchPositionSignatureCut::cut(sig_array);

    for (const auto& elem : sig_array) {
      all_patches.addPatch(elem.patch());
    }
    nb_overlap_cells /= 2;
    nb_overlap_cells += 1;

    /////////
    /*
    {
      Real global_efficacity = 0;
      m_cmesh->traceMng()->info() << "All patch : ";
      for (auto& elem : sig_array) {
        m_cmesh->traceMng()->info() << "\tPatch -- min = " << elem.patch().minPoint() << " -- max = " << elem.patch().maxPoint() << " -- Efficacité : " << elem.efficacity();
        global_efficacity += elem.efficacity();
      }
      global_efficacity /= sig_array.size();
      m_cmesh->traceMng()->info() << "Global efficacity : " << global_efficacity;
      UniqueArray<Integer> out(numbering->globalNbCellsY(level) * numbering->globalNbCellsX(level), -1);
      Array2View<Integer> av_out(out.data(), numbering->globalNbCellsY(level), numbering->globalNbCellsX(level));
      ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allCells()) {
        if (icell->level() != level)
          continue;
        Integer pos_x = numbering->cellUniqueIdToCoordX(*icell);
        Integer pos_y = numbering->cellUniqueIdToCoordY(*icell);
        Integer pos_z = numbering->cellUniqueIdToCoordZ(*icell);
        Integer patch = -1;
        for (Integer i = 0; i < sig_array.size(); ++i) {
          const AMRPatchPositionSignature& elem = sig_array[i];
          if (elem.patch().isInWithOverlap(pos_x, pos_y, pos_z)) {
            patch = -2;
          }
          if (elem.isIn(pos_x, pos_y, pos_z)) {
            if (patch >= 0) {
              ARCANE_FATAL("ABCDEFG -- old : {0} -- new : {1}", patch, i);
            }
            patch = i;
          }
        }
        if (patch == -1 && icell->hasFlags(ItemFlags::II_Refine)) {
          ARCANE_FATAL("Bad Patch");
        }
        av_out(pos_y, pos_x) = patch;
      }

      StringBuilder str = "";
      for (Integer i = 0; i < numbering->globalNbCellsX(level); ++i) {
        str += "\n";
        for (Integer j = 0; j < numbering->globalNbCellsY(level); ++j) {
          Integer c = av_out(i, j);
          if (c >= 0) {
            str += "[";
            if (c < 10)
              str += " ";
            str += c;
            str += "]";
          }
          else if (c == -2) {
            str += "[RE]";
          }
          else
            str += "[  ]";
        }
      }
      m_cmesh->traceMng()->info() << str;
    }
    */
    ////////////
  }


  {
    clearRefineRelatedFlags();
  }

  _removeAllPatches();
  applyPatchEdit(false);

  for (Int32 level = min_level; level <= future_max_level; ++level) {
    all_patches.fusionPatches(level);

    ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level)) {
      if (!icell->hasHChildren()) {
        const CartCoord3Type pos = numbering->cellUniqueIdToCoord(*icell);
        for (const AMRPatchPosition& patch : all_patches.patches(level)) {
          if (patch.isInWithOverlap(pos)) {
            icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
          }
        }
      }
    }

    /*
    {
      UniqueArray<Integer> out(numbering->globalNbCellsY(level) * numbering->globalNbCellsX(level), -1);
      Array2View<Integer> av_out(out.data(), numbering->globalNbCellsY(level), numbering->globalNbCellsX(level));
      ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level)) {
        Integer pos_x = numbering->cellUniqueIdToCoordX(*icell);
        Integer pos_y = numbering->cellUniqueIdToCoordY(*icell);
        Integer pos_z = numbering->cellUniqueIdToCoordZ(*icell);
        if (icell->hasHChildren()) {
          av_out(pos_y, pos_x) = 0;
        }
        if (icell->hasFlags(ItemFlags::II_Refine)) {
          av_out(pos_y, pos_x) = 1;
        }
        if (icell->hasHChildren() && icell->hasFlags(ItemFlags::II_Refine)) {
          ARCANE_FATAL("Bad refine cell");
        }
      }

      StringBuilder str = "";
      for (Integer i = 0; i < numbering->globalNbCellsX(level); ++i) {
        str += "\n";
        for (Integer j = 0; j < numbering->globalNbCellsY(level); ++j) {
          Integer c = av_out(i, j);
          if (c == 1)
            str += "[++]";
          else if (c == 0)
            str += "[XX]";
          else
            str += "[  ]";
        }
      }
      m_cmesh->traceMng()->info() << str;
    }
    */

    amr->refine();

    // // Pour debug, forcer le else de la methode addPatch(AV<Int32>).
    // UniqueArray<Int32> d_cell_ids;
    // ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level + 1)) {
    //   d_cell_ids.add(icell.localId());
    // }
    // addPatch(d_cell_ids);

    for (const AMRPatchPosition& patch : all_patches.patches(level)) {
      _addPatch(patch.patchUp(dimension));
    }
  }

  // m_cmesh->traceMng()->info() << "max_level : " << future_max_level << " -- min_level : " << min_level;

  // On retire les mailles qui n'auront plus de parent.
  // Exemple :
  // À l'itération précédente, on a mis des flags II_Refine sur des mailles de niveau 0 et 1,
  // le niveau max était 2.
  // Alors, dans cette itération, old_max_level = 2.
  // À cette itération, on a mis des flags II_Refine uniquement sur des mailles de niveau 0.
  // Alors, future_max_level = 0.
  //
  // On doit retirer toutes les mailles de niveau 2 pour éviter les mailles orphelines.
  {
    for (Int32 level = old_max_level; level > future_max_level + 1; --level) {
      ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level)) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Coarsen);
      }
      amr->coarsen(true);
    }
  }

  for (Int32 level = future_max_level + 1; level > min_level; --level) {
    ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level)) {
      const CartCoord3Type pos = numbering->cellUniqueIdToCoord(*icell);

      bool is_in = false;
      for (const AMRPatchPosition& patch : all_patches.patches(level - 1)) {
        if (patch.patchUp(dimension).isInWithOverlap(pos)) {
          is_in = true;
          break;
        }
      }
      if (!is_in) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Coarsen);
      }
    }
    /*
    {
      UniqueArray<Integer> out(numbering->globalNbCellsY(level - 1) * numbering->globalNbCellsX(level - 1), -1);
      Array2View<Integer> av_out(out.data(), numbering->globalNbCellsY(level - 1), numbering->globalNbCellsX(level - 1));
      ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(level - 1)) {
        Integer pos_x = numbering->cellUniqueIdToCoordX(*icell);
        Integer pos_y = numbering->cellUniqueIdToCoordY(*icell);
        Integer pos_z = numbering->cellUniqueIdToCoordZ(*icell);
        if (icell->hasHChildren()) {
          if (icell->hChild(0).hasFlags(ItemFlags::II_Coarsen)) {
            av_out(pos_y, pos_x) = 1;
          }
          else {
            av_out(pos_y, pos_x) = 0;
          }
        }
      }

      StringBuilder str = "";
      for (Integer i = 0; i < numbering->globalNbCellsX(level - 1); ++i) {
        str += "\n";
        for (Integer j = 0; j < numbering->globalNbCellsY(level - 1); ++j) {
          Integer c = av_out(i, j);
          if (c == 1)
            str += "[--]";
          else if (c == 0)
            str += "[XX]";
          else
            str += "[  ]";
        }
      }
      m_cmesh->traceMng()->info() << str;
    }
    */

    amr->coarsen(true);
  }
  m_cmesh->computeDirections();

  if (clear_refine_flag) {
    ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allCells()) {
      if (icell->hasFlags(ItemFlags::II_Coarsen)) {
        ARCANE_FATAL("Pas normal");
      }
      icell->mutableItemBase().removeFlags(ItemFlags::II_Refine);
    }
  }

  // m_cmesh->traceMng()->info() << "NbPatch : " << m_cmesh->patches().size();
  //
  // for (Integer i = 0; i < m_cmesh->patches().size(); ++i) {
  //   auto patch = m_cmesh->amrPatch(i);
  //   m_cmesh->traceMng()->info() << "Patch #" << i;
  //   m_cmesh->traceMng()->info() << "\tMin Point : " << patch.patchInterface()->position().minPoint();
  //   m_cmesh->traceMng()->info() << "\tMax Point : " << patch.patchInterface()->position().maxPoint();
  //   m_cmesh->traceMng()->info() << "\tLevel : " << patch.patchInterface()->position().level();
  //   m_cmesh->traceMng()->info() << "\tNbCells : " << patch.patchInterface()->cells().size();
  //   m_cmesh->traceMng()->info() << "\tIndex : " << patch.patchInterface()->index();
  // }
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
  m_size_of_overlap_layer_sub_top_level = (size_of_overlap_layer_top_level + 1) / 2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 CartesianPatchGroup::
overlapLayerSize(Int32 level)
{
  if (level < 0 || level > m_higher_level) {
    ARCANE_FATAL("Level doesn't exist");
  }
  if (level == m_higher_level) {
    return m_size_of_overlap_layer_sub_top_level * 2;
  }
  Integer nb_overlap_cells = m_size_of_overlap_layer_sub_top_level;
  for (Integer i = m_higher_level - 1; i > level; --i) {
    nb_overlap_cells /= 2;
    nb_overlap_cells += 1;
  }
  return nb_overlap_cells;
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
  // m_cmesh->traceMng()->info() << "_removeOnePatch() -- Save group_index : " << m_available_group_index.back();

  m_amr_patch_cell_groups_all[index - 1].clear();
  m_amr_patch_cell_groups_all.remove(index - 1);

  if (m_cmesh->mesh()->meshKind().meshAMRKind() == eMeshAMRKind::PatchCartesianMeshOnly) {
    m_amr_patch_cell_groups_inpatch[index - 1].clear();
    m_amr_patch_cell_groups_inpatch.remove(index - 1);
    m_amr_patch_cell_groups_overall[index - 1].clear();
    m_amr_patch_cell_groups_overall.remove(index - 1);
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
    for (CellGroup cell_group : m_amr_patch_cell_groups_overall) {
      cell_group.clear();
    }
    m_amr_patch_cell_groups_inpatch.clear();
    m_amr_patch_cell_groups_overall.clear();
  }

  m_amr_patches_pointer.clear();
  m_amr_patches.clear();
  m_available_group_index.clear();
  m_patches_to_delete.clear();
  m_index_new_patches = 1;

  m_amr_patches.add(ground_patch);
  m_amr_patches_pointer.add(ground_patch.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_createGroundPatch()
{
  if (!m_amr_patches.empty())
    return;
  auto patch = makeRef(new CartesianMeshPatch(m_cmesh, -1));
  _addPatchInstance(patch);

  if (m_cmesh->mesh()->meshKind().meshAMRKind() == eMeshAMRKind::PatchCartesianMeshOnly) {
    auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();
    patch->_internalApi()->positionRef().setMinPoint({ 0, 0, 0 });
    patch->_internalApi()->positionRef().setMaxPoint({ numbering->globalNbCellsX(0), numbering->globalNbCellsY(0), numbering->globalNbCellsZ(0) });
    patch->_internalApi()->positionRef().setLevel(0);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_addCellGroup(CellGroup cell_group, CartesianMeshPatch* patch)
{
  m_amr_patch_cell_groups_all.add(cell_group);

  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    // Patch non-régulier.
    // m_amr_patch_cell_groups_inpatch.add(cell_group);
    // m_amr_patch_cell_groups_overall.add(CellGroup());
    return;
  }

  AMRPatchPosition patch_position = patch->position();
  Ref<ICartesianMeshNumberingMngInternal> numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();

  UniqueArray<Int32> inpatch_items_lid;
  UniqueArray<Int32> overall_items_lid;

  ENUMERATE_ (Cell, icell, cell_group) {
    Cell cell = *icell;
    const CartCoord3Type pos = numbering->cellUniqueIdToCoord(cell);

    if (cell.isOwn() && patch_position.isIn(pos)) {
      inpatch_items_lid.add(cell.localId());
    }
    else {
      overall_items_lid.add(cell.localId());
    }
  }

  CellGroup own = m_cmesh->mesh()->cellFamily()->createGroup(cell_group.name().clone() + "_InPatch", inpatch_items_lid, true);
  m_amr_patch_cell_groups_inpatch.add(own);

  CellGroup overall = m_cmesh->mesh()->cellFamily()->createGroup(cell_group.name().clone() + "_Overall", overall_items_lid, true);
  m_amr_patch_cell_groups_overall.add(overall);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Il est nécessaire que le patch source et le patch part_to_remove soient
// en contact pour que cette méthode fonctionne.
void CartesianPatchGroup::
_splitPatch(Integer index_patch, const AMRPatchPosition& part_to_remove)
{
  // m_cmesh->traceMng()->info() << "Coarse Zone"
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
  auto cut_points_p0 = [](CartCoordType p0_min, CartCoordType p0_max, CartCoordType p1_min, CartCoordType p1_max) -> std::pair<CartCoordType, CartCoordType> {
    std::pair to_return{ -1, -1 };
    if (p1_min > p0_min && p1_min < p0_max) {
      to_return.first = p1_min;
    }
    if (p1_max > p0_min && p1_max < p0_max) {
      to_return.second = p1_max;
    }
    return to_return;
  };

  ICartesianMeshPatch* patch = m_amr_patches_pointer[index_patch];
  AMRPatchPosition patch_position = patch->position();

  UniqueArray<AMRPatchPosition> new_patch_out;

  CartCoord3Type min_point_of_patch_to_exclude(-1, -1, -1);

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
    // m_cmesh->traceMng()->info() << "Nb of new patch before fusion : " << new_patch_out.size();
    // m_cmesh->traceMng()->info() << "min_point_of_patch_to_exclude : " << min_point_of_patch_to_exclude;

    // On met à null le patch représentant le bout de patch à retirer.
    for (AMRPatchPosition& new_patch : new_patch_out) {
      if (new_patch.minPoint() == min_point_of_patch_to_exclude) {
        new_patch.setLevel(-2); // Devient null.
      }
      // else {
      //   m_cmesh->traceMng()->info() << "\tPatch before fusion"
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
        // m_cmesh->traceMng()->info() << "\tNew cut patch"
        //                                     << " -- Min point : " << new_patch.minPoint()
        //                                     << " -- Max point : " << new_patch.maxPoint()
        //                                     << " -- Level : " << new_patch.level();
        _addCutPatch(new_patch, m_amr_patch_cell_groups_all[index_patch - 1]);
        d_nb_patch_final++;
      }
    }
    // m_cmesh->traceMng()->info() << "Nb of new patch after fusion : " << d_nb_patch_final;
  }

  removePatch(index_patch);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_addCutPatch(const AMRPatchPosition& new_patch_position, CellGroup parent_patch_cell_group)
{
  // Si cette méthode est utilisé par une autre méthode que _splitPatch(),
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
    const CartCoord3Type pos = numbering->cellUniqueIdToCoord(*icell);
    if (new_patch_position.isIn(pos)) {
      cells_local_id.add(icell.localId());
    }
  }

  CellGroup parent_cells = cell_family->createGroup(patch_group_name, cells_local_id, true);
  _addCellGroup(parent_cells, cdi);

  // m_cmesh->traceMng()->info() << "_addCutPatch()"
  //                             << " -- m_amr_patch_cell_groups : " << m_amr_patch_cell_groups_all.size()
  //                             << " -- m_amr_patches : " << m_amr_patches.size()
  //                             << " -- group_index : " << group_index
  //                             << " -- cell_group name : " << m_amr_patch_cell_groups_all.back().name();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_addPatch(const AMRPatchPosition& new_patch_position)
{
  UniqueArray<Int32> cells_local_id;

  auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMngInternal();
  ENUMERATE_ (Cell, icell, m_cmesh->mesh()->allLevelCells(new_patch_position.level())) {
    const CartCoord3Type pos = numbering->cellUniqueIdToCoord(*icell);
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
  _addCellGroup(parent_cells, cdi);

  if (new_patch_position.level() > m_higher_level) {
    m_higher_level = new_patch_position.level();
  }

  // m_cmesh->traceMng()->info() << "_addPatch()"
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
