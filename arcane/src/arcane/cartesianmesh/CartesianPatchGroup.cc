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

#include "arcane/cartesianmesh/CartesianPatchGroup.h"

#include "arcane/cartesianmesh/ICartesianMeshNumberingMng.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FixedArray.h"
#include "arcane/utils/Vector3.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/MeshKind.h"

#include "arcane/cartesianmesh/internal/CartesianMeshPatch.h"
#include "arcane/cartesianmesh/internal/ICartesianMeshInternal.h"

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

CartesianPatchGroup::CartesianPatchGroup(ICartesianMesh* cmesh)
: m_cmesh(cmesh)
, m_index_new_patches(0)
{}

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
addPatch(CellGroup cell_group)
{
  _createGroundPatch();
  if (cell_group.null())
    ARCANE_FATAL("Null cell group");
  Integer index = nextIndexForNewPatch();
  auto* cdi = new CartesianMeshPatch(m_cmesh, index);
  m_amr_patch_cell_groups.add(cell_group);
  _addPatchInstance(makeRef(cdi));
  m_cmesh->traceMng()->info() << "m_amr_patch_cell_groups : " << m_amr_patch_cell_groups.size()
                              << " -- m_amr_patches : " << m_amr_patches.size()
                              << " -- m_amr_patches_pointer : " << m_amr_patches_pointer.size()
                              << " -- index : " << index - 1
                              << " -- cell_group name : " << m_amr_patch_cell_groups[index - 1].name(); // m_index_new_patches commence par 1, pas le tableau m_amr_patch_cell_groups

  if (m_cmesh->mesh()->meshKind().meshAMRKind() == eMeshAMRKind::PatchCartesianMeshOnly) {
    auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMng();
    FixedArray<Int64, 6> min_n_max;
    min_n_max[0] = INT64_MAX;
    min_n_max[1] = INT64_MAX;
    min_n_max[2] = INT64_MAX;
    min_n_max[3] = -1;
    min_n_max[4] = -1;
    min_n_max[5] = -1;
    ArrayView min(min_n_max.view().subView(0, 3));
    ArrayView max(min_n_max.view().subView(3, 3));
    Int64 nb_cells = 0;
    Integer level = -1;
    ENUMERATE_ (Cell, icell, cell_group) {
      if (icell->isOwn())
        nb_cells++;
      if (level == -1) {
        level = icell->level();
      }
      if (level != icell->level()) {
        ARCANE_FATAL("Level pb -- Zone with cells to different levels -- Level recorded before : {0} -- Cell Level : {1} -- CellUID : {2}", level, icell->level(), icell->uniqueId());
      }
      Int64 pos_x = numbering->cellUniqueIdToCoordX(*icell);
      if (pos_x < min[MD_DirX])
        min[MD_DirX] = pos_x;
      if (pos_x > max[MD_DirX])
        max[MD_DirX] = pos_x;

      Int64 pos_y = numbering->cellUniqueIdToCoordY(*icell);
      if (pos_y < min[MD_DirY])
        min[MD_DirY] = pos_y;
      if (pos_y > max[MD_DirY])
        max[MD_DirY] = pos_y;

      Int64 pos_z = numbering->cellUniqueIdToCoordZ(*icell);
      if (pos_z < min[MD_DirZ])
        min[MD_DirZ] = pos_z;
      if (pos_z > max[MD_DirZ])
        max[MD_DirZ] = pos_z;
    }
    m_cmesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMin, min);
    m_cmesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMax, max);
    nb_cells = m_cmesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceSum, nb_cells);
    Integer level_r = m_cmesh->mesh()->parallelMng()->reduce(MessagePassing::ReduceMax, level);

    if (level != -1 && level != level_r) {
      ARCANE_FATAL("Bad level reduced");
    }

    max[MD_DirX] += 1;
    max[MD_DirY] += 1;
    max[MD_DirZ] += 1;

    {
      Int64 nb_cells_patch = (max[MD_DirX] - min[MD_DirX]) * (max[MD_DirY] - min[MD_DirY]) * (max[MD_DirZ] - min[MD_DirZ]);
      if (nb_cells != nb_cells_patch) {
        ARCANE_FATAL("Not regular patch");
      }
    }

    cdi->position().setMinPoint({ min[MD_DirX], min[MD_DirY], min[MD_DirZ] });
    cdi->position().setMaxPoint({ max[MD_DirX], max[MD_DirY], max[MD_DirZ] });
    cdi->position().setLevel(level_r);
  }
  m_cmesh->traceMng()->info() << "Min Point : " << cdi->position().minPoint();
  m_cmesh->traceMng()->info() << "Max Point : " << cdi->position().maxPoint();
  m_cmesh->traceMng()->info() << "Level : " << cdi->position().level();
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
cells(const Integer index)
{
  if (index == 0) {
    ARCANE_FATAL("You cannot get cells of ground patch with this method");
  }
  return m_amr_patch_cell_groups[index - 1];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Attention : efface aussi le ground patch. Nécessaire de le récupérer après coup.
void CartesianPatchGroup::
clear()
{
  m_amr_patch_cell_groups.clear();
  m_amr_patches_pointer.clear();
  m_amr_patches.clear();
  m_index_new_patches = 0;
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
  for (CellGroup cells : m_amr_patch_cell_groups) {
    cells.removeItems(cells_local_id);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
removeCellsInAllPatches(const AMRPatchPosition& zone_to_delete)
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

    if (_isPatchInContact(patch->position(), zone_to_delete)) {
      _splitPatch(i, zone_to_delete);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
applyPatchEdit(bool remove_empty_patches)
{
  m_cmesh->mesh()->traceMng()->info() << "applyPatchEdit() -- Remove nb patch : " << m_patches_to_delete.size();

  std::sort(m_patches_to_delete.begin(), m_patches_to_delete.end(),
            [](const Integer a, const Integer b) {
              return a < b;
            });

  _removeMultiplePatches(m_patches_to_delete);
  m_patches_to_delete.clear();

  if (remove_empty_patches) {
    UniqueArray<Integer> size_of_patches(m_amr_patch_cell_groups.size());
    for (Integer i = 0; i < m_amr_patch_cell_groups.size(); ++i) {
      size_of_patches[i] = m_amr_patch_cell_groups[i].size();
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
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
updateLevelsBeforeAddGroundPatch()
{
  if (m_cmesh->mesh()->meshKind().meshAMRKind() != eMeshAMRKind::PatchCartesianMeshOnly) {
    return;
  }
  auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMng();
  for (ICartesianMeshPatch* patch : m_amr_patches_pointer) {
    Integer level = patch->position().level();
    // Si le niveau est 0, c'est le patch spécial 0 donc on ne modifie que le max, le niveau reste à 0.
    if (level == 0) {
      Int64x3 max_point = patch->position().maxPoint();
      if (m_cmesh->mesh()->dimension() == 2) {
        patch->position().setMaxPoint({
        numbering->offsetLevelToLevel(max_point.x, level, level - 1),
        numbering->offsetLevelToLevel(max_point.y, level, level - 1),
        1,
        });
      }
      else {
        patch->position().setMaxPoint({
        numbering->offsetLevelToLevel(max_point.x, level, level - 1),
        numbering->offsetLevelToLevel(max_point.y, level, level - 1),
        numbering->offsetLevelToLevel(max_point.z, level, level - 1),
        });
      }
    }
    // Sinon, on "surélève" le niveau des patchs vu qu'il va y avoir le patch "-1"
    else {
      patch->position().setLevel(level + 1);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer CartesianPatchGroup::
nextIndexForNewPatch()
{
  return m_index_new_patches;
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

    std::sort(index_n_nb_cells.begin(), index_n_nb_cells.end(),
              [](const std::pair<Integer, Int64>& a, const std::pair<Integer, Int64>& b) {
                return a.second < b.second;
              });

    for (Integer p0 = 0; p0 < index_n_nb_cells.size(); ++p0) {
      auto [index_p0, nb_cells_p0] = index_n_nb_cells[p0];

      AMRPatchPosition& patch_fusion_0 = m_amr_patches_pointer[index_p0]->position();
      if (patch_fusion_0.isNull())
        continue;

      // Si une fusion a déjà eu lieu, on doit alors regarder les patchs avant "p0"
      // (vu qu'il y en a au moins un qui a été modifié).
      // (une "optimisation" pourrait être de récupérer la position du premier
      // patch fusionné mais bon, moins lisible + pas beaucoup de patchs).
      Integer p1 = (fusion ? 0 : p0 + 1);
      for (; p1 < m_amr_patches_pointer.size(); ++p1) {
        if (p1 == p0)
          continue;
        auto [index_p1, nb_cells_p1] = index_n_nb_cells[p1];

        AMRPatchPosition& patch_fusion_1 = m_amr_patches_pointer[index_p1]->position();

        if (patch_fusion_1.isNull())
          continue;

        // m_cmesh->traceMng()->info() << "\tCheck fusion"
        //                                     << " -- 0 Min point : " << patch_fusion_0.minPoint()
        //                                     << " -- 0 Max point : " << patch_fusion_0.maxPoint()
        //                                     << " -- 0 Level : " << patch_fusion_0.level()
        //                                     << " -- 1 Min point : " << patch_fusion_1.minPoint()
        //                                     << " -- 1 Max point : " << patch_fusion_1.maxPoint()
        //                                     << " -- 1 Level : " << patch_fusion_1.level();

        if (patch_fusion_0.canBeFusion(patch_fusion_1)) {
          // m_cmesh->traceMng()->info() << "Fusion OK";
          patch_fusion_0.fusion(patch_fusion_1);
          patch_fusion_1.setLevel(-2); // Devient null.
          index_n_nb_cells[p0].second = patch_fusion_0.nbCells();

          UniqueArray<Int32> local_ids;
          cells(index_p1).view().fillLocalIds(local_ids);
          cells(index_p0).addItems(local_ids, false);

          m_cmesh->traceMng()->info() << "Remove patch : " << index_p1;
          removePatch(index_p1);

          fusion = true;
          break;
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_addPatchInstance(Ref<CartesianMeshPatch> v)
{
  m_amr_patches.add(v);
  m_amr_patches_pointer.add(v.get());
  m_index_new_patches++;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_removeOnePatch(Integer index)
{
  m_amr_patch_cell_groups.remove(index - 1);
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
_createGroundPatch()
{
  if (!m_amr_patches.empty())
    return;
  auto patch = makeRef(new CartesianMeshPatch(m_cmesh, -1));
  _addPatchInstance(patch);

  if (m_cmesh->mesh()->meshKind().meshAMRKind() == eMeshAMRKind::PatchCartesianMeshOnly) {
    auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMng();
    patch->position().setMinPoint({ 0, 0, 0 });
    patch->position().setMaxPoint({ numbering->globalNbCellsX(0), numbering->globalNbCellsY(0), numbering->globalNbCellsZ(0) });
    patch->position().setLevel(0);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool CartesianPatchGroup::
_isPatchInContact(const AMRPatchPosition& patch_position0, const AMRPatchPosition& patch_position1)
{
  return (
  (patch_position0.level() == patch_position1.level()) &&
  (patch_position0.maxPoint().x > patch_position1.minPoint().x && patch_position1.maxPoint().x > patch_position0.minPoint().x) &&
  (patch_position0.maxPoint().y > patch_position1.minPoint().y && patch_position1.maxPoint().y > patch_position0.minPoint().y) &&
  (m_cmesh->mesh()->dimension() == 2 || (patch_position0.maxPoint().z > patch_position1.minPoint().z && patch_position1.maxPoint().z > patch_position0.minPoint().z)));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Il est nécessaire que le patch source et le patch part_to_remove soient
// en contact pour que cette méthode fonctionne.
void CartesianPatchGroup::
_splitPatch(Integer index_patch, const AMRPatchPosition& part_to_remove)
{
  m_cmesh->traceMng()->info() << "Coarse Zone"
                              << " -- Min point : " << part_to_remove.minPoint()
                              << " -- Max point : " << part_to_remove.maxPoint()
                              << " -- Level : " << part_to_remove.level();

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
  auto cut_points_p0 = [](Int64 p0_min, Int64 p0_max, Int64 p1_min, Int64 p1_max) -> std::pair<Int64, Int64> {
    std::pair<Int64, Int64> to_return{ -1, -1 };
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

  Int64x3 min_point_of_patch_to_exclude(-1, -1, -1);

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

      std::pair<Int64, Int64> cut_point_z = cut_points_p0(patch_position.minPoint().z, patch_position.maxPoint().z, part_to_remove.minPoint().z, part_to_remove.maxPoint().z);

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
    m_cmesh->traceMng()->info() << "Nb of new patch before fusion : " << new_patch_out.size();
    m_cmesh->traceMng()->info() << "min_point_of_patch_to_exclude : " << min_point_of_patch_to_exclude;

    // On met à null le patch représentant le bout de patch à retirer.
    for (AMRPatchPosition& new_patch : new_patch_out) {
      if (new_patch.minPoint() == min_point_of_patch_to_exclude) {
        new_patch.setLevel(-2); // Devient null.
      }
    }

    // Algo de fusion.
    // D'abord, on trie les patchs du plus petit nb de mailles au plus grand nb de mailles (optionnel).
    // Ensuite, pour chaque patch, on regarde si l'on peut le fusionner avec un autre.
    // Si on arrive à faire une fusion, on recommence l'algo jusqu'à ne plus pouvoir fusionner.
    bool fusion = true;
    while (fusion) {
      fusion = false;

      std::sort(new_patch_out.begin(), new_patch_out.end(),
                [](const AMRPatchPosition& a, const AMRPatchPosition& b) {
                  return a.nbCells() < b.nbCells();
                });

      for (Integer p0 = 0; p0 < new_patch_out.size(); ++p0) {
        AMRPatchPosition& patch_fusion_0 = new_patch_out[p0];
        if (patch_fusion_0.isNull())
          continue;

        // Si une fusion a déjà eu lieu, on doit alors regarder les patchs avant "p0"
        // (vu qu'il y en a au moins un qui a été modifié).
        // (une "optimisation" pourrait être de récupérer la position du premier
        // patch fusionné mais bon, moins lisible + pas beaucoup de patchs).
        Integer p1 = (fusion ? 0 : p0 + 1);
        for (; p1 < new_patch_out.size(); ++p1) {
          if (p1 == p0)
            continue;

          AMRPatchPosition& patch_fusion_1 = new_patch_out[p1];

          if (patch_fusion_1.isNull())
            continue;

          // m_cmesh->traceMng()->info() << "\tCheck fusion"
          //                                     << " -- 0 Min point : " << patch_fusion_0.minPoint()
          //                                     << " -- 0 Max point : " << patch_fusion_0.maxPoint()
          //                                     << " -- 0 Level : " << patch_fusion_0.level()
          //                                     << " -- 1 Min point : " << patch_fusion_1.minPoint()
          //                                     << " -- 1 Max point : " << patch_fusion_1.maxPoint()
          //                                     << " -- 1 Level : " << patch_fusion_1.level();
          if (patch_fusion_0.canBeFusion(patch_fusion_1)) {
            // m_cmesh->traceMng()->info() << "Fusion OK";
            patch_fusion_0.fusion(patch_fusion_1);
            patch_fusion_1.setLevel(-2); // Devient null.
            fusion = true;
            break;
          }
        }
      }
    }

    // On ajoute les nouveaux patchs dans la liste des patchs.
    Integer d_nb_patch_final = 0;
    for (const auto& new_patch : new_patch_out) {
      if (!new_patch.isNull()) {
        // m_cmesh->traceMng()->info() << "\tNew cut patch"
        //                                     << " -- Min point : " << new_patch.minPoint()
        //                                     << " -- Max point : " << new_patch.maxPoint()
        //                                     << " -- Level : " << new_patch.level();
        _addCutPatch(new_patch, m_amr_patch_cell_groups[index_patch - 1]);
        d_nb_patch_final++;
      }
    }
    m_cmesh->traceMng()->info() << "Nb of new patch after fusion : " << d_nb_patch_final;
  }

  removePatch(index_patch);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianPatchGroup::
_addCutPatch(const AMRPatchPosition& new_patch_position, CellGroup parent_patch_cell_group)
{
  if (parent_patch_cell_group.null())
    ARCANE_FATAL("Null cell group");

  // TODO : Le nextIndexForNewPatch() actuel ne fonctionnera pas si on s'amuse à créer, détruire et recréer des patchs !
  // TODO : Attribut que l'on incrémente de 1 lors de la création d'un patch.
  IItemFamily* cell_family = m_cmesh->mesh()->cellFamily();
  Integer index = nextIndexForNewPatch();
  String parent_group_name = String("CartesianMeshPatchParentCells") + index;

  auto* cdi = new CartesianMeshPatch(m_cmesh, index + 1);

  _addPatchInstance(makeRef(cdi));

  UniqueArray<Int32> cells_local_id;

  cdi->position().setLevel(new_patch_position.level());
  cdi->position().setMinPoint(new_patch_position.minPoint());
  cdi->position().setMaxPoint(new_patch_position.maxPoint());

  auto numbering = m_cmesh->_internalApi()->cartesianMeshNumberingMng();
  ENUMERATE_ (Cell, icell, parent_patch_cell_group) {
    Int64 pos_x = numbering->cellUniqueIdToCoordX(*icell);
    Int64 pos_y = numbering->cellUniqueIdToCoordY(*icell);
    Int64 pos_z = numbering->cellUniqueIdToCoordZ(*icell);
    if (new_patch_position.isIn(pos_x, pos_y, pos_z)) {
      cells_local_id.add(icell.localId());
    }
  }

  CellGroup parent_cells = cell_family->createGroup(parent_group_name, cells_local_id, true);
  m_amr_patch_cell_groups.add(parent_cells);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
