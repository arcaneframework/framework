// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshCoarsening.cc                                  (C) 2000-2023 */
/*                                                                           */
/* Déraffinement d'un maillage cartésien.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshCoarsening2.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/CartesianGridDimension.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/SimpleSVGMeshExporter.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/MeshStats.h"

#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/CellDirectionMng.h"

#include <unordered_set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshCoarsening2::
CartesianMeshCoarsening2(ICartesianMesh* m)
: TraceAccessor(m->traceMng())
, m_cartesian_mesh(m)
{
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CARTESIANMESH_COARSENING_VERBOSITY_LEVEL", true))
    m_verbosity_level = v.value();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Retourne le max des uniqueId() des entités de \a group
Int64 CartesianMeshCoarsening2::
_getMaxUniqueId(const ItemGroup& group)
{
  Int64 max_offset = 0;
  ENUMERATE_ (Item, iitem, group) {
    Item item = *iitem;
    if (max_offset < item.uniqueId())
      max_offset = item.uniqueId();
  }
  return max_offset;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshCoarsening2::
createCoarseCells()
{
  if (m_is_create_coarse_called)
    ARCANE_FATAL("This method has already been called");
  m_is_create_coarse_called = true;

  const bool is_verbose = m_verbosity_level > 0;
  IMesh* mesh = m_cartesian_mesh->mesh();
  Integer nb_patch = m_cartesian_mesh->nbPatch();
  if (nb_patch != 1)
    ARCANE_FATAL("This method is only valid for 1 patch (nb_patch={0})", nb_patch);

  if (!mesh->isAmrActivated())
    ARCANE_FATAL("AMR is not activated for this case");

  // TODO: Supprimer les mailles fantômes puis les reconstruire
  // TODO: Mettre à jour les informations dans CellDirectionMng
  // de ownNbCell(), globalNbCell(), ...

  Integer nb_dir = mesh->dimension();
  if (nb_dir != 2)
    ARCANE_FATAL("This method is only valid for 2D mesh");

  IParallelMng* pm = mesh->parallelMng();
  const Int32 my_rank = pm->commRank();
  info() << "CoarseCartesianMesh nb_direction=" << nb_dir;

  for (Integer idir = 0; idir < nb_dir; ++idir) {
    CellDirectionMng cdm(m_cartesian_mesh->cellDirection(idir));
    Int32 nb_own_cell = cdm.ownNbCell();
    info() << "NB_OWN_CELL dir=" << idir << " n=" << nb_own_cell;
    if ((nb_own_cell % 2) != 0)
      ARCANE_FATAL("Invalid number of cells ({0}) for direction {1}. Should be a multiple of 2",
                   nb_own_cell, idir);
  }

  // Calcule l'offset pour la création des uniqueId().
  // On prend comme offset le max des uniqueId() des faces et des mailles.
  // A terme avec la numérotation cartésienne partout, on pourra déterminer
  // directement cette valeur
  Int64 max_cell_uid = _getMaxUniqueId(mesh->allCells());
  Int64 max_face_uid = _getMaxUniqueId(mesh->allFaces());
  const Int64 coarse_grid_cell_offset = 1 + pm->reduce(Parallel::ReduceMax, math::max(max_cell_uid, max_face_uid));
  m_first_own_cell_unique_id_offset = coarse_grid_cell_offset;

  CellDirectionMng cdm_x(m_cartesian_mesh->cellDirection(0));
  CellDirectionMng cdm_y(m_cartesian_mesh->cellDirection(1));

  const Int64 global_nb_cell_x = cdm_x.globalNbCell();
  const Int64 global_nb_cell_y = cdm_y.globalNbCell();
  CartesianGridDimension refined_grid_dim(global_nb_cell_x, global_nb_cell_y);
  CartesianGridDimension coarse_grid_dim(global_nb_cell_x / 2, global_nb_cell_y / 2);
  CartesianGridDimension::CellUniqueIdComputer2D refined_cell_uid_computer(refined_grid_dim.getCellComputer2D(0));
  CartesianGridDimension::NodeUniqueIdComputer2D refined_node_uid_computer(refined_grid_dim.getNodeComputer2D(0));
  CartesianGridDimension::CellUniqueIdComputer2D coarse_cell_uid_computer(coarse_grid_dim.getCellComputer2D(coarse_grid_cell_offset));
  CartesianGridDimension::FaceUniqueIdComputer2D coarse_face_uid_computer(coarse_grid_dim.getFaceComputer2D(coarse_grid_cell_offset));

  // Pour les mailles et faces grossières, les noeuds existent déjà
  // On ne peut donc pas utiliser la connectivité cartésienne de la grille grossière
  // pour eux (on pourra le faire lorsque l'AMR par patch avec duplication sera active)
  // En attendant on utilise la numérotation de la grille raffinée.

  // TODO: Calculer le nombre de faces et de mailles et allouer en conséquence.
  UniqueArray<Int64> faces_infos;
  UniqueArray<Int64> cells_infos;
  Int32 nb_coarse_face = 0;
  Int32 nb_coarse_cell = 0;
  //! Liste de la première fille de chaque maille grossière
  UniqueArray<Int64> first_child_cell_unique_ids;
  ENUMERATE_ (Cell, icell, mesh->ownCells()) {
    Cell cell = *icell;
    Int64 cell_uid = cell.uniqueId();
    Int64x3 cell_xy = refined_cell_uid_computer.compute(cell_uid);
    const Int64 cell_x = cell_xy.x;
    const Int64 cell_y = cell_xy.y;
    // Comme on déraffine par 2, ne prend que les mailles dont les coordoonnées
    // topologiques sont paires
    if ((cell_x % 2) != 0 || (cell_y % 2) != 0)
      continue;
    if (is_verbose)
      info() << "CELLCoarse uid=" << cell_uid << " x=" << cell_x << " y=" << cell_y;
    const Int64 coarse_cell_x = cell_x / 2;
    const Int64 coarse_cell_y = cell_y / 2;
    std::array<Int64, 4> node_uids_container;
    ArrayView<Int64> node_uids(node_uids_container);
    node_uids[0] = refined_node_uid_computer.compute(cell_x + 0, cell_y + 0);
    node_uids[1] = refined_node_uid_computer.compute(cell_x + 2, cell_y + 0);
    node_uids[2] = refined_node_uid_computer.compute(cell_x + 2, cell_y + 2);
    node_uids[3] = refined_node_uid_computer.compute(cell_x + 0, cell_y + 2);
    if (is_verbose)
      info() << "CELLNodes uid=" << node_uids;
    std::array<Int64, 4> coarse_face_uids = coarse_face_uid_computer.computeForCell(coarse_cell_x, coarse_cell_y);
    const ItemTypeInfo* cell_type = cell.typeInfo();
    // Ajoute les 4 faces
    for (Int32 z = 0; z < 4; ++z) {
      ItemTypeInfo::LocalFace lface = cell_type->localFace(z);
      faces_infos.add(IT_Line2);
      faces_infos.add(coarse_face_uids[z]);
      faces_infos.add(node_uids[lface.node(0)]);
      faces_infos.add(node_uids[lface.node(1)]);
      ++nb_coarse_face;
    }
    // Ajoute la maille
    {
      cells_infos.add(IT_Quad4);
      cells_infos.add(coarse_cell_uid_computer.compute(coarse_cell_x, coarse_cell_y));
      for (Int32 z = 0; z < 4; ++z)
        cells_infos.add(node_uids[z]);
      ++nb_coarse_cell;
      first_child_cell_unique_ids.add(cell.uniqueId());
    }
  }

  UniqueArray<Int32> cells_local_ids;
  UniqueArray<Int32> faces_local_ids;
  cells_local_ids.resize(nb_coarse_cell);
  faces_local_ids.resize(nb_coarse_face);
  mesh->modifier()->addFaces(nb_coarse_face, faces_infos, faces_local_ids);
  mesh->modifier()->addCells(nb_coarse_cell, cells_infos, cells_local_ids);

  // Maintenant que les mailles grossières sont créées, il faut indiquer
  // qu'elles sont parentes.
  IItemFamily* cell_family = mesh->cellFamily();

  // Positionne les propriétaires des nouvelles mailles
  // et ajoute un flag (ItemFlags::II_UserMark1) pour les marquer.
  // Cela sera utilisé pour détruire les mailles raffinées par la suite.
  {
    ENUMERATE_ (Cell, icell, cell_family->view(cells_local_ids)) {
      Cell cell = *icell;
      cell.mutableItemBase().setOwner(my_rank, my_rank);
      cell.mutableItemBase().addFlags(ItemFlags::II_UserMark1);
    }
    cell_family->notifyItemsOwnerChanged();
  }

  // Il faut donner un propriétaire aux faces.
  // Comme les nouvelles faces utilisent un noeud déjà existant, on prend comme propriétaire
  // celui du premier noeud de la face
  {
    IItemFamily* face_family = mesh->faceFamily();
    ENUMERATE_ (Face, iface, face_family->view(faces_local_ids)) {
      Face face = *iface;
      Int32 owner = face.node(0).owner();
      face.mutableItemBase().setOwner(owner, my_rank);
    }
    face_family->notifyItemsOwnerChanged();
  }

  // Met à jour le maillage
  mesh->modifier()->endUpdate();

  // Après l'appel à endUpdate() les numéros locaux ne changent plus.
  // On peut s'en servir pour conserver pour chaque maille grossière la liste des mailles
  // raffinées
  m_coarse_cells.resize(nb_coarse_cell);
  m_refined_cells.resize(nb_coarse_cell, 4);

  {
    CellInfoListView cells(mesh->cellFamily());
    UniqueArray<Int32> first_child_cell_local_ids(nb_coarse_cell);
    cell_family->itemsUniqueIdToLocalId(first_child_cell_local_ids, first_child_cell_unique_ids);
    Int32 coarse_index = 0;
    std::array<Int32, 4> sub_cell_lids_container;
    ArrayView<Int32> sub_cell_lids(sub_cell_lids_container);

    ENUMERATE_ (Cell, icell, mesh->ownCells()) {
      Cell coarse_cell = *icell;
      if (!(coarse_cell.itemBase().flags() & ItemFlags::II_UserMark1))
        continue;
      // Supprime le flag
      coarse_cell.mutableItemBase().removeFlags(ItemFlags::II_UserMark1);
      m_coarse_cells[coarse_index] = coarse_cell.itemLocalId();
      Cell first_child_cell = cells[first_child_cell_local_ids[coarse_index]];
      // A partir de la première sous-maille, on peut connaître les 3 autres
      // car elles sont respectivement à droite, en haut à droite et en haut.
      sub_cell_lids[0] = first_child_cell.localId();
      sub_cell_lids[1] = cdm_x[first_child_cell].next().localId();
      sub_cell_lids[2] = cdm_y[CellLocalId(sub_cell_lids[1])].next().localId();
      sub_cell_lids[3] = cdm_y[first_child_cell].next().localId();
      if (is_verbose)
        info() << "AddChildForCoarseCell i=" << coarse_index << " coarse=" << ItemPrinter(coarse_cell)
               << " children_lid=" << sub_cell_lids;
      for (Int32 z = 0; z < 4; ++z) {
        CellLocalId sub_local_id = CellLocalId(sub_cell_lids[z]);
        m_refined_cells[coarse_index][z] = sub_local_id;
        if (is_verbose)
          info() << " AddParentCellToCell: z=" << z << " child=" << ItemPrinter(cells[sub_local_id]);
      }
      ++coarse_index;
    }
  }

  if (is_verbose) {
    ENUMERATE_ (Cell, icell, mesh->allCells()) {
      Cell cell = *icell;
      info() << "Final cell=" << ItemPrinter(cell) << " level=" << cell.level();
    }
  }

  const bool dump_coarse_mesh = false;
  if (dump_coarse_mesh) {
    String filename = String::format("mesh_coarse_{0}.svg", my_rank);
    std::ofstream ofile(filename.localstr());
    SimpleSVGMeshExporter writer(ofile);
    writer.write(mesh->allCells());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshCoarsening2::
removeRefinedCells()
{
  if (!m_is_create_coarse_called)
    ARCANE_FATAL("You need to call createCoarseCells() before");
  if (m_is_remove_refined_called)
    ARCANE_FATAL("This method has already been called");
  m_is_remove_refined_called = true;

  IMesh* mesh = m_cartesian_mesh->mesh();
  IMeshModifier* mesh_modifier = mesh->modifier();

  // Supprime toutes les mailles raffinées ainsi que toutes les mailles fantômes
  {
    std::unordered_set<Int32> coarse_cells_set;
    for (Int32 cell_lid : m_coarse_cells)
      coarse_cells_set.insert(cell_lid);

    UniqueArray<Int32> cells_to_remove;
    ENUMERATE_ (Cell, icell, mesh->ownCells()) {
      Int32 local_id = icell.itemLocalId();
      Cell cell = *icell;
      if (!cell.isOwn() || (coarse_cells_set.find(local_id) != coarse_cells_set.end()))
        cells_to_remove.add(local_id);
    }
    mesh_modifier->removeCells(cells_to_remove);
    mesh_modifier->endUpdate();
  }

  // Reconstruit les mailles fantômes
  mesh_modifier->setDynamic(true);
  mesh_modifier->updateGhostLayers();

  // Affiche les statistiques du nouveau maillage
  {
    MeshStats ms(traceMng(), mesh, mesh->parallelMng());
    ms.dumpStats();
  }

  _recomputeMeshGenerationInfo();

  // Il faut recalculer les nouvelles directions
  m_cartesian_mesh->computeDirections();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Recalcule les informations sur le nombre de mailles par direction.
 */
void CartesianMeshCoarsening2::
_recomputeMeshGenerationInfo()
{
  IMesh* mesh = m_cartesian_mesh->mesh();
  auto* cmgi = ICartesianMeshGenerationInfo::getReference(mesh, false);
  if (!cmgi)
    return;

  // Coefficient de dé-raffinement
  const Int32 cf = 2;

  {
    ConstArrayView<Int64> v = cmgi->ownCellOffsets();
    cmgi->setOwnCellOffsets(v[0] / cf, v[1] / cf, v[2] / cf);
  }
  {
    ConstArrayView<Int64> v = cmgi->globalNbCells();
    cmgi->setGlobalNbCells(v[0] / cf, v[1] / cf, v[2] / cf);
  }
  {
    ConstArrayView<Int32> v = cmgi->ownNbCells();
    cmgi->setOwnNbCells(v[0] / cf, v[1] / cf, v[2] / cf);
  }
  cmgi->setFirstOwnCellUniqueId(m_first_own_cell_unique_id_offset);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
