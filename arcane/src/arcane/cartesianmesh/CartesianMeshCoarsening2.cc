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
#include "arcane/core/IGhostLayerMng.h"

#include "arcane/mesh/CellFamily.h"

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
_writeMeshSVG(const String& name)
{
  if (m_verbosity_level<=0)
    return;
  IMesh* mesh = m_cartesian_mesh->mesh();
  IParallelMng* pm = mesh->parallelMng();
  const Int32 mesh_rank = pm->commRank();
  String filename = String::format("mesh_{0}_{1}.svg", name, mesh_rank);
  info() << "WriteMesh name=" << filename;
  std::ofstream ofile(filename.localstr());
  SimpleSVGMeshExporter writer(ofile);
  writer.write(mesh->allCells());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Double la couche de mailles fantômes du maillage initial.
 *
 * Cela permettra ensuite d'avoir la bonne valeur de couches de mailles
 * fantômes pour le maillage final grossier.
 */
void CartesianMeshCoarsening2::
_doDoubleGhostLayers()
{
  IMesh* mesh = m_cartesian_mesh->mesh();
  IMeshModifier* mesh_modifier = mesh->modifier();
  IGhostLayerMng* gm = mesh->ghostLayerMng();
  // Il faut au moins utiliser la version 3 pour pouvoir supporter
  // plusieurs couches de mailles fantômes
  Int32 version = gm->builderVersion();
  if (version < 3)
    gm->setBuilderVersion(3);
  Int32 nb_ghost_layer = gm->nbGhostLayer();
  gm->setNbGhostLayer(nb_ghost_layer * 2);
  mesh_modifier->setDynamic(true);
  mesh_modifier->updateGhostLayers();
  // Remet le nombre initial de couches de mailles fantômes
  gm->setNbGhostLayer(nb_ghost_layer);

  // Comme le nombre d'entités a été modifié, il faut recalculer les directions
  m_cartesian_mesh->computeDirections();

  // Écrit le nouveau maillage
  _writeMeshSVG("double_ghost");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshCoarsening2::
createCoarseCells()
{
  const bool is_verbose = m_verbosity_level > 0;
  IMesh* mesh = m_cartesian_mesh->mesh();
  Integer nb_patch = m_cartesian_mesh->nbPatch();
  if (nb_patch != 1)
    ARCANE_FATAL("This method is only valid for 1 patch (nb_patch={0})", nb_patch);

  if (!mesh->isAmrActivated())
    ARCANE_FATAL("AMR is not activated for this case");

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

  _writeMeshSVG("orig");

  // Double la couche de mailles fantômes
  _doDoubleGhostLayers();

  // Calcul l'offset pour la création des uniqueId().
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

  if (is_verbose) {
    ENUMERATE_ (Cell, icell, mesh->allCells()) {
      Cell cell = *icell;
      info() << "Orig cell=" << ItemPrinter(cell) << " Face0=" << cell.face(0).uniqueId()
             << " Face1=" << cell.face(1).uniqueId() << " level=" << cell.level();
      for (Node node : cell.nodes()) {
        info() << "NodeUid=" << node.uniqueId();
      }
    }
  }

  // TODO: Calculer en avance le nombre de faces et de mailles et allouer en conséquence.
  UniqueArray<Int64> faces_infos;
  UniqueArray<Int64> cells_infos;
  Int32 nb_coarse_face = 0;
  Int32 nb_coarse_cell = 0;
  //! Liste de la première fille de chaque maille grossière
  UniqueArray<Cell> first_child_cells;

  UniqueArray<Int64> refined_cells_lids;
  UniqueArray<Int64> coarse_cells_uids;
  UniqueArray<Int32> coarse_faces_owner;

  ENUMERATE_ (Cell, icell, mesh->allCells()) {
    Cell cell = *icell;
    Int64 cell_uid = cell.uniqueId();
    Int64x3 cell_xy = refined_cell_uid_computer.compute(cell_uid);
    const Int64 cell_x = cell_xy.x;
    const Int64 cell_y = cell_xy.y;
    // Nécessaire pour le recalcul des mailles fantômes. On considère ces
    // mailles comme si elles venaient juste d'être raffinées.
    cell.mutableItemBase().addFlags(ItemFlags::II_JustRefined);
    // Comme on déraffine par 2, ne prend que les mailles dont les coordoonnées
    // topologiques sont paires
    if ((cell_x % 2) != 0 || (cell_y % 2) != 0)
      continue;
    if (is_verbose)
      info() << "CellToCoarse refined_uid=" << cell_uid << " x=" << cell_x << " y=" << cell_y;
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
      Int64 node_uid0 = node_uids[lface.node(0)];
      Int64 node_uid1 = node_uids[lface.node(1)];

      if (node_uid0 > node_uid1)
        std::swap(node_uid0, node_uid1);
      if (is_verbose)
        info() << "ADD_FACE uid=" << coarse_face_uids[z] << " n0=" << node_uid0 << " n1=" << node_uid1;
      faces_infos.add(node_uid0);
      faces_infos.add(node_uid1);
      ++nb_coarse_face;
    }
    // Ajoute la maille
    {
      cells_infos.add(IT_Quad4);
      Int64 coarse_cell_uid = coarse_cell_uid_computer.compute(coarse_cell_x, coarse_cell_y);
      cells_infos.add(coarse_cell_uid);
      if (is_verbose)
        info() << "CoarseCellUid=" << coarse_cell_uid;
      coarse_cells_uids.add(coarse_cell_uid);
      for (Int32 z = 0; z < 4; ++z)
        cells_infos.add(node_uids[z]);
      ++nb_coarse_cell;
      first_child_cells.add(cell);
    }
    // A partir de la première sous-maille, on peut connaitre les 3 autres
    // car elles sont respectivement à droite, en haut à droite et en haut.
    {
      std::array<Int32, 4> sub_cell_lids_container;
      ArrayView<Int32> sub_lids(sub_cell_lids_container);
      Cell cell1 = cdm_x[cell].next();
      // Vérifie la validité des sous-mailles.
      // Normalement il ne devrait pas y avoir de problèmes sauf si le
      // nombre de mailles dans chaque direction du sous-domaine
      // n'est pas un nombre pair.
      if (cell1.null())
        ARCANE_FATAL("Bad right cell for cell {0}", ItemPrinter(cell));
      Cell cell2 = cdm_y[cell1].next();
      if (cell2.null())
        ARCANE_FATAL("Bad upper right cell for cell {0}", ItemPrinter(cell));
      Cell cell3 = cdm_y[cell].next();
      if (cell3.null())
        ARCANE_FATAL("Bad upper cell for cell {0}", ItemPrinter(cell));
      sub_lids[0] = cell.localId();
      sub_lids[1] = cell1.localId();
      sub_lids[2] = cell2.localId();
      sub_lids[3] = cell3.localId();
      // Il faudra donner un propriétaire aux faces.
      // Ces nouvelles faces auront le même propriétaire que les faces raffinées
      // auquelles elles correspondent
      //info() << "CELL_NB_FACE=" << cell.nbFace() << " " << cell1.nbFace() << " " << cell2.nbFace() << " " << cell3.nbFace();
      coarse_faces_owner.add(cell.face(0).owner());
      coarse_faces_owner.add(cell1.face(1).owner());
      coarse_faces_owner.add(cell2.face(2).owner());
      coarse_faces_owner.add(cell3.face(3).owner());
      for (Int32 i = 0; i < 4; ++i)
        refined_cells_lids.add(sub_lids[i]);
    }
  }

  UniqueArray<Int32> cells_local_ids;
  UniqueArray<Int32> faces_local_ids;
  cells_local_ids.resize(nb_coarse_cell);
  faces_local_ids.resize(nb_coarse_face);
  mesh->modifier()->addFaces(nb_coarse_face, faces_infos, faces_local_ids);
  mesh->modifier()->addCells(nb_coarse_cell, cells_infos, cells_local_ids);

  IItemFamily* cell_family = mesh->cellFamily();

  // Maintenant que les mailles grossières sont créées, il faut indiquer
  // qu'elles sont parentes.
  using mesh::CellFamily;
  CellInfoListView cells(mesh->cellFamily());
  CellFamily* true_cell_family = ARCANE_CHECK_POINTER(dynamic_cast<CellFamily*>(cell_family));
  std::array<Int32, 4> sub_cell_lids_container;
  ArrayView<Int32> sub_cell_lids(sub_cell_lids_container);
  for (Int32 i = 0; i < nb_coarse_cell; ++i) {
    Int32 coarse_cell_lid = cells_local_ids[i];
    Cell coarse_cell = cells[coarse_cell_lid];
    Cell first_child_cell = first_child_cells[i];
    // A partir de la première sous-maille, on peut connaitre les 3 autres
    // car elles sont respectivement à droite, en haut à droite et en haut.
    sub_cell_lids[0] = first_child_cell.localId();
    sub_cell_lids[1] = cdm_x[first_child_cell].next().localId();
    sub_cell_lids[2] = cdm_y[CellLocalId(sub_cell_lids[1])].next().localId();
    sub_cell_lids[3] = cdm_y[first_child_cell].next().localId();
    if (is_verbose)
      info() << "AddChildForCoarseCell i=" << i << " coarse=" << ItemPrinter(coarse_cell)
             << " children_lid=" << sub_cell_lids;
    for (Int32 z = 0; z < 4; ++z) {
      Cell child_cell = cells[sub_cell_lids[z]];
      if (is_verbose)
        info() << " AddParentCellToCell: z=" << z << " child=" << ItemPrinter(child_cell);
      true_cell_family->_addParentCellToCell(child_cell, coarse_cell);
    }
    true_cell_family->_addChildrenCellsToCell(coarse_cell, sub_cell_lids);
  }

  // Positionne les propriétaires des nouvelles mailles et faces
  {
    IItemFamily* face_family = mesh->faceFamily();
    Int64 sub_cell_index = 0;
    ENUMERATE_ (Cell, icell, cell_family->view(cells_local_ids)) {
      Cell cell = *icell;
      cell.mutableItemBase().setOwner(my_rank, my_rank);
      for (Int32 z = 0; z < 4; ++z) {
        cell.face(z).mutableItemBase().setOwner(coarse_faces_owner[sub_cell_index + z], my_rank);
      }
      sub_cell_index += 4;
    }
    cell_family->notifyItemsOwnerChanged();
    face_family->notifyItemsOwnerChanged();
  }

  mesh->modifier()->endUpdate();

  if (is_verbose) {
    ENUMERATE_ (Cell, icell, mesh->allCells()) {
      Cell cell = *icell;
      info() << "Final cell=" << ItemPrinter(cell) << " Face0=" << cell.face(0).uniqueId()
             << " Face1=" << cell.face(1).uniqueId() << " level=" << cell.level();
    }
  }

  _recomputeMeshGenerationInfo();

  // Affiche les statistiques du nouveau maillage
  {
    MeshStats ms(traceMng(), mesh, mesh->parallelMng());
    ms.dumpStats();
  }

  // Il faut recalculer les nouvelles directions
  m_cartesian_mesh->computeDirections();

  _writeMeshSVG("coarse");
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
