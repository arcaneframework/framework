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

#include "arcane/cartesianmesh/CartesianMeshCoarsening.h"

#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/CartesianGridDimension.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/SimpleSVGMeshExporter.h"
#include "arcane/core/ItemPrinter.h"

#include "arcane/mesh/CellFamily.h"

#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/CellDirectionMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshCoarsening::
CartesianMeshCoarsening(ICartesianMesh* m)
: TraceAccessor(m->traceMng())
, m_cartesian_mesh(m)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshCoarsening::
coarseCartesianMesh()
{
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

  for (Integer idir = 0; idir < nb_dir; ++idir) {
    CellDirectionMng cdm(m_cartesian_mesh->cellDirection(idir));
    Int32 nb_own_cell = cdm.ownNbCell();
    info() << "NB_OWN_CELL dir=" << idir << " n=" << nb_own_cell;
    if ((nb_own_cell % 2) != 0)
      ARCANE_FATAL("Invalid number of cells ({0}) for direction {1}. Should be a multiple of 2",
                   nb_own_cell, idir);
  }

  CellDirectionMng cdm_x(m_cartesian_mesh->cellDirection(0));
  CellDirectionMng cdm_y(m_cartesian_mesh->cellDirection(1));

  const Int64 global_nb_cell_x = cdm_x.globalNbCell();
  const Int64 global_nb_cell_y = cdm_y.globalNbCell();
  CartesianGridDimension refined_grid_dim(global_nb_cell_x, global_nb_cell_y);
  CartesianGridDimension coarse_grid_dim(global_nb_cell_x / 2, global_nb_cell_y / 2);
  const Int64 coarse_grid_cell_offset = 10000; // TODO: à calculer automatiquement
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
  UniqueArray<Cell> first_child_cells;
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
    info() << "CELLCoarse uid=" << cell_uid << " x=" << cell_x << " y=" << cell_y;
    const Int64 coarse_cell_x = cell_x / 2;
    const Int64 coarse_cell_y = cell_y / 2;
    std::array<Int64, 4> node_uids_container;
    ArrayView<Int64> node_uids(node_uids_container);
    node_uids[0] = refined_node_uid_computer.compute(cell_x + 0, cell_y + 0);
    node_uids[1] = refined_node_uid_computer.compute(cell_x + 2, cell_y + 0);
    node_uids[2] = refined_node_uid_computer.compute(cell_x + 2, cell_y + 2);
    node_uids[3] = refined_node_uid_computer.compute(cell_x + 0, cell_y + 2);
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
      first_child_cells.add(cell);
    }
  }

  UniqueArray<Int32> cells_local_ids;
  cells_local_ids.resize(nb_coarse_cell);
  mesh->modifier()->addFaces(nb_coarse_face, faces_infos);
  mesh->modifier()->addCells(nb_coarse_cell, cells_infos, cells_local_ids);

  // Maintenant que les mailles grossières sont créées, il faut indiquer
  // qu'elles sont parentes.
  IItemFamily* cell_family = mesh->cellFamily();
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
    info() << "AddChildForCoarseCell i=" << i << " coarse=" << ItemPrinter(coarse_cell)
           << " children_lid=" << sub_cell_lids;
    for (Int32 z = 0; z < 4; ++z) {
      Cell child_cell = cells[sub_cell_lids[z]];
      info() << " AddParentCellToCell: z=" << z << " child=" << ItemPrinter(child_cell);
      true_cell_family->_addParentCellToCell(child_cell, coarse_cell);
    }
    true_cell_family->_addChildrenCellsToCell(coarse_cell, sub_cell_lids);
  }

  mesh->modifier()->endUpdate();

  ENUMERATE_ (Cell, icell, mesh->ownCells()) {
    Cell cell = *icell;
    info() << "Final cell=" << ItemPrinter(cell) << " level=" << cell.level();
  }

  {
    const Int32 mesh_rank = pm->commRank();
    String filename = String::format("mesh_coarse_{0}.svg", mesh_rank);
    std::ofstream ofile(filename.localstr());
    SimpleSVGMeshExporter writer(ofile);
    writer.write(mesh->allCells());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
