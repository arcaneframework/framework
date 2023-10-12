// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshAMRPatchMng.cc                                 (C) 2000-2023 */
/*                                                                           */
/* Gestionnaire de l'AMR par patch d'un maillage cartésien.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "CartesianMeshAMRPatchMng.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/Properties.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/MeshStats.h"
#include "arcane/core/ICartesianMeshGenerationInfo.h"
#include "arcane/core/MeshEvents.h"
#include "arcane/utils/Real3.h"
#include "arcane/cartesianmesh/CartesianMeshNumberingMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


CartesianMeshAMRPatchMng::
CartesianMeshAMRPatchMng(IMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
{

}

void CartesianMeshAMRPatchMng::
refine()
{
  CartesianMeshNumberingMng num_mng(m_mesh);

  UniqueArray<Cell> cell_to_refine_internals;
  ENUMERATE_CELL(icell,m_mesh->ownCells()) {
    Cell cell = *icell;
    if (cell.itemBase().flags() & ItemFlags::II_Refine) {
      cell_to_refine_internals.add(cell);
    }
  }

  Int64UniqueArray m_cells_infos;

  Int64UniqueArray m_faces_infos;
  Int64UniqueArray m_nodes_infos;

  // TODO : TRÈS Moche !
  Integer total_nb_cells = 0;
  Integer total_nb_nodes = 0;
  Integer total_nb_faces = 0;


  UniqueArray<Int64> ua_node_uid(num_mng.getNbNode());
  UniqueArray<Int64> ua_face_uid(num_mng.getNbFace());

  UniqueArray<Cell> parent_cells;

  if(m_mesh->dimension() == 2) {
    m_cells_infos.reserve(cell_to_refine_internals.size() * 4 * (2 + num_mng.getNbNode()));
    m_faces_infos.reserve(cell_to_refine_internals.size() * 12 * (2 + 2));
    m_nodes_infos.reserve(cell_to_refine_internals.size() * 9);

    for (Cell cell : cell_to_refine_internals) {
      Int64 uid = cell.uniqueId();
      Int32 level = cell.level();
      Int64 coord_x = num_mng.uidToCoordX(uid, level);
      Int64 coord_y = num_mng.uidToCoordY(uid, level);

      Int64 ori_x = num_mng.getOffsetLevelToLevel(coord_x, level, level + 1);
      Int64 ori_y = num_mng.getOffsetLevelToLevel(coord_y, level, level + 1);

      Integer pattern = num_mng.getPattern();
      for (Integer j = ori_y; j < ori_y+pattern; ++j) {
        for (Integer i = ori_x; i < ori_x+pattern; ++i) {
          parent_cells.add(cell);
          total_nb_cells++;
          Int64 uid_child = num_mng.getCellUid(level+1, i, j);
          info() << "Test 1 -- x : " << i << " -- y : " << j << " -- level : " << level+1 << " -- uid : " << uid_child;

          num_mng.getNodeUids(ua_node_uid, level+1, i, j);
          num_mng.getFaceUids(ua_face_uid, level+1, i, j);

          Integer type_cell = IT_Quad4;
          Integer type_face = IT_Line2;

          m_cells_infos.add(type_cell);
          m_cells_infos.add(uid_child);
          for (Integer nc = 0; nc < num_mng.getNbNode(); nc++) {
            m_cells_infos.add(ua_node_uid[nc]);
          }

          // Partie Face.
          // TODO : Face doublon entre les parents.
          Integer begin = (j == ori_y ? 0 : 1);
          Integer end = (i == ori_x ? num_mng.getNbFace() : num_mng.getNbFace()-1);

          for(Integer l = begin; l < end; ++l){
            m_faces_infos.add(type_face);
            m_faces_infos.add(ua_face_uid[l]);
            for (Integer nc = l; nc < l+2; nc++) {
              m_faces_infos.add(ua_node_uid[nc%num_mng.getNbNode()]);
            }
            info() << "Test 12 -- x : " << i << " -- y : " << j << " -- level : " << level+1 << " -- face : " << l << " -- uid_face : " << ua_face_uid[l];
            total_nb_faces++;
          }

          // Partie Node.
          // TODO : Node doublon entre les parents.

          begin = (j != ori_y ? 2 : (i != ori_x ? 1 : 0));
          end = (i == ori_x ? num_mng.getNbNode() : num_mng.getNbNode()-1);

          for(Integer l = begin; l < end; ++l){
            m_nodes_infos.add(ua_node_uid[l]);
            info() << "Test 11 -- x : " << i << " -- y : " << j << " -- level : " << level+1 << " -- node : " << l << " -- uid_node : " << ua_node_uid[l];
            total_nb_nodes++;
          }

        }
      }
    }
  }

  else if(m_mesh->dimension() == 3) {

    // TODO : Demander ça à num_mng !
    const bool node001[] = {false, true, true, false, false, true, true, false};
    const bool node010[] = {false, false, false, false, true, true, true, true};
    const bool node100[] = {false, false, true, true, false, false, true, true};

    const Integer nodes_in_face_0[] = {0, 1, 2, 3};
    const Integer nodes_in_face_1[] = {0, 3, 7, 4};
    const Integer nodes_in_face_2[] = {0, 1, 5, 4};
    const Integer nodes_in_face_3[] = {4, 5, 6, 7};
    const Integer nodes_in_face_4[] = {1, 2, 6, 5};
    const Integer nodes_in_face_5[] = {3, 2, 6, 7};
    Integer nb_nodes_in_face = 4;

    m_cells_infos.reserve(cell_to_refine_internals.size() * 8 * (2 + num_mng.getNbNode()));
    m_faces_infos.reserve(cell_to_refine_internals.size() * 36 * (2 + 4));
    m_nodes_infos.reserve(cell_to_refine_internals.size() * 27);
    for (Cell cell : cell_to_refine_internals) {
      Int64 uid = cell.uniqueId();
      Int32 level = cell.level();
      Int64 coord_x = num_mng.uidToCoordX(uid, level);
      Int64 coord_y = num_mng.uidToCoordY(uid, level);
      Int64 coord_z = num_mng.uidToCoordZ(uid, level);

      Int64 ori_x = num_mng.getOffsetLevelToLevel(coord_x, level, level + 1);
      Int64 ori_y = num_mng.getOffsetLevelToLevel(coord_y, level, level + 1);
      Int64 ori_z = num_mng.getOffsetLevelToLevel(coord_z, level, level + 1);

      Integer pattern = num_mng.getPattern();
      for (Integer k = ori_z; k < ori_z+pattern; ++k) {
        for (Integer j = ori_y; j < ori_y+pattern; ++j) {
          for (Integer i = ori_x; i < ori_x+pattern; ++i) {
            parent_cells.add(cell);
            total_nb_cells++;
            Int64 uid_child = num_mng.getCellUid(level+1, i, j, k);
            info() << "Test 2 -- x : " << i << " -- y : " << j << " -- z : " << k << " -- level : " << level+1 << " -- uid : " << uid_child;

            num_mng.getNodeUids(ua_node_uid, level+1, i, j, k);
            num_mng.getFaceUids(ua_face_uid, level+1, i, j, k);

            Integer type_cell = IT_Hexaedron8;
            Integer type_face = IT_Quad4;

            m_cells_infos.add(type_cell);
            m_cells_infos.add(uid_child);
            for (Integer nc = 0; nc < num_mng.getNbNode(); nc++) {
              m_cells_infos.add(ua_node_uid[nc]);
            }

            // Partie Face.
            // TODO : Face doublon entre les parents.

            for(Integer l = 0; l < num_mng.getNbFace(); ++l){
              // On évite des faces doublons au niveau de la maille raffinée.
              if(i != ori_x && l == 1) continue;
              if(j != ori_y && l == 2) continue;
              if(k != ori_z && l == 0) continue;

              m_faces_infos.add(type_face);
              m_faces_infos.add(ua_face_uid[l]);

              ConstArrayView<Integer> nodes_in_face_l;
              switch (l) {
              case 0:
                nodes_in_face_l = ConstArrayView<Integer>::create(nodes_in_face_0, nb_nodes_in_face);
                break;
              case 1:
                nodes_in_face_l = ConstArrayView<Integer>::create(nodes_in_face_1, nb_nodes_in_face);
                break;
              case 2:
                nodes_in_face_l = ConstArrayView<Integer>::create(nodes_in_face_2, nb_nodes_in_face);
                break;
              case 3:
                nodes_in_face_l = ConstArrayView<Integer>::create(nodes_in_face_3, nb_nodes_in_face);
                break;
              case 4:
                nodes_in_face_l = ConstArrayView<Integer>::create(nodes_in_face_4, nb_nodes_in_face);
                break;
              case 5:
                nodes_in_face_l = ConstArrayView<Integer>::create(nodes_in_face_5, nb_nodes_in_face);
                break;
              default:
                ARCANE_FATAL("Bizarre...");
              }
              info() << "Test 22 -- x : " << i << " -- y : " << j << " -- z : " << k << " -- level : " << level+1 << " -- face : " << l << " -- uid_face : " << ua_face_uid[l];
              for(Integer nc : nodes_in_face_l){
                m_faces_infos.add(ua_node_uid[nc]);
                //info() << "Test 221 -- x : " << i << " -- y : " << j << " -- z : " << k << " -- level : " << level+1 << " -- node : " << nc << " -- uid_node : " << ua_node_uid[nc];
              }
              total_nb_faces++;
            }

            // Partie Node.
            // TODO : Node doublon entre les parents.

            for(Integer l = 0; l < num_mng.getNbNode(); ++l){
              if( ((i == ori_x || node001[l]) && (j == ori_y || node010[l]) && (k == ori_z || node100[l])) ){
                m_nodes_infos.add(ua_node_uid[l]);
                info() << "Test 21 -- x : " << i << " -- y : " << j << " -- z : " << k << " -- level : " << level+1 << " -- node : " << l << " -- uid_node : " << ua_node_uid[l];
                total_nb_nodes++;
              }
            }
          }
        }
      }
    }
  }
  else{
    ARCANE_FATAL("Bad dimension");
  }

  Int32UniqueArray m_nodes_lid;
  Int32UniqueArray m_faces_lid;
  Int32UniqueArray m_cells_lid;

  // Nodes
  {
    info() << "total_nb_nodes : " << total_nb_nodes;
    m_nodes_lid.resize(total_nb_nodes);
    m_mesh->modifier()->addNodes(m_nodes_infos, m_nodes_lid);
    m_mesh->nodeFamily()->endUpdate();
    NodeInfoListView nodes(m_mesh->nodeFamily());
//    for (Integer i = 0; i < num_mng.getNbNode(); ++i) {
//      m_orig_nodes_coords[nodes[m_nodes_lid[i]]] = m_nodes_to_create_coords[i];
//    }
  }

  // Faces
  {
    info() << "total_nb_faces : " << total_nb_faces;
    m_faces_lid.resize(total_nb_faces);
    m_mesh->modifier()->addFaces(total_nb_faces, m_faces_infos, m_faces_lid);
  }

  // Cells
  {
    info() << "total_nb_cells : " << total_nb_cells;
    m_cells_lid.resize(total_nb_cells);
    m_mesh->modifier()->addCells(total_nb_cells, m_cells_infos, m_cells_lid);

    CellInfoListView cells(m_mesh->cellFamily());
    for (Integer i = 0; i < total_nb_cells; ++i){
      Cell child = cells[m_cells_lid[i]];
      child.mutableItemBase().addFlags(ItemFlags::II_JustAdded);
      m_mesh->modifier()->addParentCellToCell(child, parent_cells[i]);
      info() << "ParentCellToCell -- Child : " << child.uniqueId() << " -- Parent : " << parent_cells[i].uniqueId();
    }
  }
  m_mesh->modifier()->endUpdate();
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
