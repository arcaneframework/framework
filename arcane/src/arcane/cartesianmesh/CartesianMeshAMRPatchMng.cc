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
  m_cells_infos.reserve(cell_to_refine_internals.size() * (2 + num_mng.getNbNode()));

  Int64UniqueArray m_faces_infos;


  UniqueArray<Int64> ua_node_uid(num_mng.getNbNode());
  UniqueArray<Int64> ua_face_uid(num_mng.getNbFace());

  if(m_mesh->dimension() == 2) {
    m_faces_infos.reserve(cell_to_refine_internals.size() * 12 * (2 + 2));

    for (Cell cell : cell_to_refine_internals) {
      Int64 uid = cell.uniqueId();
      Int32 level = cell.level();
      Int64 coord_x = num_mng.uidToCoordX(uid, level);
      Int64 coord_y = num_mng.uidToCoordY(uid, level);

      Int64 ori_x = num_mng.getOffsetLevelToLevel(coord_x, level, level + 1);
      Int64 ori_y = num_mng.getOffsetLevelToLevel(coord_y, level, level + 1);

      Integer pattern = num_mng.getPattern();
      for (Integer i = ori_x; i < ori_x+pattern; ++i) {
        for (Integer j = ori_y; j < ori_y+pattern; ++j) {
          Int64 uid_child = num_mng.getCellUid(level+1, i, j);
          info() << "Test 1 -- x : " << i << " -- y : " << j << " -- level : " << level+1 << " -- uid : " << uid_child;

          num_mng.getNodeUids(ua_node_uid, level+1, i, j);

          for(Integer l = 0; l < num_mng.getNbNode(); ++l){
            info() << "Test 11 -- x : " << i << " -- y : " << j << " -- level : " << level+1 << " -- node : " << l << " -- uid_node : " << ua_node_uid[l];
          }

          num_mng.getFaceUids(ua_face_uid, level+1, i, j);

          for(Integer l = 0; l < num_mng.getNbFace(); ++l){
            info() << "Test 12 -- x : " << i << " -- y : " << j << " -- level : " << level+1 << " -- face : " << l << " -- uid_face : " << ua_face_uid[l];
          }

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
          }

        }
      }
    }
  }

  else if(m_mesh->dimension() == 3) {
    m_faces_infos.reserve(cell_to_refine_internals.size() * 36 * (2 + 4));
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
      for (Integer i = ori_x; i < ori_x+pattern; ++i) {
        for (Integer j = ori_y; j < ori_y+pattern; ++j) {
          for (Integer k = ori_z; k < ori_z+pattern; ++k) {
            Int64 uid_child = num_mng.getCellUid(level+1, i, j, k);
            info() << "Test 2 -- x : " << i << " -- y : " << j << " -- z : " << k << " -- level : " << level+1 << " -- uid : " << uid_child;

            num_mng.getNodeUids(ua_node_uid, level+1, i, j, k);

            for(Integer l = 0; l < num_mng.getNbNode(); ++l){
              info() << "Test 21 -- x : " << i << " -- y : " << j << " -- z : " << k << " -- level : " << level+1 << " -- node : " << l << " -- uid_node : " << ua_node_uid[l];
            }

            num_mng.getFaceUids(ua_face_uid, level+1, i, j, k);

            for(Integer l = 0; l < num_mng.getNbFace(); ++l){
              info() << "Test 22 -- x : " << i << " -- y : " << j << " -- z : " << k << " -- level : " << level+1 << " -- face : " << l << " -- uid_face : " << ua_face_uid[l];
            }

            Integer type_cell = IT_Hexaedron8;
            Integer type_face = IT_Quad4;

            m_cells_infos.add(type_cell);
            m_cells_infos.add(uid_child);
            for (Integer nc = 0; nc < num_mng.getNbNode(); nc++) {
              m_cells_infos.add(ua_node_uid[nc]);
            }

//            // Partie Face.
//            // TODO : Face doublon entre les parents.
//            Integer begin = (j == ori_y ? 0 : 1);
//            Integer end = (i == ori_x ? num_mng.getNbFace() : num_mng.getNbFace()-1);
//
//            for(Integer l = begin; l < end; ++l){
//              m_faces_infos.add(type_face);
//              m_faces_infos.add(ua_face_uid[l]);
//              for (Integer nc = l; nc < l+2; nc++) {
//                m_faces_infos.add(ua_node_uid[nc]);
//              }
//            }
          }
        }
      }
    }
  }
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
