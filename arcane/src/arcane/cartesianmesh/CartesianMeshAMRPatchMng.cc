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
#include "arcane/core/MeshEvents.h"
#include "arcane/cartesianmesh/CellDirectionMng.h"
#include "arcane/mesh/FaceFamily.h"
#include "arcane/cartesianmesh/CartesianMeshNumberingMng.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


CartesianMeshAMRPatchMng::
CartesianMeshAMRPatchMng(ICartesianMesh* cmesh)
: TraceAccessor(cmesh->mesh()->traceMng())
, m_cmesh(cmesh)
, m_mesh(cmesh->mesh())
, m_num_mng(Arccore::makeRef(new CartesianMeshNumberingMng(cmesh->mesh())))
, m_flag_cells_consistent(Arccore::makeRef(new VariableCellInteger(VariableBuildInfo(cmesh->mesh(), "FlagCellsConsistent"))))
{

}

void CartesianMeshAMRPatchMng::
flagCellToRefine(Int32ConstArrayView cells_lids)
{
  ItemInfoListView cells(m_mesh->cellFamily());
  for (int lid : cells_lids) {
    Item item = cells[lid];
    item.mutableItemBase().addFlags(ItemFlags::II_Refine);
  }
  _syncFlagCell();
}

void CartesianMeshAMRPatchMng::
_syncFlagCell()
{
  if (!m_mesh->parallelMng()->isParallel())
    return ;

  VariableCellInteger& flag_cells_consistent = (*m_flag_cells_consistent.get());
  ENUMERATE_(Cell, icell, m_mesh->ownCells()){
    Cell cell = *icell;
    flag_cells_consistent[cell] = cell.mutableItemBase().flags();
    debug() << "Send " << cell << " flag : " << cell.mutableItemBase().flags() << " II_Refine : " << (cell.itemBase().flags() & ItemFlags::II_Refine) << " -- II_Inactive : " << (cell.itemBase().flags() & ItemFlags::II_Inactive);
  }

  flag_cells_consistent.synchronize();

  ENUMERATE_(Cell, icell, m_mesh->allCells().ghost()){
    Cell cell = *icell;

    // On ajoute uniquement les flags qui nous interesse (pour éviter d'ajouter le flag "II_Own" par exemple).
    if(flag_cells_consistent[cell] & ItemFlags::II_Refine) {
      cell.mutableItemBase().setFlags(ItemFlags::II_Refine);
    }
    if(flag_cells_consistent[cell] & ItemFlags::II_Inactive) {
      cell.mutableItemBase().setFlags(ItemFlags::II_Inactive);
    }
    debug() << "After Compute " << cell << " flag : " << cell.mutableItemBase().flags() << " II_Refine : " << (cell.itemBase().flags() & ItemFlags::II_Refine) << " -- II_Inactive : " << (cell.itemBase().flags() & ItemFlags::II_Inactive);
  }
}


/*
 * Pour les commentaires de cette méthode, on considère le repère suivant :
 *       (top)
 *         y    (front)
 *         ^    z
*          |   /
 *         | /
 *  (left) ------->x (right)
 *  (rear)(bottom)
 */
void CartesianMeshAMRPatchMng::
refine()
{

  UniqueArray<Cell> cell_to_refine_internals;
  ENUMERATE_CELL(icell,m_mesh->ownActiveCells()) {
    Cell cell = *icell;
    if(cell.owner() != m_mesh->parallelMng()->commRank()) continue;
    if (cell.itemBase().flags() & ItemFlags::II_Refine) {
      cell_to_refine_internals.add(cell);
    }
  }

  Int64UniqueArray m_cells_infos;

  Int64UniqueArray m_faces_infos;
  Int64UniqueArray m_nodes_infos;

  Integer total_nb_cells = 0;
  Integer total_nb_nodes = 0;
  Integer total_nb_faces = 0;

  std::map<Int64, Int32> node_uid_to_owner;
  std::map<Int64, Int32> face_uid_to_owner;

  // Deux tableaux permettant de récupérer les uniqueIds des noeuds et des faces
  // de chaque maille à chaque appel à getNodeUids()/getFaceUids().
  UniqueArray<Int64> ua_node_uid(m_num_mng->getNbNode());
  UniqueArray<Int64> ua_face_uid(m_num_mng->getNbFace());

  // On doit enregistrer les parents de chaque enfant pour mettre à jour les connectivités
  // lors de la création des mailles.
  UniqueArray<Cell> parent_cells;

  if(m_mesh->dimension() == 2) {

    // Masques pour les cas "voisins enfants" et "voisins parents du même patch".
    // Ces masques permettent de savoir si l'on doit créer un noeud ou pas selon
    // les mailles autour.
    // Par exemple, si l'on est en train d'étudier une maille enfant et qu'il y a
    // une maille enfant à gauche, on ne doit pas créer les noeuds 0 et 3 (mask_node_if_cell_left[]) (car
    // ils ont déjà été créés par la maille à gauche).
    // Idem pour les mailles parentes voisines : si l'on est sur une maille enfant située
    // sur la partie gauche de la maille parente (mailles enfants 0 et 2 dans le cas d'un
    // pattern de raffinement = 2), qu'il y a une maille parente à gauche et que cette maille parente
    // est en train ((d'être raffiné et à notre sous-domaine) ou (qu'elle est inactive)), on applique
    // la règle mask_node_if_cell_left[] car les noeuds ont été créé par celle-ci et on veut éviter
    // les noeuds en doubles.
    // Ces masques permettent aussi de déterminer le propriétaire des noeuds dans
    // le cas de multiples sous-domaines.
    // Par exemple, si l'on est sur une maille enfant située
    // sur la partie gauche de la maille parente (mailles enfants 0 et 2 dans le cas d'un
    // pattern de raffinement = 2), qu'il y a une maille parente à gauche et que cette maille
    // parente (appartient à un autre sous-domaine) et (est en train d'être raffiné),
    // on crée ce noeud mais on lui donne comme propriétaire le processus à qui appartient
    // la maille parente à gauche.
    const bool mask_node_if_cell_left[] = {false, true, true, false};
    const bool mask_node_if_cell_bottom[] = {false, false, true, true};

    const bool mask_node_if_cell_right[] = {true, false, false, true};
    const bool mask_node_if_cell_top[] = {true, true, false, false};


    const bool mask_face_if_cell_left[] = {true, true, true, false};
    const bool mask_face_if_cell_bottom[] = {false, true, true, true};

    const bool mask_face_if_cell_right[] = {true, false, true, true};
    const bool mask_face_if_cell_top[] = {true, true, false, true};

    // Pour la taille :
    // - on a "cell_to_refine_internals.size() * 4" mailles enfants,
    // - pour chaque maille, on a 2 infos (type de maille et uniqueId de la maille)
    // - pour chaque maille, on a "m_num_mng->getNbNode()" uniqueIds (les uniqueId de chaque noeud de la maille).
    m_cells_infos.reserve((cell_to_refine_internals.size() * 4) * (2 + m_num_mng->getNbNode()));

    // Pour la taille, au maximum :
    // - on a "cell_to_refine_internals.size() * 12" faces
    // - pour chaque face, on a 2 infos (type de face et uniqueId de la face)
    // - pour chaque face, on a 2 uniqueIds de noeuds.
    m_faces_infos.reserve((cell_to_refine_internals.size() * 12) * (2 + 2));

    // Pour la taille, au maximum :
    // - on a (cell_to_refine_internals.size() * 9) uniqueIds de noeuds.
    m_nodes_infos.reserve(cell_to_refine_internals.size() * 9);

    for (Cell parent_cell : cell_to_refine_internals) {
      Int64 uid = parent_cell.uniqueId();
      Int32 level = parent_cell.level();

      Int64 parent_coord_x = m_num_mng->uidToCoordX(uid, level);
      Int64 parent_coord_y = m_num_mng->uidToCoordY(uid, level);

      Int64 child_coord_x = m_num_mng->getOffsetLevelToLevel(parent_coord_x, level, level + 1);
      Int64 child_coord_y = m_num_mng->getOffsetLevelToLevel(parent_coord_y, level, level + 1);

      Integer pattern = m_num_mng->getPattern();

      CellDirectionMng cdmx(m_cmesh->cellDirection(MD_DirX));
      CellDirectionMng cdmy(m_cmesh->cellDirection(MD_DirY));

      DirCell ccx(cdmx.cell(parent_cell));
      DirCell ccy(cdmy.cell(parent_cell));

      Cell left_parent_cell = ccx.previous();
      Cell right_parent_cell = ccx.next();
      Cell bottom_parent_cell = ccy.previous();
      Cell top_parent_cell = ccy.next();

      Cell left_bottom_parent_cell;
      if(!left_parent_cell.null() && !bottom_parent_cell.null()){
        DirCell ccx2(cdmx.cell(bottom_parent_cell));
        left_bottom_parent_cell = ccx2.previous();
      }

      debug() << "parent_cell : " << parent_cell
              << " -- left_cell : " << left_parent_cell
              << " -- right_cell : " << right_parent_cell
              << " -- bottom_cell : " << bottom_parent_cell
              << " -- top_cell : " << top_parent_cell
              << " -- left_bottom_cell : " << left_bottom_parent_cell;

      // On peut noter une différence entre "left" et "right" et entre "bottom" et "top".
      // En effet, au sein d'un même patch, se sont "left" et "bottom" qui crée les noeuds/faces
      // en communs, c'est pour ça que l'on n'intègre pas "II_Refine" dans "right" et "top".
      // (Toutes les mailles avec le flag "II_Refine" dans cette boucle sont dans le même patch).
      // Dans le cas des mailles "II_Inactive", les noeuds/faces ont déjà été créés, nous n'avons donc pas
      // besoin de les créer.
      //
      // Un autre cas à gérer est le cas où il y a une maille à gauche et/ou bas qui a le flag "II_Refine" (donc
      // dans notre patch) mais qui n'est pas à notre processus. Dans ce cas, on doit créer le noeud/face
      // mais en modifiant le propriétaire...
      bool is_parent_cell_left = (!left_parent_cell.null() && ( (left_parent_cell.itemBase().flags() & ItemFlags::II_Inactive) || (left_parent_cell.isOwn() && (left_parent_cell.itemBase().flags() & ItemFlags::II_Refine)) ));
      bool is_parent_cell_right = (!right_parent_cell.null() && (right_parent_cell.itemBase().flags() & ItemFlags::II_Inactive));
      bool is_parent_cell_bottom = (!bottom_parent_cell.null() && ( (bottom_parent_cell.itemBase().flags() & ItemFlags::II_Inactive) || (bottom_parent_cell.isOwn() && (bottom_parent_cell.itemBase().flags() & ItemFlags::II_Refine)) ));
      bool is_parent_cell_top = (!top_parent_cell.null() && (top_parent_cell.itemBase().flags() & ItemFlags::II_Inactive));

      // ... ce qui est possible grâce à ces deux booléens.
      bool is_ghost_parent_cell_left_same_patch = (!left_parent_cell.null() && !left_parent_cell.isOwn() && (left_parent_cell.itemBase().flags() & ItemFlags::II_Refine));
      bool is_ghost_parent_cell_bottom_same_patch = (!bottom_parent_cell.null() && !bottom_parent_cell.isOwn() && (bottom_parent_cell.itemBase().flags() & ItemFlags::II_Refine));

      debug() << "is_cell_left : " << is_parent_cell_left
              << " -- is_cell_right : " << is_parent_cell_right
              << " -- is_cell_bottom : " << is_parent_cell_bottom
              << " -- is_cell_top : " << is_parent_cell_top
              << " -- is_ghost_cell_left_same_patch : " << is_ghost_parent_cell_left_same_patch
              << " -- is_ghost_cell_bottom_same_patch : " << is_ghost_parent_cell_bottom_same_patch;


      for (Integer j = child_coord_y; j < child_coord_y + pattern; ++j) {
        for (Integer i = child_coord_x; i < child_coord_x + pattern; ++i) {
          parent_cells.add(parent_cell);
          total_nb_cells++;
          Int64 uid_child = m_num_mng->getCellUid(level+1, i, j);
          debug() << "Test 1 -- x : " << i << " -- y : " << j << " -- level : " << level+1 << " -- uid : " << uid_child;

          m_num_mng->getNodeUids(ua_node_uid, level+1, i, j);
          m_num_mng->getFaceUids(ua_face_uid, level+1, i, j);

          Integer type_cell = IT_Quad4;
          Integer type_face = IT_Line2;

          // Partie Cell.
          m_cells_infos.add(type_cell);
          m_cells_infos.add(uid_child);
          for (Integer nc = 0; nc < m_num_mng->getNbNode(); nc++) {
            m_cells_infos.add(ua_node_uid[nc]);
          }

          // Partie Face.
          for(Integer l = 0; l < m_num_mng->getNbFace(); ++l){
            // Si la maille enfant est sur la {gauche, droite, bas, haut} et que la maille parente à
            // {gauche, droite, bas, haut} ne nous intéresse pas, on crée la face "l".
            // Sinon, on applique le masque pour savoir si on doit créer la face ou pas
            // pour éviter les doublons entre deux mailles.
            if (
                ( (i == child_coord_x && !is_parent_cell_left) || (mask_face_if_cell_left[l]) )
                &&
                ( (i != (child_coord_x + pattern-1) || !is_parent_cell_right) || mask_face_if_cell_right[l] )
                &&
                ( (j == child_coord_y && !is_parent_cell_bottom) || (mask_face_if_cell_bottom[l]) )
                &&
                ( (j != (child_coord_y + pattern-1) || !is_parent_cell_top) || mask_face_if_cell_top[l] )
                )
            {
              m_faces_infos.add(type_face);
              m_faces_infos.add(ua_face_uid[l]);

              // Les noeuds de la face sont toujours les noeuds l et l+1
              // car on utilise la même exploration pour les deux cas.
              for (Integer nc = l; nc < l + 2; nc++) {
                m_faces_infos.add(ua_node_uid[nc % m_num_mng->getNbNode()]);
              }
              total_nb_faces++;


              Integer new_owner = -1;

              // Ici, on doit choisir le propriétaire selon si la maille à gauche/en dessous est à nous
              // ou pas.
              // À noter l'inversion du masque. En effet, dans ce cas, on doit changer le propriétaire
              // des faces à gauche/en bas, le masque étant prévu pour traiter uniquement les faces
              // qui ne sont pas à gauche/en bas (pour éviter les doublons), il suffit de l'inverser.
              if(i == child_coord_x && is_ghost_parent_cell_left_same_patch && (!mask_face_if_cell_left[l])){
                new_owner = left_parent_cell.owner();
              }
              else if(j == child_coord_y && is_ghost_parent_cell_bottom_same_patch && (!mask_face_if_cell_bottom[l])){
                new_owner = bottom_parent_cell.owner();
              }
              else{
                new_owner = parent_cell.owner();
              }

              face_uid_to_owner[ua_face_uid[l]] = new_owner;

              debug() << "Test 12 -- x : " << i << " -- y : " << j << " -- level : " << level + 1 << " -- face : " << l << " -- uid_face : " << ua_face_uid[l] << " -- owner : " << new_owner;
            }
          }

          // Partie Node.
          for(Integer l = 0; l < m_num_mng->getNbNode(); ++l) {
            // Si la maille enfant est sur la {gauche, droite, bas, haut} et que la maille parente à
            // {gauche, droite, bas, haut} ne nous intéresse pas, on crée le noeud "l".
            // Sinon, on applique le masque pour savoir si on doit créer le noeud ou pas
            // pour éviter les doublons entre deux mailles.
            if (
                ( (i == child_coord_x && !is_parent_cell_left) || (mask_node_if_cell_left[l]) )
                &&
                ( (i != (child_coord_x +pattern-1) || !is_parent_cell_right) || mask_node_if_cell_right[l] )
                &&
                ( (j == child_coord_y && !is_parent_cell_bottom) || (mask_node_if_cell_bottom[l]) )
                &&
                ( (j != (child_coord_y +pattern-1) || !is_parent_cell_top) || mask_node_if_cell_top[l] )
               )
            {
              m_nodes_infos.add(ua_node_uid[l]);
              total_nb_nodes++;

              Integer new_owner = -1;

              // Légère différence entre la partie face et ici. Il y a le cas du noeud 0 (en bas à gauche)
              // qui est créé par le propriétaire de la maille en bas à gauche.
              // Pour prendre en compte cette différence, on doit ajouter un cas qui est l'application
              // des deux masques : gauche et bas. Si on traverse ces deux masques, le propriétaire sera la
              // maille gauche/bas. Quatre propriétaires possibles.
              // (Et oui, en 3D, c'est encore plus amusant !)
              if(
                i == child_coord_x && is_ghost_parent_cell_left_same_patch && (!mask_node_if_cell_left[l])
                &&
                j == child_coord_y && is_ghost_parent_cell_bottom_same_patch && (!mask_node_if_cell_bottom[l])
              ){
                new_owner = left_bottom_parent_cell.owner();
              }

              else if(i == child_coord_x && is_ghost_parent_cell_left_same_patch && (!mask_node_if_cell_left[l])){
                new_owner = left_parent_cell.owner();
              }

              else if(j == child_coord_y && is_ghost_parent_cell_bottom_same_patch && (!mask_node_if_cell_bottom[l])){
                new_owner = bottom_parent_cell.owner();
              }

              else{
                new_owner = parent_cell.owner();
              }

              node_uid_to_owner[ua_node_uid[l]] = new_owner;

              debug() << "Test 11 -- x : " << i << " -- y : " << j << " -- level : " << level + 1 << " -- node : " << l << " -- uid_node : " << ua_node_uid[l] << " -- owner : " << new_owner;
            }
          }
        }
      }
    }
  }

  // Pour le 3D, c'est très ressemblant, juste un peu plus long. Je recopie les commentaires, mais avec quelques adaptations.
  else if(m_mesh->dimension() == 3) {

    // Masques pour les cas "voisins enfants" et "voisins parents du même patch".
    // Ces masques permettent de savoir si l'on doit créer un noeud ou pas selon
    // les mailles autour.
    // Par exemple, si l'on est en train d'étudier une maille enfant et qu'il y a
    // une maille enfant à gauche, on ne doit pas créer les noeuds 0, 3, 4, 7 (mask_node_if_cell_left[]) (car
    // ils ont déjà été créés par la maille à gauche).
    // Idem pour les mailles parentes voisines : si l'on est sur une maille enfant située
    // sur la partie gauche de la maille parente (mailles enfants 0, 2, 4, 6 dans le cas d'un
    // pattern de raffinement = 2), qu'il y a une maille parente à gauche et que cette maille parente
    // est en train ((d'être raffiné et à notre sous-domaine) ou (qu'elle est inactive)), on applique
    // la règle mask_node_if_cell_left[] car les noeuds ont été créé par celle-ci et on veut éviter
    // les noeuds en doubles.
    // Ces masques permettent aussi de déterminer le propriétaire des noeuds dans
    // le cas de multiples sous-domaines.
    // Par exemple, si l'on est sur une maille enfant située
    // sur la partie gauche de la maille parente (mailles enfants 0, 2, 4, 6 dans le cas d'un
    // pattern de raffinement = 2), qu'il y a une maille parente à gauche et que cette maille
    // parente (appartient à un autre sous-domaine) et (est en train d'être raffiné),
    // on crée ce noeud mais on lui donne comme propriétaire le processus à qui appartient
    // la maille parente à gauche.
    const bool mask_node_if_cell_left[] = {false, true, true, false, false, true, true, false};
    const bool mask_node_if_cell_bottom[] = {false, false, true, true, false, false, true, true};
    const bool mask_node_if_cell_rear[] = {false, false, false, false, true, true, true, true};

    const bool mask_node_if_cell_right[] = {true, false, false, true, true, false, false, true};
    const bool mask_node_if_cell_top[] = {true, true, false, false, true, true, false, false};
    const bool mask_node_if_cell_front[] = {true, true, true, true, false, false, false, false};


    const bool mask_face_if_cell_left[] = {true, false, true, true, true, true};
    const bool mask_face_if_cell_bottom[] = {true, true, false, true, true, true};
    const bool mask_face_if_cell_rear[] = {false, true, true, true, true, true};

    const bool mask_face_if_cell_right[] = {true, true, true, true, false, true};
    const bool mask_face_if_cell_top[] = {true, true, true, true, true, false};
    const bool mask_face_if_cell_front[] = {true, true, true, false, true, true};


    // Petite différence par rapport au 2D. Pour le 2D, la position des noeuds des faces
    // dans le tableau "ua_node_uid" est toujours pareil (l et l+1, voir le 2D).
    // Pour le 3D, ce n'est pas le cas donc on a des tableaux pour avoir une correspondance
    // entre les noeuds de chaque face et la position des noeuds dans le tableau "ua_node_uid".
    // (Exemple : pour la face 1 (même ordre d'énumération qu'Arcane), on doit prendre le
    // tableau "nodes_in_face_1" et donc les noeuds "ua_node_uid[0]", "ua_node_uid[3]",
    // "ua_node_uid[7]" et "ua_node_uid[4]").
    const Integer nodes_in_face_0[] = {0, 1, 2, 3};
    const Integer nodes_in_face_1[] = {0, 3, 7, 4};
    const Integer nodes_in_face_2[] = {0, 1, 5, 4};
    const Integer nodes_in_face_3[] = {4, 5, 6, 7};
    const Integer nodes_in_face_4[] = {1, 2, 6, 5};
    const Integer nodes_in_face_5[] = {3, 2, 6, 7};
    Integer nb_nodes_in_face = 4;

    // Pour la taille :
    // - on a "cell_to_refine_internals.size() * 8" mailles enfants,
    // - pour chaque maille, on a 2 infos (type de maille et uniqueId de la maille)
    // - pour chaque maille, on a "m_num_mng->getNbNode()" uniqueIds (les uniqueId de chaque noeud de la maille).
    m_cells_infos.reserve((cell_to_refine_internals.size() * 8) * (2 + m_num_mng->getNbNode()));

    // Pour la taille, au maximum :
    // - on a "cell_to_refine_internals.size() * 36" faces enfants,
    // - pour chaque face, on a 2 infos (type de face et uniqueId de la face)
    // - pour chaque face, on a 4 uniqueIds de noeuds.
    m_faces_infos.reserve((cell_to_refine_internals.size() * 36) * (2 + 4));

    // Pour la taille, au maximum :
    // - on a (cell_to_refine_internals.size() * 27) uniqueIds de noeuds.
    m_nodes_infos.reserve(cell_to_refine_internals.size() * 27);

    for (Cell cell : cell_to_refine_internals) {
      Int64 uid = cell.uniqueId();
      Int32 level = cell.level();
      Int64 parent_coord_x = m_num_mng->uidToCoordX(uid, level);
      Int64 parent_coord_y = m_num_mng->uidToCoordY(uid, level);
      Int64 parent_coord_z = m_num_mng->uidToCoordZ(uid, level);

      Int64 child_coord_x = m_num_mng->getOffsetLevelToLevel(parent_coord_x, level, level + 1);
      Int64 child_coord_y = m_num_mng->getOffsetLevelToLevel(parent_coord_y, level, level + 1);
      Int64 child_coord_z = m_num_mng->getOffsetLevelToLevel(parent_coord_z, level, level + 1);

      Integer pattern = m_num_mng->getPattern();

      CellDirectionMng cdmx(m_cmesh->cellDirection(MD_DirX));
      CellDirectionMng cdmy(m_cmesh->cellDirection(MD_DirY));
      CellDirectionMng cdmz(m_cmesh->cellDirection(MD_DirZ));

      DirCell ccx(cdmx.cell(cell));
      DirCell ccy(cdmy.cell(cell));
      DirCell ccz(cdmz.cell(cell));

      Cell left_parent_cell = ccx.previous();
      Cell right_parent_cell = ccx.next();

      Cell bottom_parent_cell = ccy.previous();
      Cell top_parent_cell = ccy.next();

      Cell rear_parent_cell = ccz.previous();
      Cell front_parent_cell = ccz.next();

      Cell left_bottom_rear_parent_cell;
      if(!left_parent_cell.null() && !bottom_parent_cell.null() && !rear_parent_cell.null()){
        DirCell ccz2(cdmz.cell(left_parent_cell));
        Cell tmp = ccz2.previous();
        DirCell ccy2(cdmy.cell(tmp));
        left_bottom_rear_parent_cell = ccy2.previous();
      }

      Cell left_bottom_parent_cell;
      if(!left_parent_cell.null() && !bottom_parent_cell.null()){
        DirCell ccx2(cdmx.cell(bottom_parent_cell));
        left_bottom_parent_cell = ccx2.previous();
      }

      Cell bottom_rear_parent_cell;
      if(!bottom_parent_cell.null() && !rear_parent_cell.null()){
        DirCell ccy2(cdmy.cell(rear_parent_cell));
        bottom_rear_parent_cell = ccy2.previous();
      }

      Cell rear_left_parent_cell;
      if(!rear_parent_cell.null() && !left_parent_cell.null()){
        DirCell ccz2(cdmz.cell(left_parent_cell));
        rear_left_parent_cell = ccz2.previous();
      }

      debug() << "cell : " << cell
              << " -- left_cell : " << left_parent_cell
              << " -- right_cell : " << right_parent_cell
              << " -- bottom_cell : " << bottom_parent_cell
              << " -- top_cell : " << top_parent_cell
              << " -- rear_cell : " << rear_parent_cell
              << " -- front_cell : " << front_parent_cell
              << " -- left_bottom_rear_cell : " << left_bottom_rear_parent_cell
              << " -- left_bottom_cell : " << left_bottom_parent_cell
              << " -- bottom_rear_cell : " << bottom_rear_parent_cell
              << " -- rear_left_cell : " << rear_left_parent_cell;

      // On peut noter une différence entre "left" et "right", entre "bottom" et "top" et entre "rear" et "front".
      // En effet, au sein d'un même patch, se sont "left", "bottom" et "rear" qui crée les noeuds/faces
      // en communs, c'est pour ça que l'on n'intègre pas "II_Refine" dans "right", "top" et "front".
      // (Toutes les mailles avec le flag "II_Refine" dans cette boucle sont dans le même patch).
      // Dans le cas des mailles "II_Inactive", les noeuds/faces ont déjà été créés, nous n'avons donc pas
      // besoin de les créer.
      //
      // Un autre cas à gérer est le cas où il y a une maille à gauche et/ou bas et/ou derrière qui a le flag "II_Refine" (donc
      // dans notre patch) mais qui n'est pas à notre processus. Dans ce cas, on doit créer le noeud/face
      // mais en modifiant le propriétaire...
      bool is_parent_cell_left = (!left_parent_cell.null() && ( (left_parent_cell.itemBase().flags() & ItemFlags::II_Inactive) || (left_parent_cell.isOwn() && (left_parent_cell.itemBase().flags() & ItemFlags::II_Refine)) ));
      bool is_parent_cell_right = (!right_parent_cell.null() && (right_parent_cell.itemBase().flags() & ItemFlags::II_Inactive));

      bool is_parent_cell_bottom = (!bottom_parent_cell.null() && ( (bottom_parent_cell.itemBase().flags() & ItemFlags::II_Inactive) || (bottom_parent_cell.isOwn() && (bottom_parent_cell.itemBase().flags() & ItemFlags::II_Refine)) ));
      bool is_parent_cell_top = (!top_parent_cell.null() && (top_parent_cell.itemBase().flags() & ItemFlags::II_Inactive));

      bool is_parent_cell_rear = (!rear_parent_cell.null() && ( (rear_parent_cell.itemBase().flags() & ItemFlags::II_Inactive) || (rear_parent_cell.isOwn() && (rear_parent_cell.itemBase().flags() & ItemFlags::II_Refine)) ));
      bool is_parent_cell_front = (!front_parent_cell.null() && (front_parent_cell.itemBase().flags() & ItemFlags::II_Inactive));

      // ... ce qui est possible grâce à ces trois booléens.
      bool is_ghost_parent_cell_left_same_patch = (!left_parent_cell.null() && !left_parent_cell.isOwn() && (left_parent_cell.itemBase().flags() & ItemFlags::II_Refine));
      bool is_ghost_parent_cell_bottom_same_patch = (!bottom_parent_cell.null() && !bottom_parent_cell.isOwn() && (bottom_parent_cell.itemBase().flags() & ItemFlags::II_Refine));
      bool is_ghost_parent_cell_rear_same_patch = (!rear_parent_cell.null() && !rear_parent_cell.isOwn() && (rear_parent_cell.itemBase().flags() & ItemFlags::II_Refine));

      debug() << "is_cell_left : " << is_parent_cell_left
              << " -- is_cell_right : " << is_parent_cell_right
              << " -- is_cell_bottom : " << is_parent_cell_bottom
              << " -- is_cell_top : " << is_parent_cell_top
              << " -- is_cell_rear : " << is_parent_cell_rear
              << " -- is_cell_front : " << is_parent_cell_front
              << " -- is_ghost_cell_left_same_patch : " << is_ghost_parent_cell_left_same_patch
              << " -- is_ghost_cell_bottom_same_patch : " << is_ghost_parent_cell_bottom_same_patch
              << " -- is_ghost_cell_rear_same_patch : " << is_ghost_parent_cell_rear_same_patch;


      for (Integer k = child_coord_z; k < child_coord_z + pattern; ++k) {
        for (Integer j = child_coord_y; j < child_coord_y + pattern; ++j) {
          for (Integer i = child_coord_x; i < child_coord_x + pattern; ++i) {
            parent_cells.add(cell);
            total_nb_cells++;
            Int64 uid_child = m_num_mng->getCellUid(level+1, i, j, k);
            debug() << "Test 2 -- x : " << i << " -- y : " << j << " -- z : " << k << " -- level : " << level+1 << " -- uid : " << uid_child;

            m_num_mng->getNodeUids(ua_node_uid, level+1, i, j, k);
            m_num_mng->getFaceUids(ua_face_uid, level+1, i, j, k);

            Integer type_cell = IT_Hexaedron8;
            Integer type_face = IT_Quad4;

            // Partie Cell.
            m_cells_infos.add(type_cell);
            m_cells_infos.add(uid_child);
            for (Integer nc = 0; nc < m_num_mng->getNbNode(); nc++) {
              m_cells_infos.add(ua_node_uid[nc]);
            }

            // Partie Face.
            for(Integer l = 0; l < m_num_mng->getNbFace(); ++l){
              // Si la maille enfant est sur la {gauche, droite, bas, haut, derrière, devant} et que la maille parente à
              // {gauche, droite, bas, haut, derrière, devant} ne nous intéresse pas, on crée la face "l".
              // Sinon, on applique le masque pour savoir si on doit créer la face ou pas
              // pour éviter les doublons entre deux mailles.
              if (
                ( (i == child_coord_x && !is_parent_cell_left) || (mask_face_if_cell_left[l]) )
                &&
                ( (i != (child_coord_x +pattern-1) || !is_parent_cell_right) || mask_face_if_cell_right[l] )
                &&
                ( (j == child_coord_y && !is_parent_cell_bottom) || (mask_face_if_cell_bottom[l]) )
                &&
                ( (j != (child_coord_y +pattern-1) || !is_parent_cell_top) || mask_face_if_cell_top[l] )
                &&
                ( (k == child_coord_z && !is_parent_cell_rear) || (mask_face_if_cell_rear[l]) )
                &&
                ( (k != (child_coord_z +pattern-1) || !is_parent_cell_front) || mask_face_if_cell_front[l] )
              ){
                m_faces_infos.add(type_face);
                m_faces_infos.add(ua_face_uid[l]);

                // On récupère la position des noeuds de la face dans le tableau "ua_node_uid".
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
                for (Integer nc : nodes_in_face_l) {
                  m_faces_infos.add(ua_node_uid[nc]);
                }
                total_nb_faces++;


                Integer new_owner = -1;

                // Ici, on doit choisir le propriétaire selon si la maille à gauche et/ou en dessous et/ou derrière est à nous
                // ou pas.
                // À noter l'inversion du masque. En effet, dans ce cas, on doit changer le propriétaire
                // des faces à gauche et/ou en bas et/ou derrière, le masque étant prévu pour traiter uniquement les faces
                // qui ne sont pas à gauche et/ou en bas et/ou derrière (pour éviter les doublons), il suffit de l'inverser.
                if(i == child_coord_x && is_ghost_parent_cell_left_same_patch && (!mask_face_if_cell_left[l])){
                  new_owner = left_parent_cell.owner();
                }
                else if(j == child_coord_y && is_ghost_parent_cell_bottom_same_patch && (!mask_face_if_cell_bottom[l])){
                  new_owner = bottom_parent_cell.owner();
                }
                else if(k == child_coord_z && is_ghost_parent_cell_rear_same_patch && (!mask_face_if_cell_rear[l])){
                  new_owner = rear_parent_cell.owner();
                }
                else{
                  new_owner = cell.owner();
                }

                face_uid_to_owner[ua_face_uid[l]] = new_owner;

                debug() << "Test 22 -- x : " << i << " -- y : " << j << " -- z : " << k << " -- level : " << level + 1 << " -- face : " << l << " -- uid_face : " << ua_face_uid[l] << " -- owner : " << new_owner;
              }
            }


            // Partie Node.
            for(Integer l = 0; l < m_num_mng->getNbNode(); ++l){
              // Si la maille enfant est sur la {gauche, droite, bas, haut, derrière, devant} et que la maille parente à
              // {gauche, droite, bas, haut, derrière, devant} ne nous intéresse pas, on crée le noeud "l".
              // Sinon, on applique le masque pour savoir si on doit créer le noeud ou pas
              // pour éviter les doublons entre deux mailles.
              if (
                ( (i == child_coord_x && !is_parent_cell_left) || (mask_node_if_cell_left[l]) )
                &&
                ( (i != (child_coord_x +pattern-1) || !is_parent_cell_right) || mask_node_if_cell_right[l] )
                &&
                ( (j == child_coord_y && !is_parent_cell_bottom) || (mask_node_if_cell_bottom[l]) )
                &&
                ( (j != (child_coord_y +pattern-1) || !is_parent_cell_top) || mask_node_if_cell_top[l] )
                &&
                ( (k == child_coord_z && !is_parent_cell_rear) || (mask_node_if_cell_rear[l]) )
                &&
                ( (k != (child_coord_z +pattern-1) || !is_parent_cell_front) || mask_node_if_cell_front[l] )
              )
              {
                m_nodes_infos.add(ua_node_uid[l]);
                total_nb_nodes++;

                Integer new_owner = -1;

                // Par rapport au 2D, un noeud peut être lié à 8 mailles différentes. On regarde donc chaque
                // possibilité.
                if(
                  i == child_coord_x && is_ghost_parent_cell_left_same_patch && (!mask_node_if_cell_left[l])
                  &&
                  j == child_coord_y && is_ghost_parent_cell_bottom_same_patch && (!mask_node_if_cell_bottom[l])
                  &&
                  k == child_coord_z && is_ghost_parent_cell_rear_same_patch && (!mask_node_if_cell_rear[l])
                ){
                  new_owner = left_bottom_rear_parent_cell.owner();
                }

                else if(
                  i == child_coord_x && is_ghost_parent_cell_left_same_patch && (!mask_node_if_cell_left[l])
                  &&
                  j == child_coord_y && is_ghost_parent_cell_bottom_same_patch && (!mask_node_if_cell_bottom[l])
                ){
                  new_owner = left_bottom_parent_cell.owner();
                }

                else if(
                  j == child_coord_y && is_ghost_parent_cell_bottom_same_patch && (!mask_node_if_cell_bottom[l])
                  &&
                  k == child_coord_z && is_ghost_parent_cell_rear_same_patch && (!mask_node_if_cell_rear[l])
                ){
                  new_owner = bottom_rear_parent_cell.owner();
                }

                else if(
                  k == child_coord_z && is_ghost_parent_cell_rear_same_patch && (!mask_node_if_cell_rear[l])
                  &&
                  i == child_coord_x && is_ghost_parent_cell_left_same_patch && (!mask_node_if_cell_left[l])
                ){
                  new_owner = rear_left_parent_cell.owner();
                }

                else if(i == child_coord_x && is_ghost_parent_cell_left_same_patch && (!mask_node_if_cell_left[l])){
                  new_owner = left_parent_cell.owner();
                }

                else if(j == child_coord_y && is_ghost_parent_cell_bottom_same_patch && (!mask_node_if_cell_bottom[l])){
                  new_owner = bottom_parent_cell.owner();
                }

                else if(k == child_coord_z && is_ghost_parent_cell_rear_same_patch && (!mask_node_if_cell_rear[l])){
                  new_owner = rear_parent_cell.owner();
                }

                else{
                  new_owner = cell.owner();
                }

                node_uid_to_owner[ua_node_uid[l]] = new_owner;

                debug() << "Test 21 -- x : " << i << " -- y : " << j << " -- z : " << k << " -- level : " << level+1 << " -- node : " << l << " -- uid_node : " << ua_node_uid[l] << " -- owner : " << new_owner;
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
    debug() << "total_nb_nodes : " << total_nb_nodes;
    m_nodes_lid.resize(total_nb_nodes);
    m_mesh->modifier()->addNodes(m_nodes_infos, m_nodes_lid);

    ENUMERATE_ (Node, inode, m_mesh->nodeFamily()->view(m_nodes_lid)) {
      Node node = *inode;
      node.mutableItemBase().setOwner(node_uid_to_owner[node.uniqueId()], m_mesh->parallelMng()->commRank());

      if(node_uid_to_owner[node.uniqueId()] == m_mesh->parallelMng()->commRank()){
        node.mutableItemBase().addFlags(ItemFlags::II_Own);
      }

    }
    m_mesh->nodeFamily()->notifyItemsOwnerChanged();
  }

  // Faces
  {
    debug() << "total_nb_faces : " << total_nb_faces;
    m_faces_lid.resize(total_nb_faces);
    m_mesh->modifier()->addFaces(total_nb_faces, m_faces_infos, m_faces_lid);

    ENUMERATE_ (Face, iface, m_mesh->faceFamily()->view(m_faces_lid)) {
      Face face = *iface;
      face.mutableItemBase().setOwner(face_uid_to_owner[face.uniqueId()], m_mesh->parallelMng()->commRank());

      if(face_uid_to_owner[face.uniqueId()] == m_mesh->parallelMng()->commRank()){
        face.mutableItemBase().addFlags(ItemFlags::II_Own);
      }

    }
    m_mesh->faceFamily()->notifyItemsOwnerChanged();
  }

  // Cells
  {
    debug() << "total_nb_cells : " << total_nb_cells;
    m_cells_lid.resize(total_nb_cells);
    m_mesh->modifier()->addCells(total_nb_cells, m_cells_infos, m_cells_lid);

    CellInfoListView cells(m_mesh->cellFamily());
    for (Integer i = 0; i < total_nb_cells; ++i){
      Cell child = cells[m_cells_lid[i]];

      child.mutableItemBase().addFlags(ItemFlags::II_Own);
      child.mutableItemBase().addFlags(ItemFlags::II_JustAdded);
      if(parent_cells[i].itemBase().flags() & ItemFlags::II_Shared){
        child.mutableItemBase().addFlags(ItemFlags::II_Shared);
      }

      m_mesh->modifier()->addParentCellToCell(child, parent_cells[i]);
      m_mesh->modifier()->addChildCellToCell(parent_cells[i], child);
      debug() << "addParent/ChildCellToCell -- Child : " << child.uniqueId() << " -- Parent : " << parent_cells[i].uniqueId();
    }
    for(Cell cell : cell_to_refine_internals){
      cell.mutableItemBase().removeFlags(ItemFlags::II_Refine);
      cell.mutableItemBase().addFlags(ItemFlags::II_JustRefined);
      cell.mutableItemBase().addFlags(ItemFlags::II_Inactive);
    }
  }
  m_mesh->modifier()->endUpdate();

  {
    VariableNodeReal3& nodes_coords = m_mesh->nodesCoordinates();
    for(Cell parent_cell : cell_to_refine_internals){
      for(Integer i = 0; i < parent_cell.nbHChildren(); ++i){
        Cell child = parent_cell.hChild(i);
        m_num_mng->setNodeCoordinates(child);
        debug() << "setNodeCoordinates -- Child : " << child.uniqueId() << " -- Parent : " << parent_cell.uniqueId();
        for(Node node : child.nodes()){
          debug() << "\tChild Node : " << node.uniqueId() << " -- Coord : " << nodes_coords[node];
        }
      }
    }
  }




//  ENUMERATE_(Cell, icell, m_mesh->allCells()){
//    debug() << "\t" << *icell;
//    for(Node node : icell->nodes()){
//      debug() << "\t\t" << node;
//    }
//    for(Face face : icell->faces()){
//      debug() << "\t\t\t" << face;
//      if(face.uniqueId() == 14){
//        debug() << "\t\t\t\t" << face.backCell() << " " << face.frontCell();
//      }
//    }
//  }

  info() << "Résumé :";
  ENUMERATE_ (Cell, icell, m_mesh->allCells()){
    debug() << "\tCell uniqueId : " << icell->uniqueId() << " -- level : " << icell->level() << " -- nbChildren : " << icell->nbHChildren();
    for(Integer i = 0; i < icell->nbHChildren(); ++i){
      debug() << "\t\tChild uniqueId : " << icell->hChild(i).uniqueId() << " -- level : " << icell->hChild(i).level() << " -- nbChildren : " << icell->hChild(i).nbHChildren();
    }
  }

  //m_mesh->modifier()->setDynamic(true);
  //m_mesh->modifier()->updateGhostLayers();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
