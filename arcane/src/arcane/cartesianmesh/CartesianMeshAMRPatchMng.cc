// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshAMRPatchMng.cc                                 (C) 2000-2024 */
/*                                                                           */
/* Gestionnaire de l'AMR par patch d'un maillage cartésien.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "CartesianMeshAMRPatchMng.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/IMeshModifier.h"

#include "arcane/cartesianmesh/CellDirectionMng.h"
#include "arcane/cartesianmesh/CartesianMeshNumberingMng.h"
#include "arcane/utils/Array2View.h"
#include "arcane/utils/Array3View.h"

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
  IParallelMng* pm = m_mesh->parallelMng();
  Int32 nb_rank = pm->commSize();
  Int32 my_rank = pm->commRank();

  UniqueArray<Cell> cell_to_refine_internals;
  ENUMERATE_CELL(icell,m_mesh->ownActiveCells()) {
    Cell cell = *icell;
    if(cell.owner() != pm->commRank()) continue;
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

  std::map<Int32, UniqueArray<Int64>> get_back_face_owner;
  std::map<Int32, UniqueArray<Int64>> get_back_node_owner;

  for(Integer rank = 0; rank < nb_rank; ++rank){
    get_back_face_owner[rank].add(my_rank);
    get_back_face_owner[rank].add(0);
    get_back_node_owner[rank].add(my_rank);
    get_back_node_owner[rank].add(0);
  }

  // Deux tableaux permettant de récupérer les uniqueIds des noeuds et des faces
  // de chaque maille à chaque appel à getNodeUids()/getFaceUids().
  UniqueArray<Int64> ua_node_uid(m_num_mng->getNbNode());
  UniqueArray<Int64> ua_face_uid(m_num_mng->getNbFace());

  // On doit enregistrer les parents de chaque enfant pour mettre à jour les connectivités
  // lors de la création des mailles.
  UniqueArray<Cell> parent_cells;



  std::unordered_map<Int64, Integer> around_parent_cells_uid_to_owner;
  std::unordered_map<Int64, Int32> around_parent_cells_uid_to_flags;
  {
    Int32 usefull_flag = ItemFlags::II_Refine + ItemFlags::II_Inactive;
    {


      ENUMERATE_CELL (icell, m_mesh->ownCells()) {
        Cell cell = *icell;
        around_parent_cells_uid_to_owner[cell.uniqueId()] = my_rank;
        around_parent_cells_uid_to_flags[cell.uniqueId()] = (cell.itemBase().flags() & usefull_flag);
        debug() << "Proc : " << my_rank
                << " -- uid : " << icell->uniqueId()
                << (cell.itemBase().flags() & ItemFlags::II_Refine ? " -- II_Refine " : "")
                << (cell.itemBase().flags() & ItemFlags::II_Inactive ? " -- II_Inactive " : "")
        ;
      }

      UniqueArray<Int64> uid_of_cells_needed;
      UniqueArray<Int64> cell_uids_around((m_mesh->dimension() == 2) ? 9 : 27);
      for (Cell parent_cell : cell_to_refine_internals) {
        m_num_mng->getCellUidsAround(cell_uids_around, parent_cell);
        for (Int64 elem : cell_uids_around) {
          if (elem == -1)
            continue;
          // TODO C++20 : Mettre map.contains().
          if (around_parent_cells_uid_to_owner.find(elem) != around_parent_cells_uid_to_owner.end() && around_parent_cells_uid_to_owner[elem] == my_rank)
            continue;
          uid_of_cells_needed.add(elem);
        }
      }

      UniqueArray<Integer> size_ask(nb_rank);
      Integer sizeof_ask = uid_of_cells_needed.size();
      ArrayView<Integer> av(1, &sizeof_ask);
      pm->allGather(av, size_ask);

      UniqueArray<Int64> ask_all;
      pm->allGatherVariable(uid_of_cells_needed, ask_all);

      UniqueArray<Int32> flag_all(ask_all.size());
      UniqueArray<Int32> ask2_all(ask_all.size());

      UniqueArray<Int32> local_ids(ask_all.size());
      m_mesh->cellFamily()->itemsUniqueIdToLocalId(local_ids, ask_all, false);
      Integer compt = 0;
      ENUMERATE_ (Cell, icell, m_mesh->cellFamily()->view(local_ids)) {
        if(!icell->null() && icell->isOwn()) {
          ask2_all[compt] = my_rank;
          flag_all[compt] = (icell->itemBase().flags() & usefull_flag);
        }
        else {
          ask2_all[compt] = -1;
          flag_all[compt] = 0;
        }
        compt++;
      }

      ARCANE_ASSERT((compt == ask2_all.size()), ("Pb..."));
      ARCANE_ASSERT((compt == flag_all.size()), ("Pb..."));

      pm->reduce(Parallel::eReduceType::ReduceMax, ask2_all);
      pm->reduce(Parallel::eReduceType::ReduceMax, flag_all);

      Integer my_pos = 0;
      for (Integer i = 0; i < my_rank; ++i) {
        my_pos += size_ask[i];
      }

      ArrayView<Int32> reduced_ask = ask2_all.subView(my_pos, sizeof_ask);
      ArrayView<Int32> reduced_flag = flag_all.subView(my_pos, sizeof_ask);
      for (Integer i = 0; i < sizeof_ask; ++i) {
        around_parent_cells_uid_to_owner[uid_of_cells_needed[i]] = reduced_ask[i];
        around_parent_cells_uid_to_flags[uid_of_cells_needed[i]] = reduced_flag[i];
      }
    }

    {
      UniqueArray<Int64> around((m_mesh->dimension() == 2) ? 9 : 27);
      for (Cell parent_cell : cell_to_refine_internals) {
        m_num_mng->getCellUidsAround(around, parent_cell);
        debug() << around;
        for (Int64 elem : around) {
          if(around_parent_cells_uid_to_owner.find(elem) != around_parent_cells_uid_to_owner.end()){
            debug() << "Rank : " << my_rank
                    << " -- parent_cell : " << parent_cell.uniqueId()
                    << " -- around_cell : " << elem
                    << " -- owner : " << around_parent_cells_uid_to_owner[elem]
                    << " -- flags : " << around_parent_cells_uid_to_flags[elem]
                    << (around_parent_cells_uid_to_flags[elem] & ItemFlags::II_Refine ? " -- II_Refine" : "")
                    << (around_parent_cells_uid_to_flags[elem] & ItemFlags::II_Inactive ? " -- II_Inactive" : "")
            ;
          }
          else{
            debug() << "Rank : " << my_rank << " -- elem : -1";
          }
        }
      }
    }
  }



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





      
      UniqueArray<Int64> uid_cells_1d(9);
      m_num_mng->getCellUidsAround(uid_cells_1d, parent_cell);

      UniqueArray<Int32> owner_cells_1d(9);
      UniqueArray<Int32> flags_cells_1d(9);

      // Si uid_cell != -1 alors il y a peut-être une maille (mais on ne sait pas si elle est bien présente).
      // Si muo[uid_cell] != -1 alors il y a bien une maille.
      for(Integer i = 0; i < 9; ++i){
        Int64 uid_cell = uid_cells_1d[i];
        if(uid_cell != -1 && around_parent_cells_uid_to_owner[uid_cell] != -1) {
          owner_cells_1d[i] = around_parent_cells_uid_to_owner[uid_cell];
          flags_cells_1d[i] = around_parent_cells_uid_to_flags[uid_cell];
        }
        else{
          uid_cells_1d[i] = -1;
          owner_cells_1d[i] = -1;
          flags_cells_1d[i] = 0;
        }
      }

      Array2View uid_cells(uid_cells_1d.data(), 3, 3);
      Array2View owner_cells(owner_cells_1d.data(), 3, 3);
      Array2View flags_cells(flags_cells_1d.data(), 3, 3);

      info() << "uid_cells_1d : " << uid_cells_1d;
      info() << "owner_cells_1d : " << owner_cells_1d;
      info() << "flags_cells_1d : " << flags_cells_1d;












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
      bool is_parent_cell_left = ((uid_cells[1][0] != -1) && ((owner_cells[1][0] == my_rank && (flags_cells[1][0] & ItemFlags::II_Refine))));
      bool is_parent_cell_right = false;
      bool is_parent_cell_bottom = ((uid_cells[0][1] != -1) && ((owner_cells[0][1] == my_rank && (flags_cells[0][1] & ItemFlags::II_Refine))));
      bool is_parent_cell_top = false;

      // ... ce qui est possible grâce à ces deux booléens.
      // ┌─────────┐
      // │6   7   8│
      // └───────┐ │
      // ┌─┐ ┌─┐ │ │
      // │3│ │4│ │5│
      // │ │ └─┘ └─┘
      // │ └───────┐
      // │0   1   2│
      // └─────────┘
      // 4 = parent_cell
      bool is_ghost[3][3] = {{false}};

      is_ghost[0][0] = ((uid_cells[0][0] != -1) && (owner_cells[0][0] != my_rank) && (flags_cells[0][0] & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_ghost[0][1] = ((uid_cells[0][1] != -1) && (owner_cells[0][1] != my_rank) && (flags_cells[0][1] & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_ghost[0][2] = ((uid_cells[0][2] != -1) && (owner_cells[0][2] != my_rank) && (flags_cells[0][2] & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));

      is_ghost[1][0] = ((uid_cells[1][0] != -1) && (owner_cells[1][0] != my_rank) && (flags_cells[1][0] & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_ghost[1][2] = ((uid_cells[1][2] != -1) && (owner_cells[1][2] != my_rank) && (flags_cells[1][2] & ItemFlags::II_Inactive));

      is_ghost[2][0] = ((uid_cells[2][0] != -1) && (owner_cells[2][0] != my_rank) && (flags_cells[2][0] & ItemFlags::II_Inactive));
      is_ghost[2][1] = ((uid_cells[2][1] != -1) && (owner_cells[2][1] != my_rank) && (flags_cells[2][1] & ItemFlags::II_Inactive));
      is_ghost[2][2] = ((uid_cells[2][2] != -1) && (owner_cells[2][2] != my_rank) && (flags_cells[2][2] & ItemFlags::II_Inactive));



      debug() << "is_parent_cell_left : " << is_parent_cell_left
              << " -- is_parent_cell_right : " << is_parent_cell_right
              << " -- is_parent_cell_bottom : " << is_parent_cell_bottom
              << " -- is_parent_cell_top : " << is_parent_cell_top
              << " -- is_ghost[0][1] : " << is_ghost[1][0]
              << " -- is_ghost[1][0] : " << is_ghost[0][1]
              << " -- is_ghost[0][0] : " << is_ghost[0][0]
              << " -- is_ghost[2][0] : " << is_ghost[0][2]
              << " -- is_ghost[2][1] : " << is_ghost[1][2]
              << " -- is_ghost[1][2] : " << is_ghost[2][1]
              << " -- is_ghost[0][2] : " << is_ghost[2][0]
              << " -- is_ghost[2][2] : " << is_ghost[2][2]
      ;

      for (Int64 j = child_coord_y; j < child_coord_y + pattern; ++j) {
        for (Int64 i = child_coord_x; i < child_coord_x + pattern; ++i) {
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
              if(i == child_coord_x && is_ghost[1][0] && (!mask_face_if_cell_left[l])){
                new_owner = owner_cells[1][0];
              }
              else if(j == child_coord_y && is_ghost[0][1] && (!mask_face_if_cell_bottom[l])){
                new_owner = owner_cells[0][1];
              }

              else if(i == (child_coord_x + pattern-1) && is_ghost[1][2] && (!mask_face_if_cell_right[l])){
                get_back_face_owner[owner_cells[1][2]][1]++;
                get_back_face_owner[owner_cells[1][2]].add(ua_face_uid[l]);
                new_owner = parent_cell.owner();
              }
              else if(j == (child_coord_y + pattern-1) && is_ghost[2][1] && (!mask_face_if_cell_top[l])){
                get_back_face_owner[owner_cells[2][1]][1]++;
                get_back_face_owner[owner_cells[2][1]].add(ua_face_uid[l]);
                new_owner = parent_cell.owner();
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
            ( (i != (child_coord_x + pattern-1) || !is_parent_cell_right) || mask_node_if_cell_right[l] )
            &&
            ( (j == child_coord_y && !is_parent_cell_bottom) || (mask_node_if_cell_bottom[l]) )
            &&
            ( (j != (child_coord_y + pattern-1) || !is_parent_cell_top) || mask_node_if_cell_top[l] )
            )
            {
              m_nodes_infos.add(ua_node_uid[l]);
              total_nb_nodes++;

              Integer new_owner = -1;

              // Owner priority :
              // +---+---+
              // | 2 | 3 |
              // +---+---+
              // | 0 | 1 |
              // +---+---+

              // Légère différence entre la partie face et ici. Il y a le cas du noeud 0 (en bas à gauche)
              // qui est créé par le propriétaire de la maille en bas à gauche.
              // Pour prendre en compte cette différence, on doit ajouter un cas qui est l'application
              // des deux masques : gauche et bas. Si on traverse ces deux masques, le propriétaire sera la
              // maille gauche/bas. Quatre propriétaires possibles.
              // (Et oui, en 3D, c'est encore plus amusant !)
              if(i == child_coord_x && j == child_coord_y && (!mask_node_if_cell_left[l]) && (!mask_node_if_cell_bottom[l])){
                if(is_ghost[0][0]){
                  new_owner = owner_cells[0][0];
                }
                else if(is_ghost[0][1]){
                  new_owner = owner_cells[0][1];
                }
                else if(is_ghost[1][0]){
                  new_owner = owner_cells[1][0];
                }
                else{
                  new_owner = parent_cell.owner();
                }
              }
              // Noeud en bas à droite de la maille bas droite.
              else if(i == (child_coord_x + pattern-1) && j == child_coord_y && (!mask_node_if_cell_right[l]) && (!mask_node_if_cell_bottom[l])){
                if(is_ghost[0][1]){
                  new_owner = owner_cells[0][1];
                }
                else if(is_ghost[0][2]){
                  new_owner = owner_cells[0][2];
                }
                else {
                  if (is_ghost[1][2]) {
                    get_back_node_owner[owner_cells[1][2]][1]++;
                    get_back_node_owner[owner_cells[1][2]].add(ua_node_uid[l]);
                  }
                  new_owner = parent_cell.owner();
                }
              }

              else if(i == child_coord_x && j == (child_coord_y + pattern-1) && (!mask_node_if_cell_left[l]) && (!mask_node_if_cell_top[l])) {
                if(is_ghost[1][0]){
                  new_owner = owner_cells[1][0];
                }
                else {
                  if (is_ghost[2][0]) {
                    get_back_node_owner[owner_cells[2][0]][1]++;
                    get_back_node_owner[owner_cells[2][0]].add(ua_node_uid[l]);
                  }
                  if (is_ghost[2][1]) {
                    get_back_node_owner[owner_cells[2][1]][1]++;
                    get_back_node_owner[owner_cells[2][1]].add(ua_node_uid[l]);
                  }
                  new_owner = parent_cell.owner();
                }
              }

              else if(i == (child_coord_x + pattern-1) && j == (child_coord_y + pattern-1) && (!mask_node_if_cell_right[l]) && (!mask_node_if_cell_top[l])) {
                if(is_ghost[1][2]){
                  get_back_node_owner[owner_cells[1][2]][1]++;
                  get_back_node_owner[owner_cells[1][2]].add(ua_node_uid[l]);
                }
                if (is_ghost[2][1]) {
                  get_back_node_owner[owner_cells[2][1]][1]++;
                  get_back_node_owner[owner_cells[2][1]].add(ua_node_uid[l]);
                }
                if (is_ghost[2][2]) {
                  get_back_node_owner[owner_cells[2][2]][1]++;
                  get_back_node_owner[owner_cells[2][2]].add(ua_node_uid[l]);
                }
                new_owner = parent_cell.owner();
              }

              else if(i == child_coord_x && is_ghost[1][0] && (!mask_node_if_cell_left[l])){
                new_owner = owner_cells[1][0];
              }

              else if(j == child_coord_y && is_ghost[0][1] && (!mask_node_if_cell_bottom[l])){
                new_owner = owner_cells[0][1];
              }

              else if(i == (child_coord_x + pattern-1) && is_ghost[1][2] && (!mask_node_if_cell_right[l])){
                get_back_node_owner[owner_cells[1][2]][1]++;
                get_back_node_owner[owner_cells[1][2]].add(ua_node_uid[l]);
                new_owner = parent_cell.owner();
              }

              else if(j == (child_coord_y + pattern-1) && is_ghost[2][1] && (!mask_node_if_cell_top[l])){
                get_back_node_owner[owner_cells[2][1]][1]++;
                get_back_node_owner[owner_cells[2][1]].add(ua_node_uid[l]);
                new_owner = parent_cell.owner();
              }

              // Noeuds qui ne sont pas à la frontière.
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

    for (Cell parent_cell : cell_to_refine_internals) {
      Int64 uid = parent_cell.uniqueId();
      Int32 level = parent_cell.level();
      Int64 parent_coord_x = m_num_mng->uidToCoordX(uid, level);
      Int64 parent_coord_y = m_num_mng->uidToCoordY(uid, level);
      Int64 parent_coord_z = m_num_mng->uidToCoordZ(uid, level);

      Int64 child_coord_x = m_num_mng->getOffsetLevelToLevel(parent_coord_x, level, level + 1);
      Int64 child_coord_y = m_num_mng->getOffsetLevelToLevel(parent_coord_y, level, level + 1);
      Int64 child_coord_z = m_num_mng->getOffsetLevelToLevel(parent_coord_z, level, level + 1);

      Integer pattern = m_num_mng->getPattern();




      UniqueArray<Int64> uid_cells_1d(27);
      m_num_mng->getCellUidsAround(uid_cells_1d, parent_cell);

      UniqueArray<Int32> owner_cells_1d(27);
      UniqueArray<Int32> flags_cells_1d(27);

      // Si uid_cell != -1 alors il y a peut-être une maille (mais on ne sait pas si elle est bien présente).
      // Si muo[uid_cell] != -1 alors il y a bien une maille.
      for(Integer i = 0; i < 27; ++i){
        Int64 uid_cell = uid_cells_1d[i];
        if(uid_cell != -1 && around_parent_cells_uid_to_owner[uid_cell] != -1) {
          owner_cells_1d[i] = around_parent_cells_uid_to_owner[uid_cell];
          flags_cells_1d[i] = around_parent_cells_uid_to_flags[uid_cell];
        }
        else{
          uid_cells_1d[i] = -1;
          owner_cells_1d[i] = -1;
          flags_cells_1d[i] = 0;
        }
      }

      Array3View uid_cells(uid_cells_1d.data(), 3, 3, 3);
      Array3View owner_cells(owner_cells_1d.data(), 3, 3, 3);
      Array3View flags_cells(flags_cells_1d.data(), 3, 3, 3);

      info() << "uid_cells_1d : " << uid_cells_1d;
      info() << "owner_cells_1d : " << owner_cells_1d;
      info() << "flags_cells_1d : " << flags_cells_1d;


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
      bool is_parent_cell_left = ((uid_cells[1][1][0] != -1) && ((owner_cells[1][1][0] == my_rank && (flags_cells[1][1][0] & ItemFlags::II_Refine))));
      bool is_parent_cell_right = false;

      bool is_parent_cell_bottom = ((uid_cells[1][0][1] != -1) && ((owner_cells[1][0][1] == my_rank && (flags_cells[1][0][1] & ItemFlags::II_Refine))));
      bool is_parent_cell_top = false;

      bool is_parent_cell_rear = ((uid_cells[0][1][1] != -1) && ((owner_cells[0][1][1] == my_rank && (flags_cells[0][1][1] & ItemFlags::II_Refine))));
      bool is_parent_cell_front = false;

      // ... ce qui est possible grâce à ces trois booléens.
      bool is_ghost[3][3][3] = {{{false}}};
      //is_ghost[1][1][1]; //parent_cell

      is_ghost[0][0][0] = ((uid_cells[0][0][0] != -1) && (owner_cells[0][0][0] != my_rank) && (flags_cells[0][0][0] & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));//rear
      is_ghost[0][0][1] = ((uid_cells[0][0][1] != -1) && (owner_cells[0][0][1] != my_rank) && (flags_cells[0][0][1] & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));//rear
      is_ghost[0][0][2] = ((uid_cells[0][0][2] != -1) && (owner_cells[0][0][2] != my_rank) && (flags_cells[0][0][2] & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));//rear
      is_ghost[0][1][0] = ((uid_cells[0][1][0] != -1) && (owner_cells[0][1][0] != my_rank) && (flags_cells[0][1][0] & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));//rear
      is_ghost[0][1][1] = ((uid_cells[0][1][1] != -1) && (owner_cells[0][1][1] != my_rank) && (flags_cells[0][1][1] & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));//rear
      is_ghost[0][1][2] = ((uid_cells[0][1][2] != -1) && (owner_cells[0][1][2] != my_rank) && (flags_cells[0][1][2] & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));//rear
      is_ghost[0][2][0] = ((uid_cells[0][2][0] != -1) && (owner_cells[0][2][0] != my_rank) && (flags_cells[0][2][0] & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));//rear
      is_ghost[0][2][1] = ((uid_cells[0][2][1] != -1) && (owner_cells[0][2][1] != my_rank) && (flags_cells[0][2][1] & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));//rear
      is_ghost[0][2][2] = ((uid_cells[0][2][2] != -1) && (owner_cells[0][2][2] != my_rank) && (flags_cells[0][2][2] & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));//rear

      is_ghost[1][0][0] = ((uid_cells[1][0][0] != -1) && (owner_cells[1][0][0] != my_rank) && (flags_cells[1][0][0] & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_ghost[1][0][1] = ((uid_cells[1][0][1] != -1) && (owner_cells[1][0][1] != my_rank) && (flags_cells[1][0][1] & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_ghost[1][0][2] = ((uid_cells[1][0][2] != -1) && (owner_cells[1][0][2] != my_rank) && (flags_cells[1][0][2] & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_ghost[1][1][0] = ((uid_cells[1][1][0] != -1) && (owner_cells[1][1][0] != my_rank) && (flags_cells[1][1][0] & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_ghost[1][1][2] = ((uid_cells[1][1][2] != -1) && (owner_cells[1][1][2] != my_rank) && (flags_cells[1][1][2] & ItemFlags::II_Inactive));
      is_ghost[1][2][0] = ((uid_cells[1][2][0] != -1) && (owner_cells[1][2][0] != my_rank) && (flags_cells[1][2][0] & ItemFlags::II_Inactive));
      is_ghost[1][2][1] = ((uid_cells[1][2][1] != -1) && (owner_cells[1][2][1] != my_rank) && (flags_cells[1][2][1] & ItemFlags::II_Inactive));
      is_ghost[1][2][2] = ((uid_cells[1][2][2] != -1) && (owner_cells[1][2][2] != my_rank) && (flags_cells[1][2][2] & ItemFlags::II_Inactive));

      is_ghost[2][0][0] = ((uid_cells[2][0][0] != -1) && (owner_cells[2][0][0] != my_rank) && (flags_cells[2][0][0] & ItemFlags::II_Inactive));//rear
      is_ghost[2][0][1] = ((uid_cells[2][0][1] != -1) && (owner_cells[2][0][1] != my_rank) && (flags_cells[2][0][1] & ItemFlags::II_Inactive));//rear
      is_ghost[2][0][2] = ((uid_cells[2][0][2] != -1) && (owner_cells[2][0][2] != my_rank) && (flags_cells[2][0][2] & ItemFlags::II_Inactive));//rear
      is_ghost[2][1][0] = ((uid_cells[2][1][0] != -1) && (owner_cells[2][1][0] != my_rank) && (flags_cells[2][1][0] & ItemFlags::II_Inactive));//rear
      is_ghost[2][1][1] = ((uid_cells[2][1][1] != -1) && (owner_cells[2][1][1] != my_rank) && (flags_cells[2][1][1] & ItemFlags::II_Inactive));//rear
      is_ghost[2][1][2] = ((uid_cells[2][1][2] != -1) && (owner_cells[2][1][2] != my_rank) && (flags_cells[2][1][2] & ItemFlags::II_Inactive));//rear
      is_ghost[2][2][0] = ((uid_cells[2][2][0] != -1) && (owner_cells[2][2][0] != my_rank) && (flags_cells[2][2][0] & ItemFlags::II_Inactive));//rear
      is_ghost[2][2][1] = ((uid_cells[2][2][1] != -1) && (owner_cells[2][2][1] != my_rank) && (flags_cells[2][2][1] & ItemFlags::II_Inactive));//rear
      is_ghost[2][2][2] = ((uid_cells[2][2][2] != -1) && (owner_cells[2][2][2] != my_rank) && (flags_cells[2][2][2] & ItemFlags::II_Inactive));//rear



      debug() << "is_cell_left : " << is_parent_cell_left
              << " -- is_cell_right : " << is_parent_cell_right
              << " -- is_cell_bottom : " << is_parent_cell_bottom
              << " -- is_cell_top : " << is_parent_cell_top
              << " -- is_cell_rear : " << is_parent_cell_rear
              << " -- is_cell_front : " << is_parent_cell_front
              << " -- is_ghost_cell_left_same_patch : " << is_ghost[1][1][0]
              << " -- is_ghost_cell_bottom_same_patch : " << is_ghost[1][0][1]
              << " -- is_ghost_cell_rear_same_patch : " << is_ghost[0][1][1];


      for (Int64 k = child_coord_z; k < child_coord_z + pattern; ++k) {
        for (Int64 j = child_coord_y; j < child_coord_y + pattern; ++j) {
          for (Int64 i = child_coord_x; i < child_coord_x + pattern; ++i) {
            parent_cells.add(parent_cell);
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
              ( (i != (child_coord_x + pattern-1) || !is_parent_cell_right) || mask_face_if_cell_right[l] )
              &&
              ( (j == child_coord_y && !is_parent_cell_bottom) || (mask_face_if_cell_bottom[l]) )
              &&
              ( (j != (child_coord_y + pattern-1) || !is_parent_cell_top) || mask_face_if_cell_top[l] )
              &&
              ( (k == child_coord_z && !is_parent_cell_rear) || (mask_face_if_cell_rear[l]) )
              &&
              ( (k != (child_coord_z + pattern-1) || !is_parent_cell_front) || mask_face_if_cell_front[l] )
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
                if(i == child_coord_x && is_ghost[1][1][0] && (!mask_face_if_cell_left[l])){
                  new_owner = owner_cells[1][1][0];
                }
                else if(j == child_coord_y && is_ghost[1][0][1] && (!mask_face_if_cell_bottom[l])){
                  new_owner = owner_cells[1][0][1];
                }
                else if(k == child_coord_z && is_ghost[0][1][1] && (!mask_face_if_cell_rear[l])){
                  new_owner = owner_cells[0][1][1];
                }

                else if(i == (child_coord_x + pattern-1) && is_ghost[1][1][2] && (!mask_face_if_cell_right[l])){
                  get_back_face_owner[owner_cells[1][1][2]][1]++;
                  get_back_face_owner[owner_cells[1][1][2]].add(ua_face_uid[l]);
                  new_owner = parent_cell.owner();
                }
                else if(j == (child_coord_y + pattern-1) && is_ghost[1][2][1] && (!mask_face_if_cell_top[l])){
                  get_back_face_owner[owner_cells[1][2][1]][1]++;
                  get_back_face_owner[owner_cells[1][2][1]].add(ua_face_uid[l]);
                  new_owner = parent_cell.owner();
                }
                else if(k == (child_coord_z + pattern-1) && is_ghost[2][1][1] && (!mask_face_if_cell_front[l])){
                  get_back_face_owner[owner_cells[2][1][1]][1]++;
                  get_back_face_owner[owner_cells[2][1][1]].add(ua_face_uid[l]);
                  new_owner = parent_cell.owner();
                }

                else{
                  new_owner = parent_cell.owner();
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
                ( (i != (child_coord_x + pattern-1) || !is_parent_cell_right) || mask_node_if_cell_right[l] )
                &&
                ( (j == child_coord_y && !is_parent_cell_bottom) || (mask_node_if_cell_bottom[l]) )
                &&
                ( (j != (child_coord_y + pattern-1) || !is_parent_cell_top) || mask_node_if_cell_top[l] )
                &&
                ( (k == child_coord_z && !is_parent_cell_rear) || (mask_node_if_cell_rear[l]) )
                &&
                ( (k != (child_coord_z + pattern-1) || !is_parent_cell_front) || mask_node_if_cell_front[l] )
              )
              {
                m_nodes_infos.add(ua_node_uid[l]);
                total_nb_nodes++;

                Integer new_owner = -1;

                if(i == child_coord_x && j == child_coord_y && k == child_coord_z && (!mask_node_if_cell_left[l]) && (!mask_node_if_cell_bottom[l]) && (!mask_node_if_cell_rear[l]))
                {
                  if(is_ghost[0][0][0]){
                    new_owner = owner_cells[0][0][0];
                  }
                  else if(is_ghost[0][0][1]){
                    new_owner = owner_cells[0][0][1];
                  }
                  else if(is_ghost[0][1][0]){
                    new_owner = owner_cells[0][1][0];
                  }
                  else if(is_ghost[0][1][1]){
                    new_owner = owner_cells[0][1][1];
                  }
                  else if(is_ghost[1][0][0]){
                    new_owner = owner_cells[1][0][0];
                  }
                  else if(is_ghost[1][0][1]){
                    new_owner = owner_cells[1][0][1];
                  }
                  else if(is_ghost[1][1][0]){
                    new_owner = owner_cells[1][1][0];
                  }
                  else{
                    new_owner = parent_cell.owner();
                  }
                }

                else if(i == (child_coord_x + pattern-1) && j == child_coord_y && k == child_coord_z && (!mask_node_if_cell_right[l]) && (!mask_node_if_cell_bottom[l]) && (!mask_node_if_cell_rear[l]))
                {
                  if(is_ghost[0][0][1]){
                    new_owner = owner_cells[0][0][1];
                  }
                  else if(is_ghost[0][0][2]){
                    new_owner = owner_cells[0][0][2];
                  }
                  else if(is_ghost[0][1][1]){
                    new_owner = owner_cells[0][1][1];
                  }
                  else if(is_ghost[0][1][2]){
                    new_owner = owner_cells[0][1][2];
                  }
                  else if(is_ghost[1][0][1]){
                    new_owner = owner_cells[1][0][1];
                  }
                  else if(is_ghost[1][0][2]){
                    new_owner = owner_cells[1][0][2];
                  }
                  else{
                    if(is_ghost[1][1][2]){
                      get_back_node_owner[owner_cells[1][1][2]][1]++;
                      get_back_node_owner[owner_cells[1][1][2]].add(ua_node_uid[l]);
                    }
                    new_owner = parent_cell.owner();
                  }
                }

                else if(i == child_coord_x && j == (child_coord_y + pattern-1) && k == child_coord_z && (!mask_node_if_cell_left[l]) && (!mask_node_if_cell_top[l]) && (!mask_node_if_cell_rear[l]))
                {
                  if(is_ghost[0][1][0]){
                    new_owner = owner_cells[0][1][0];
                  }
                  else if(is_ghost[0][1][1]){
                    new_owner = owner_cells[0][1][1];
                  }
                  else if(is_ghost[0][2][0]){
                    new_owner = owner_cells[0][2][0];
                  }
                  else if(is_ghost[0][2][1]){
                    new_owner = owner_cells[0][2][1];
                  }
                  else if(is_ghost[1][1][0]){
                    new_owner = owner_cells[1][1][0];
                  }
                  else{
                    if(is_ghost[1][2][0]){
                      get_back_node_owner[owner_cells[1][2][0]][1]++;
                      get_back_node_owner[owner_cells[1][2][0]].add(ua_node_uid[l]);
                    }
                    if(is_ghost[1][2][1]){
                      get_back_node_owner[owner_cells[1][2][1]][1]++;
                      get_back_node_owner[owner_cells[1][2][1]].add(ua_node_uid[l]);
                    }
                    new_owner = parent_cell.owner();
                  }
                }

                else if(i == (child_coord_x + pattern-1) && j == (child_coord_y + pattern-1) && k == child_coord_z && (!mask_node_if_cell_right[l]) && (!mask_node_if_cell_top[l]) && (!mask_node_if_cell_rear[l]))
                {
                  if(is_ghost[0][1][1]){
                    new_owner = owner_cells[0][1][1];
                  }
                  else if(is_ghost[0][1][2]){
                    new_owner = owner_cells[0][1][2];
                  }
                  else if(is_ghost[0][2][1]){
                    new_owner = owner_cells[0][2][1];
                  }
                  else if(is_ghost[0][2][2]){
                    new_owner = owner_cells[0][2][2];
                  }
                  else{
                    if(is_ghost[1][1][2]){
                      get_back_node_owner[owner_cells[1][1][2]][1]++;
                      get_back_node_owner[owner_cells[1][1][2]].add(ua_node_uid[l]);
                    }
                    if(is_ghost[1][2][1]){
                      get_back_node_owner[owner_cells[1][2][1]][1]++;
                      get_back_node_owner[owner_cells[1][2][1]].add(ua_node_uid[l]);
                    }
                    if(is_ghost[1][2][2]){
                      get_back_node_owner[owner_cells[1][2][2]][1]++;
                      get_back_node_owner[owner_cells[1][2][2]].add(ua_node_uid[l]);
                    }
                    new_owner = parent_cell.owner();
                  }
                }

                else if(i == child_coord_x && j == child_coord_y && k == (child_coord_z + pattern-1) && (!mask_node_if_cell_left[l]) && (!mask_node_if_cell_bottom[l]) && (!mask_node_if_cell_front[l]))
                {
                  if(is_ghost[1][0][0]){
                    new_owner = owner_cells[1][0][0];
                  }
                  else if(is_ghost[1][0][1]){
                    new_owner = owner_cells[1][0][1];
                  }
                  else if(is_ghost[1][1][0]){
                    new_owner = owner_cells[1][1][0];
                  }
                  else{
                    if(is_ghost[2][0][0]){
                      get_back_node_owner[owner_cells[2][0][0]][1]++;
                      get_back_node_owner[owner_cells[2][0][0]].add(ua_node_uid[l]);
                    }
                    if(is_ghost[2][0][1]){
                      get_back_node_owner[owner_cells[2][0][1]][1]++;
                      get_back_node_owner[owner_cells[2][0][1]].add(ua_node_uid[l]);
                    }
                    if(is_ghost[2][1][0]){
                      get_back_node_owner[owner_cells[2][1][0]][1]++;
                      get_back_node_owner[owner_cells[2][1][0]].add(ua_node_uid[l]);
                    }
                    if(is_ghost[2][1][1]){
                      get_back_node_owner[owner_cells[2][1][1]][1]++;
                      get_back_node_owner[owner_cells[2][1][1]].add(ua_node_uid[l]);
                    }
                    new_owner = parent_cell.owner();
                  }
                }

                else if(i == (child_coord_x + pattern-1) && j == child_coord_y && k == (child_coord_z + pattern-1) && (!mask_node_if_cell_right[l]) && (!mask_node_if_cell_bottom[l]) && (!mask_node_if_cell_front[l]))
                {
                  if(is_ghost[1][0][1]){
                    new_owner = owner_cells[1][0][1];
                  }
                  else if(is_ghost[1][0][2]){
                    new_owner = owner_cells[1][0][2];
                  }
                  else{
                    if(is_ghost[1][1][2]){
                      get_back_node_owner[owner_cells[1][1][2]][1]++;
                      get_back_node_owner[owner_cells[1][1][2]].add(ua_node_uid[l]);
                    }
                    if(is_ghost[2][0][1]){
                      get_back_node_owner[owner_cells[2][0][1]][1]++;
                      get_back_node_owner[owner_cells[2][0][1]].add(ua_node_uid[l]);
                    }
                    if(is_ghost[2][0][2]){
                      get_back_node_owner[owner_cells[2][0][2]][1]++;
                      get_back_node_owner[owner_cells[2][0][2]].add(ua_node_uid[l]);
                    }
                    if(is_ghost[2][1][1]){
                      get_back_node_owner[owner_cells[2][1][1]][1]++;
                      get_back_node_owner[owner_cells[2][1][1]].add(ua_node_uid[l]);
                    }
                    if(is_ghost[2][1][2]){
                      get_back_node_owner[owner_cells[2][1][2]][1]++;
                      get_back_node_owner[owner_cells[2][1][2]].add(ua_node_uid[l]);
                    }
                    new_owner = parent_cell.owner();
                  }
                }

                else if(i == child_coord_x && j == (child_coord_y + pattern-1) && k == (child_coord_z + pattern-1) && (!mask_node_if_cell_left[l]) && (!mask_node_if_cell_top[l]) && (!mask_node_if_cell_front[l]))
                {
                  if(is_ghost[1][1][0]){
                    new_owner = owner_cells[1][1][0];
                  }
                  else{
                    if(is_ghost[1][2][0]){
                      get_back_node_owner[owner_cells[1][2][0]][1]++;
                      get_back_node_owner[owner_cells[1][2][0]].add(ua_node_uid[l]);
                    }
                    if(is_ghost[1][2][1]){
                      get_back_node_owner[owner_cells[1][2][1]][1]++;
                      get_back_node_owner[owner_cells[1][2][1]].add(ua_node_uid[l]);
                    }
                    if(is_ghost[2][1][0]){
                      get_back_node_owner[owner_cells[2][1][0]][1]++;
                      get_back_node_owner[owner_cells[2][1][0]].add(ua_node_uid[l]);
                    }
                    if(is_ghost[2][1][1]){
                      get_back_node_owner[owner_cells[2][1][1]][1]++;
                      get_back_node_owner[owner_cells[2][1][1]].add(ua_node_uid[l]);
                    }
                    if(is_ghost[2][2][0]){
                      get_back_node_owner[owner_cells[2][2][0]][1]++;
                      get_back_node_owner[owner_cells[2][2][0]].add(ua_node_uid[l]);
                    }
                    if(is_ghost[2][2][1]){
                      get_back_node_owner[owner_cells[2][2][1]][1]++;
                      get_back_node_owner[owner_cells[2][2][1]].add(ua_node_uid[l]);
                    }
                    new_owner = parent_cell.owner();
                  }
                }

                else if(i == (child_coord_x + pattern-1) && j == (child_coord_y + pattern-1) && k == (child_coord_z + pattern-1) && (!mask_node_if_cell_right[l]) && (!mask_node_if_cell_top[l]) && (!mask_node_if_cell_front[l]))
                {
                  if(is_ghost[1][1][2]){
                    get_back_node_owner[owner_cells[1][1][2]][1]++;
                    get_back_node_owner[owner_cells[1][1][2]].add(ua_node_uid[l]);
                  }
                  if(is_ghost[1][2][1]){
                    get_back_node_owner[owner_cells[1][2][1]][1]++;
                    get_back_node_owner[owner_cells[1][2][1]].add(ua_node_uid[l]);
                  }
                  if(is_ghost[1][2][2]){
                    get_back_node_owner[owner_cells[1][2][2]][1]++;
                    get_back_node_owner[owner_cells[1][2][2]].add(ua_node_uid[l]);
                  }
                  if(is_ghost[2][1][1]){
                    get_back_node_owner[owner_cells[2][1][1]][1]++;
                    get_back_node_owner[owner_cells[2][1][1]].add(ua_node_uid[l]);
                  }
                  if(is_ghost[2][1][2]){
                    get_back_node_owner[owner_cells[2][1][2]][1]++;
                    get_back_node_owner[owner_cells[2][1][2]].add(ua_node_uid[l]);
                  }
                  if(is_ghost[2][2][1]){
                    get_back_node_owner[owner_cells[2][2][1]][1]++;
                    get_back_node_owner[owner_cells[2][2][1]].add(ua_node_uid[l]);
                  }
                  if(is_ghost[2][2][2]){
                    get_back_node_owner[owner_cells[2][2][2]][1]++;
                    get_back_node_owner[owner_cells[2][2][2]].add(ua_node_uid[l]);
                  }
                  new_owner = parent_cell.owner();
                }





                else if(i == child_coord_x && j == child_coord_y && (!mask_node_if_cell_left[l]) && (!mask_node_if_cell_bottom[l])){
                  if(is_ghost[1][0][0]){
                    new_owner = owner_cells[1][0][0];
                  }
                  else if(is_ghost[1][0][1]){
                    new_owner = owner_cells[1][0][1];
                  }
                  else if(is_ghost[1][1][0]){
                    new_owner = owner_cells[1][1][0];
                  }
                  else{
                    new_owner = parent_cell.owner();
                  }
                }
                else if(i == (child_coord_x + pattern-1) && j == child_coord_y && (!mask_node_if_cell_right[l]) && (!mask_node_if_cell_bottom[l])){
                  if(is_ghost[1][0][1]){
                    new_owner = owner_cells[1][0][1];
                  }
                  else if(is_ghost[1][0][2]){
                    new_owner = owner_cells[1][0][2];
                  }
                  else {
                    if (is_ghost[1][1][2]) {
                      get_back_node_owner[owner_cells[1][1][2]][1]++;
                      get_back_node_owner[owner_cells[1][1][2]].add(ua_node_uid[l]);
                    }
                    new_owner = parent_cell.owner();
                  }
                }

                else if(i == child_coord_x && j == (child_coord_y + pattern-1) && (!mask_node_if_cell_left[l]) && (!mask_node_if_cell_top[l])) {
                  if(is_ghost[1][1][0]){
                    new_owner = owner_cells[1][1][0];
                  }
                  else {
                    if (is_ghost[1][2][0]) {
                      get_back_node_owner[owner_cells[1][2][0]][1]++;
                      get_back_node_owner[owner_cells[1][2][0]].add(ua_node_uid[l]);
                    }
                    if (is_ghost[1][2][1]) {
                      get_back_node_owner[owner_cells[1][2][1]][1]++;
                      get_back_node_owner[owner_cells[1][2][1]].add(ua_node_uid[l]);
                    }
                    new_owner = parent_cell.owner();
                  }
                }

                else if(i == (child_coord_x + pattern-1) && j == (child_coord_y + pattern-1) && (!mask_node_if_cell_right[l]) && (!mask_node_if_cell_top[l])) {
                  if(is_ghost[1][1][2]){
                    get_back_node_owner[owner_cells[1][1][2]][1]++;
                    get_back_node_owner[owner_cells[1][1][2]].add(ua_node_uid[l]);
                  }
                  if (is_ghost[1][2][1]) {
                    get_back_node_owner[owner_cells[1][2][1]][1]++;
                    get_back_node_owner[owner_cells[1][2][1]].add(ua_node_uid[l]);
                  }
                  if (is_ghost[1][2][2]) {
                    get_back_node_owner[owner_cells[1][2][2]][1]++;
                    get_back_node_owner[owner_cells[1][2][2]].add(ua_node_uid[l]);
                  }
                  new_owner = parent_cell.owner();
                }






                else if(i == child_coord_x && k == child_coord_z && (!mask_node_if_cell_left[l]) && (!mask_node_if_cell_rear[l])){
                  if(is_ghost[0][1][0]){
                    new_owner = owner_cells[0][1][0];
                  }
                  else if(is_ghost[0][1][1]){
                    new_owner = owner_cells[0][1][1];
                  }
                  else if(is_ghost[1][1][0]){
                    new_owner = owner_cells[1][1][0];
                  }
                  else{
                    new_owner = parent_cell.owner();
                  }
                }
                else if(i == (child_coord_x + pattern-1) && k == child_coord_z && (!mask_node_if_cell_right[l]) && (!mask_node_if_cell_rear[l])){
                  if(is_ghost[0][1][1]){
                    new_owner = owner_cells[0][1][1];
                  }
                  else if(is_ghost[0][1][2]){
                    new_owner = owner_cells[0][1][2];
                  }
                  else {
                    if (is_ghost[1][1][2]) {
                      get_back_node_owner[owner_cells[1][1][2]][1]++;
                      get_back_node_owner[owner_cells[1][1][2]].add(ua_node_uid[l]);
                    }
                    new_owner = parent_cell.owner();
                  }
                }

                else if(i == child_coord_x && k == (child_coord_z + pattern-1) && (!mask_node_if_cell_left[l]) && (!mask_node_if_cell_front[l])) {
                  if(is_ghost[1][1][0]){
                    new_owner = owner_cells[1][1][0];
                  }
                  else {
                    if (is_ghost[2][1][0]) {
                      get_back_node_owner[owner_cells[2][1][0]][1]++;
                      get_back_node_owner[owner_cells[2][1][0]].add(ua_node_uid[l]);
                    }
                    if (is_ghost[2][1][1]) {
                      get_back_node_owner[owner_cells[2][1][1]][1]++;
                      get_back_node_owner[owner_cells[2][1][1]].add(ua_node_uid[l]);
                    }
                    new_owner = parent_cell.owner();
                  }
                }

                else if(i == (child_coord_x + pattern-1) && k == (child_coord_z + pattern-1) && (!mask_node_if_cell_right[l]) && (!mask_node_if_cell_front[l])) {
                  if(is_ghost[1][1][2]){
                    get_back_node_owner[owner_cells[1][1][2]][1]++;
                    get_back_node_owner[owner_cells[1][1][2]].add(ua_node_uid[l]);
                  }
                  if (is_ghost[2][1][1]) {
                    get_back_node_owner[owner_cells[2][1][1]][1]++;
                    get_back_node_owner[owner_cells[2][1][1]].add(ua_node_uid[l]);
                  }
                  if (is_ghost[2][1][2]) {
                    get_back_node_owner[owner_cells[2][1][2]][1]++;
                    get_back_node_owner[owner_cells[2][1][2]].add(ua_node_uid[l]);
                  }
                  new_owner = parent_cell.owner();
                }






                else if(j == child_coord_y && k == child_coord_z && (!mask_node_if_cell_bottom[l]) && (!mask_node_if_cell_rear[l])){
                  if(is_ghost[0][0][1]){
                    new_owner = owner_cells[0][0][1];
                  }
                  else if(is_ghost[0][1][1]){
                    new_owner = owner_cells[0][1][1];
                  }
                  else if(is_ghost[1][0][1]){
                    new_owner = owner_cells[1][0][1];
                  }
                  else{
                    new_owner = parent_cell.owner();
                  }
                }
                else if(j == (child_coord_y + pattern-1) && k == child_coord_z && (!mask_node_if_cell_top[l]) && (!mask_node_if_cell_rear[l])){
                  if(is_ghost[0][1][1]){
                    new_owner = owner_cells[0][1][1];
                  }
                  else if(is_ghost[0][2][1]){
                    new_owner = owner_cells[0][2][1];
                  }
                  else {
                    if (is_ghost[1][2][1]) {
                      get_back_node_owner[owner_cells[1][2][1]][1]++;
                      get_back_node_owner[owner_cells[1][2][1]].add(ua_node_uid[l]);
                    }
                    new_owner = parent_cell.owner();
                  }
                }

                else if(j == child_coord_y && k == (child_coord_z + pattern-1) && (!mask_node_if_cell_bottom[l]) && (!mask_node_if_cell_front[l])) {
                  if(is_ghost[1][0][1]){
                    new_owner = owner_cells[1][0][1];
                  }
                  else {
                    if (is_ghost[2][0][1]) {
                      get_back_node_owner[owner_cells[2][0][1]][1]++;
                      get_back_node_owner[owner_cells[2][0][1]].add(ua_node_uid[l]);
                    }
                    if (is_ghost[2][1][1]) {
                      get_back_node_owner[owner_cells[2][1][1]][1]++;
                      get_back_node_owner[owner_cells[2][1][1]].add(ua_node_uid[l]);
                    }
                    new_owner = parent_cell.owner();
                  }
                }

                else if(j == (child_coord_y + pattern-1) && k == (child_coord_z + pattern-1) && (!mask_node_if_cell_top[l]) && (!mask_node_if_cell_front[l])) {
                  if(is_ghost[1][2][1]){
                    get_back_node_owner[owner_cells[1][2][1]][1]++;
                    get_back_node_owner[owner_cells[1][2][1]].add(ua_node_uid[l]);
                  }
                  if (is_ghost[2][1][1]) {
                    get_back_node_owner[owner_cells[2][1][1]][1]++;
                    get_back_node_owner[owner_cells[2][1][1]].add(ua_node_uid[l]);
                  }
                  if (is_ghost[2][2][1]) {
                    get_back_node_owner[owner_cells[2][2][1]][1]++;
                    get_back_node_owner[owner_cells[2][2][1]].add(ua_node_uid[l]);
                  }
                  new_owner = parent_cell.owner();
                }


                else if(i == child_coord_x && is_ghost[1][1][0] && (!mask_node_if_cell_left[l])){
                  new_owner = owner_cells[1][1][0];
                }

                else if(j == child_coord_y && is_ghost[1][0][1] && (!mask_node_if_cell_bottom[l])){
                  new_owner = owner_cells[1][0][1];
                }

                else if(k == child_coord_z && is_ghost[0][1][1] && (!mask_node_if_cell_rear[l])){
                  new_owner = owner_cells[0][1][1];
                }

                else if(i == (child_coord_x + pattern-1) && is_ghost[1][1][2] && (!mask_node_if_cell_right[l])){
                  get_back_node_owner[owner_cells[1][1][2]][1]++;
                  get_back_node_owner[owner_cells[1][1][2]].add(ua_node_uid[l]);
                  new_owner = parent_cell.owner();
                }

                else if(j == (child_coord_y + pattern-1) && is_ghost[1][2][1] && (!mask_node_if_cell_top[l])){
                  get_back_node_owner[owner_cells[1][2][1]][1]++;
                  get_back_node_owner[owner_cells[1][2][1]].add(ua_node_uid[l]);
                  new_owner = parent_cell.owner();
                }

                else if(k == (child_coord_z + pattern-1) && is_ghost[2][1][1] && (!mask_node_if_cell_front[l])){
                  get_back_node_owner[owner_cells[2][1][1]][1]++;
                  get_back_node_owner[owner_cells[2][1][1]].add(ua_node_uid[l]);
                  new_owner = parent_cell.owner();
                }


                else{
                  new_owner = parent_cell.owner();
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






    UniqueArray<Int64> recv_buffer;
    UniqueArray<Int64> empty_buffer;

    for(Integer rank = 0; rank < nb_rank; ++rank){
      m_mesh->parallelMng()->gatherVariable(get_back_node_owner[rank], recv_buffer, rank);
    }

    Integer i = 0;
    while(i < recv_buffer.size()){

      Integer rank = recv_buffer[i++];
      Integer size = recv_buffer[i++];
      ArrayView<Int64> av = recv_buffer.subView(i, size);
      i += size;
      UniqueArray<Int32> local_ids(size);
      m_mesh->nodeFamily()->itemsUniqueIdToLocalId(local_ids, av, true);

      ENUMERATE_ (Node, inode, m_mesh->nodeFamily()->view(local_ids)) {
        Node node = *inode;
        debug() << "Change node owner -- UniqueId : " << node.uniqueId()
                << " -- Old Owner : " << node.owner()
                << " -- New Owner : " << rank
        ;
        node.mutableItemBase().setOwner(rank, pm->commRank());
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











    UniqueArray<Int64> recv_buffer;
    UniqueArray<Int64> empty_buffer;

    for(Integer rank = 0; rank < nb_rank; ++rank){
      m_mesh->parallelMng()->gatherVariable(get_back_face_owner[rank], recv_buffer, rank);
    }

    Integer i = 0;
    while(i < recv_buffer.size()){

      Integer rank = recv_buffer[i++];
      Integer size = recv_buffer[i++];
      ArrayView<Int64> av = recv_buffer.subView(i, size);
      i += size;
      UniqueArray<Int32> local_ids(size);
      m_mesh->faceFamily()->itemsUniqueIdToLocalId(local_ids, av, true);

      ENUMERATE_ (Face, iface, m_mesh->faceFamily()->view(local_ids)) {
        Face face = *iface;
        debug() << "Change face owner -- UniqueId : " << face.uniqueId()
                << " -- Old Owner : " << face.owner()
                << " -- New Owner : " << rank
        ;
        face.mutableItemBase().setOwner(rank, pm->commRank());
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
        for (Node node : child.nodes()) {
          debug() << "\tChild Node : " << node.uniqueId() << " -- Coord : " << nodes_coords[node];
        }
      }
    }
  }




  ENUMERATE_(Cell, icell, m_mesh->allCells()){
    debug() << "\t" << *icell;
    for(Node node : icell->nodes()){
      debug() << "\t\t" << node;
    }
    for(Face face : icell->faces()){
      debug() << "\t\t\t" << face;
    }
  }

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
