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

#include "arcane/core/IGhostLayerMng.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/IMeshModifier.h"

#include "arcane/cartesianmesh/CellDirectionMng.h"
#include "arcane/cartesianmesh/CartesianMeshNumberingMng.h"

#include "arcane/utils/Array2View.h"
#include "arcane/utils/Array3View.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


CartesianMeshAMRPatchMng::
CartesianMeshAMRPatchMng(ICartesianMesh* cmesh)
: TraceAccessor(cmesh->mesh()->traceMng())
, m_mesh(cmesh->mesh())
, m_num_mng(Arccore::makeRef(new CartesianMeshNumberingMng(cmesh->mesh())))
{

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAMRPatchMng::
flagCellToCoarse(Int32ConstArrayView cells_lids)
{
  ItemInfoListView cells(m_mesh->cellFamily());
  for (int lid : cells_lids) {
    Item item = cells[lid];
    item.mutableItemBase().addFlags(ItemFlags::II_Coarsen);
  }
  _syncFlagCell();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAMRPatchMng::
_syncFlagCell()
{
  if (!m_mesh->parallelMng()->isParallel())
    return ;

  VariableCellInteger flag_cells_consistent(VariableBuildInfo(m_mesh, "FlagCellsConsistent"));
  ENUMERATE_(Cell, icell, m_mesh->ownCells()){
    Cell cell = *icell;
    flag_cells_consistent[cell] = cell.mutableItemBase().flags();
    debug() << "Send " << cell
            << " -- flag : " << cell.mutableItemBase().flags()
            << " -- II_Refine : " << (cell.itemBase().flags() & ItemFlags::II_Refine)
            << " -- II_Inactive : " << (cell.itemBase().flags() & ItemFlags::II_Inactive)
            << " -- II_Coarsen : " << (cell.itemBase().flags() & ItemFlags::II_Coarsen)
    ;
  }

  flag_cells_consistent.synchronize();

  ENUMERATE_(Cell, icell, m_mesh->allCells().ghost()){
    Cell cell = *icell;

    // On ajoute uniquement les flags qui nous interesse (pour éviter d'ajouter le flag "II_Own" par exemple).
    // On utilise set au lieu de add puisqu'une maille ne peut être à la fois II_Refine et II_Inactive.
    if(flag_cells_consistent[cell] & ItemFlags::II_Refine) {
      cell.mutableItemBase().setFlags(ItemFlags::II_Refine);
    }
    if(flag_cells_consistent[cell] & ItemFlags::II_Inactive) {
      cell.mutableItemBase().setFlags(ItemFlags::II_Inactive);
    }
    if(flag_cells_consistent[cell] & ItemFlags::II_Coarsen) {
      cell.mutableItemBase().setFlags(ItemFlags::II_Coarsen);
    }
    debug() << "After Compute " << cell
            << " -- flag : " << cell.mutableItemBase().flags()
            << " -- II_Refine : " << (cell.itemBase().flags() & ItemFlags::II_Refine)
            << " -- II_Inactive : " << (cell.itemBase().flags() & ItemFlags::II_Inactive)
            << " -- II_Coarsen : " << (cell.itemBase().flags() & ItemFlags::II_Coarsen)
    ;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
  Int32 max_level = 0;

  UniqueArray<Cell> cell_to_refine_internals;
  ENUMERATE_ (Cell, icell, m_mesh->allActiveCells()) {
    Cell cell = *icell;
    if (cell.itemBase().flags() & ItemFlags::II_Refine) {
      cell_to_refine_internals.add(cell);
      if(cell.level() > max_level) max_level = cell.level();
    }
  }
  m_num_mng->prepareLevel(max_level+1);

  UniqueArray<Int64> cells_infos;
  UniqueArray<Int64> faces_infos;
  UniqueArray<Int64> nodes_infos;

  Integer total_nb_cells = 0;
  Integer total_nb_nodes = 0;
  Integer total_nb_faces = 0;

  std::unordered_map<Int64, Int32> node_uid_to_owner;
  std::unordered_map<Int64, Int32> face_uid_to_owner;

  UniqueArray<Int64> node_uid_change_owner_only;
  UniqueArray<Int64> face_uid_change_owner_only;

  // Maps permettant de stocker les uids des noeuds et des faces
  // dont on récupère la propriété. Un tableau par processus.
  std::unordered_map<Int32, UniqueArray<Int64>> get_back_face_owner;
  std::unordered_map<Int32, UniqueArray<Int64>> get_back_node_owner;

  // Le premier élément de chaque tableau désigne le nouveau propriétaire des
  // noeuds et des faces et le second le nombre d'uid de noeud et de faces de chaque tableau.
  for(Integer rank = 0; rank < nb_rank; ++rank){
    get_back_face_owner[rank].add(my_rank);
    get_back_face_owner[rank].add(0);

    get_back_node_owner[rank].add(my_rank);
    get_back_node_owner[rank].add(0);
  }

  // Deux tableaux permettant de récupérer les uniqueIds des noeuds et des faces
  // de chaque maille enfant à chaque appel à getNodeUids()/getFaceUids().
  UniqueArray<Int64> child_nodes_uids(m_num_mng->getNbNode());
  UniqueArray<Int64> child_faces_uids(m_num_mng->getNbFace());

  // On doit enregistrer les mailles parentes de chaque maille enfant pour mettre à jour les connectivités
  // lors de la création des mailles.
  UniqueArray<Cell> parent_cells;

  // Maps remplaçant les mailles fantômes.
  std::unordered_map<Int64, Integer> around_parent_cells_uid_to_owner;
  std::unordered_map<Int64, Int32> around_parent_cells_uid_to_flags;

  // Partie échange d'informations sur les mailles autour du patch
  // (pour remplacer les mailles fantômes).
  {
    // On a uniquement besoin de ses deux flags pour les mailles autour.
    // (II_Refine pour savoir si les mailles autour sont dans le même patch)
    // (II_Inactive pour savoir si les mailles autour sont déjà raffinées)
    Int32 usefull_flags = ItemFlags::II_Refine + ItemFlags::II_Inactive;

    // On remplit le tableau avec nos infos, pour les autres processus.
    ENUMERATE_ (Cell, icell, m_mesh->ownCells()) {
      Cell cell = *icell;
      around_parent_cells_uid_to_owner[cell.uniqueId()] = my_rank;
      around_parent_cells_uid_to_flags[cell.uniqueId()] = ((cell.itemBase().flags() & usefull_flags) + ItemFlags::II_UserMark1);
    }

    ENUMERATE_ (Cell, icell, m_mesh->allCells().ghost()){
      Cell cell = *icell;
      around_parent_cells_uid_to_owner[cell.uniqueId()] = cell.owner();
      around_parent_cells_uid_to_flags[cell.uniqueId()] = ((cell.itemBase().flags() & usefull_flags) + ItemFlags::II_UserMark1);
    }

    // Tableau qui contiendra les uids des mailles dont on a besoin des infos.
    UniqueArray<Int64> uid_of_cells_needed;
    {
      UniqueArray<Int64> cell_uids_around((m_mesh->dimension() == 2) ? 9 : 27);
      for (Cell parent_cell : cell_to_refine_internals) {
        m_num_mng->getCellUidsAround(cell_uids_around, parent_cell);
        for (Int64 cell_uid : cell_uids_around) {
          // Si -1 alors il n'y a pas de mailles à cette position.
          if (cell_uid == -1)
            continue;

          // TODO C++20 : Mettre map.contains().
          // SI on a la maille, on n'a pas besoin de demander d'infos.
          if (around_parent_cells_uid_to_owner.find(cell_uid) != around_parent_cells_uid_to_owner.end())
            continue;

          uid_of_cells_needed.add(cell_uid);
        }
      }
    }

    UniqueArray<Int64> uid_of_cells_needed_all_procs;
    pm->allGatherVariable(uid_of_cells_needed, uid_of_cells_needed_all_procs);

    UniqueArray<Int32> flags_of_cells_needed_all_procs(uid_of_cells_needed_all_procs.size());
    UniqueArray<Int32> owner_of_cells_needed_all_procs(uid_of_cells_needed_all_procs.size());

    {
      UniqueArray<Int32> local_ids(uid_of_cells_needed_all_procs.size());
      m_mesh->cellFamily()->itemsUniqueIdToLocalId(local_ids, uid_of_cells_needed_all_procs, false);
      Integer compt = 0;
      ENUMERATE_ (Cell, icell, m_mesh->cellFamily()->view(local_ids)) {
        // Le isOwn est important vu qu'il peut y avoir les mailles fantômes.
        if (!icell->null() && icell->isOwn()) {
          owner_of_cells_needed_all_procs[compt] = my_rank;
          flags_of_cells_needed_all_procs[compt] = (icell->itemBase().flags() & usefull_flags);
        }
        else {
          owner_of_cells_needed_all_procs[compt] = -1;
          flags_of_cells_needed_all_procs[compt] = 0;
        }
        compt++;
      }
    }

    pm->reduce(Parallel::eReduceType::ReduceMax, owner_of_cells_needed_all_procs);
    pm->reduce(Parallel::eReduceType::ReduceMax, flags_of_cells_needed_all_procs);

    // A partir de ce moment, si les parent_cells sont au niveau 0, le tableau
    // "owner_of_cells_needed_all_procs" ne devrait plus contenir de "-1".
    // Si les parent_cells sont au niveau 1 ou plus, il peut y avoir des "-1"
    // car les mailles autour ne sont pas forcément toutes raffinées.
    // (exemple : on est en train de faire le niveau 2, donc on regarde les mailles
    // parent de niveau 1 tout autour. Il se peut que la maille d'à coté n'ai jamais
    // été raffinée, donc n'a pas de mailles de niveau 1. Comme la maille n'existe pas,
    // aucun processus ne peut mettre un propriétaire, donc le tableau des propriétaires
    // contiendra "-1".

    // On récupère les infos des mailles autour qui nous intéressent.
    {
      Integer size_uid_of_cells_needed = uid_of_cells_needed.size();
      Integer my_pos_in_all_procs_arrays = 0;
      UniqueArray<Integer> size_uid_of_cells_needed_per_proc(nb_rank);
      ArrayView<Integer> av(1, &size_uid_of_cells_needed);
      pm->allGather(av, size_uid_of_cells_needed_per_proc);

      for (Integer i = 0; i < my_rank; ++i) {
        my_pos_in_all_procs_arrays += size_uid_of_cells_needed_per_proc[i];
      }

      ArrayView<Int32> owner_of_cells_needed = owner_of_cells_needed_all_procs.subView(my_pos_in_all_procs_arrays, size_uid_of_cells_needed);
      ArrayView<Int32> flags_of_cells_needed = flags_of_cells_needed_all_procs.subView(my_pos_in_all_procs_arrays, size_uid_of_cells_needed);
      for (Integer i = 0; i < size_uid_of_cells_needed; ++i) {
        around_parent_cells_uid_to_owner[uid_of_cells_needed[i]] = owner_of_cells_needed[i];
        around_parent_cells_uid_to_flags[uid_of_cells_needed[i]] = flags_of_cells_needed[i];
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
    cells_infos.reserve((cell_to_refine_internals.size() * 4) * (2 + m_num_mng->getNbNode()));

    // Pour la taille, au maximum :
    // - on a "cell_to_refine_internals.size() * 12" faces
    // - pour chaque face, on a 2 infos (type de face et uniqueId de la face)
    // - pour chaque face, on a 2 uniqueIds de noeuds.
    faces_infos.reserve((cell_to_refine_internals.size() * 12) * (2 + 2));

    // Pour la taille, au maximum :
    // - on a (cell_to_refine_internals.size() * 9) uniqueIds de noeuds.
    nodes_infos.reserve(cell_to_refine_internals.size() * 9);

    for (Cell parent_cell : cell_to_refine_internals) {
      const Int64 parent_cell_uid = parent_cell.uniqueId();
      const Int32 parent_cell_level = parent_cell.level();
      const bool parent_cell_is_own = (parent_cell.owner() == my_rank);

      const Int64 parent_coord_x = m_num_mng->uidToCoordX(parent_cell_uid, parent_cell_level);
      const Int64 parent_coord_y = m_num_mng->uidToCoordY(parent_cell_uid, parent_cell_level);

      const Int64 child_coord_x = m_num_mng->getOffsetLevelToLevel(parent_coord_x, parent_cell_level, parent_cell_level + 1);
      const Int64 child_coord_y = m_num_mng->getOffsetLevelToLevel(parent_coord_y, parent_cell_level, parent_cell_level + 1);

      const Integer pattern = m_num_mng->getPattern();


      UniqueArray<Int64> uid_cells_around_parent_cell_1d(9);
      UniqueArray<Int32> owner_cells_around_parent_cell_1d(9);
      UniqueArray<Int32> flags_cells_around_parent_cell_1d(9);

      m_num_mng->getCellUidsAround(uid_cells_around_parent_cell_1d, parent_cell);

      for(Integer i = 0; i < 9; ++i){
        Int64 uid_cell = uid_cells_around_parent_cell_1d[i];
        // Si uid_cell != -1 alors il y a peut-être une maille (mais on ne sait pas si elle est bien présente).
        // Si around_parent_cells_uid_to_owner[uid_cell] != -1 alors il y a bien une maille.
        if(uid_cell != -1 && around_parent_cells_uid_to_owner[uid_cell] != -1) {
          owner_cells_around_parent_cell_1d[i] = around_parent_cells_uid_to_owner[uid_cell];
          flags_cells_around_parent_cell_1d[i] = around_parent_cells_uid_to_flags[uid_cell];
        }
        else{
          uid_cells_around_parent_cell_1d[i] = -1;
          owner_cells_around_parent_cell_1d[i] = -1;
          flags_cells_around_parent_cell_1d[i] = 0;
        }
      }

      // Pour simplifier, on utilise des vues 2D. (array[Y][X]).
      ConstArray2View uid_cells_around_parent_cell(uid_cells_around_parent_cell_1d.data(), 3, 3);
      ConstArray2View owner_cells_around_parent_cell(owner_cells_around_parent_cell_1d.data(), 3, 3);
      ConstArray2View flags_cells_around_parent_cell(flags_cells_around_parent_cell_1d.data(), 3, 3);


      // Voici les priorités pour la propriété des noeuds et des faces :
      // ┌─────────┐
      // │6   7   8│
      // └───────┐ │
      // ┌─┐ ┌─┐ │ │
      // │3│ │4│ │5│
      // │ │ └─┘ └─┘
      // │ └───────┐
      // │0   1   2│
      // └─────────┘
      // Chaque chiffre désigne une maille parente et une priorité (0 étant la priorité la plus forte).
      // 4 = parent_cell ("nous")

      // Exemple 1 :
      // On cherche à raffiner des mailles de niveau 0 (donc créer des mailles de niveau 1).
      // En bas, il n'y a pas de mailles.
      // À gauche (donc priorité 3), il y a une maille qui est déjà raffinée (flag "II_Inactive").
      // On est priorité 4 donc il est prioritaire. Donc les noeuds et des faces que l'on a en commun
      // lui appartiennent.

      // Exemple 2 :
      // On cherche à raffiner des mailles de niveau 0 (donc créer des mailles de niveau 1).
      // En haut, il y a des mailles déjà raffinées (flag "II_Inactive").
      // On est prioritaire sur elles, on récupère donc la propriété des noeuds et des faces que l'on a
      // en commun. Ce changement de propriété doit leur être signalé.

      // On simplifie avec un tableau de booléens.
      // Si true, alors on doit appliquer la priorité de propriété.
      // Si false, alors on considère qu'il n'y a pas de maille à la position définie.
      bool is_cell_around_parent_cell_present_and_useful[3][3] = {{false}};

      // Pour les mailles prioritaires sur nous, on doit regarder les deux flags.
      // Si une maille a le flag "II_Refine", on n'existe pas pour elle donc elle prend la propriété
      // des faces et des noeuds qu'on a en commun.
      // Si une maille a le flag "II_Inactive", elle a déjà les bons propriétaires.
      // Quoi qu'il en soit, si true alors les faces et noeuds qu'on a en commun leurs appartiennent.
      is_cell_around_parent_cell_present_and_useful[0][0] = ((uid_cells_around_parent_cell(0, 0) != -1) && (flags_cells_around_parent_cell(0, 0) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_cell_around_parent_cell_present_and_useful[0][1] = ((uid_cells_around_parent_cell(0, 1) != -1) && (flags_cells_around_parent_cell(0, 1) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_cell_around_parent_cell_present_and_useful[0][2] = ((uid_cells_around_parent_cell(0, 2) != -1) && (flags_cells_around_parent_cell(0, 2) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));

      is_cell_around_parent_cell_present_and_useful[1][0] = ((uid_cells_around_parent_cell(1, 0) != -1) && (flags_cells_around_parent_cell(1, 0) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      // is_cell_around_parent_cell_present_and_useful[1][1] = parent_cell;

      // Pour les mailles non prioritaires, on doit regarder qu'un seul flag.
      // Si une maille a le flag "II_Inactive", alors elle doit être avertie qu'on récupère la propriété
      // des noeuds et des faces qu'on a en commun.
      // On ne regarde pas le flag "II_Refine" car, si ces mailles sont aussi en train d'être raffinée,
      // elles savent qu'on existe et qu'on obtient la propriété des noeuds et des faces qu'on a en commun.
      // En résumé, si true alors les faces et noeuds qu'on a en commun nous appartiennent.
      is_cell_around_parent_cell_present_and_useful[1][2] = ((uid_cells_around_parent_cell(1, 2) != -1) && (flags_cells_around_parent_cell(1, 2) & ItemFlags::II_Inactive));

      is_cell_around_parent_cell_present_and_useful[2][0] = ((uid_cells_around_parent_cell(2, 0) != -1) && (flags_cells_around_parent_cell(2, 0) & ItemFlags::II_Inactive));
      is_cell_around_parent_cell_present_and_useful[2][1] = ((uid_cells_around_parent_cell(2, 1) != -1) && (flags_cells_around_parent_cell(2, 1) & ItemFlags::II_Inactive));
      is_cell_around_parent_cell_present_and_useful[2][2] = ((uid_cells_around_parent_cell(2, 2) != -1) && (flags_cells_around_parent_cell(2, 2) & ItemFlags::II_Inactive));


      // En plus de regarder si chaque maille parent autour de notre maille parent existe et possède (II_Inactive) ou possédera (II_Refine) des enfants...
      // ... on regarde si chaque maille parent est présente sur notre sous-domaine, que ce soit une maille fantôme ou non.
      auto is_cell_around_parent_cell_in_subdomain = [&](Integer y, Integer x){
        return is_cell_around_parent_cell_present_and_useful[y][x] && (flags_cells_around_parent_cell(y, x) & ItemFlags::II_UserMark1);
      };

      // ... on regarde si chaque maille parent est possédé par le même propriétaire que notre maille parent.
      auto is_cell_around_parent_cell_same_owner = [&](Integer y, Integer x){
        return is_cell_around_parent_cell_present_and_useful[y][x] && (owner_cells_around_parent_cell(y, x) == owner_cells_around_parent_cell(1, 1));
      };

      // ... on regarde si chaque maille parent a un propriétaire différent par rapport à notre maille parent.
      auto is_cell_around_parent_cell_different_owner = [&](Integer y, Integer x){
        return is_cell_around_parent_cell_present_and_useful[y][x] && (owner_cells_around_parent_cell(y, x) != owner_cells_around_parent_cell(1, 1));
      };


      // On itère sur toutes les mailles enfants.
      for (Int64 j = child_coord_y; j < child_coord_y + pattern; ++j) {
        for (Int64 i = child_coord_x; i < child_coord_x + pattern; ++i) {
          parent_cells.add(parent_cell);
          total_nb_cells++;

          const Int64 child_cell_uid = m_num_mng->getCellUid(parent_cell_level + 1, i, j);
          debug() << "Child -- x : " << i << " -- y : " << j << " -- level : " << parent_cell_level + 1 << " -- uid : " << child_cell_uid;

          m_num_mng->getNodeUids(child_nodes_uids, parent_cell_level + 1, i, j);
          m_num_mng->getFaceUids(child_faces_uids, parent_cell_level + 1, i, j);

          const Integer type_cell = IT_Quad4;
          const Integer type_face = IT_Line2;

          // Partie Cell.
          cells_infos.add(type_cell);
          cells_infos.add(child_cell_uid);
          for (Integer nc = 0; nc < m_num_mng->getNbNode(); nc++) {
            cells_infos.add(child_nodes_uids[nc]);
          }

          // Partie Face.
          for(Integer l = 0; l < m_num_mng->getNbFace(); ++l){
            Integer child_face_owner = -1;
            bool is_new_face = false;

            // Deux parties :
            // D'abord, on regarde si l'on doit créer la face l. Pour cela, on doit regarder si elle est présente sur la
            // maille à côté.
            // Pour gauche/bas, c'est le même principe. Si la maille enfant est tout à gauche/bas de la maille parente, on regarde
            // s'il y a une maille parente à gauche/bas. Sinon, on crée la face. Si oui, on regarde le masque pour savoir si l'on
            // doit créer la face.
            // Pour droite/haut, le principe est différent de gauche/bas. On ne suit le masque que si on est tout à droite/haut
            // de la maille parente. Sinon on crée toujours les faces droites/hautes.
            // Enfin, on utilise le tableau "is_cell_around_parent_cell_in_subdomain". Si la maille parente d'à côté est sur
            // notre sous-domaine, alors il se peut que les faces en communes avec notre maille parente existent déjà, dans ce cas,
            // pas de doublon.
            if (
              ( (i == child_coord_x && !is_cell_around_parent_cell_in_subdomain(1, 0)) || (mask_face_if_cell_left[l]) )
              &&
              ( (i != (child_coord_x + pattern-1) || !is_cell_around_parent_cell_in_subdomain(1, 2)) || mask_face_if_cell_right[l] )
              &&
              ( (j == child_coord_y && !is_cell_around_parent_cell_in_subdomain(0, 1)) || (mask_face_if_cell_bottom[l]) )
              &&
              ( (j != (child_coord_y + pattern-1) || !is_cell_around_parent_cell_in_subdomain(2, 1)) || mask_face_if_cell_top[l] )
            )
            {
              is_new_face = true;
              faces_infos.add(type_face);
              faces_infos.add(child_faces_uids[l]);

              // Les noeuds de la face sont toujours les noeuds l et l+1
              // car on utilise la même exploration pour les deux cas.
              for (Integer nc = l; nc < l + 2; nc++) {
                faces_infos.add(child_nodes_uids[nc % m_num_mng->getNbNode()]);
              }
              total_nb_faces++;

              // Par défaut, parent_cell est propriétaire de la nouvelle face.
              child_face_owner = owner_cells_around_parent_cell(1, 1);
            }

            // Deuxième partie.
            // On doit maintenant trouver le bon propriétaire pour la face. Mis à part le tableau "is_cell_around_parent_cell_same_owner",
            // la condition est identique à celle au-dessus.
            // Le changement de tableau est important puisqu'à partir d'ici, on est sûr qu'il y a la face qui nous intéresse.
            // Le nouveau tableau permet de savoir si la maille d'à côté est aussi à nous ou pas. Si ce n'est pas le cas, alors
            // un changement de propriétaire est possible, selon les priorités définies au-dessus. On n'a pas besoin de savoir
            // si la maille est présente sur le sous-domaine.
            if (
              ( (i == child_coord_x && !is_cell_around_parent_cell_same_owner(1, 0)) || (mask_face_if_cell_left[l]) )
              &&
              ( (i != (child_coord_x + pattern-1) || !is_cell_around_parent_cell_same_owner(1, 2)) || mask_face_if_cell_right[l] )
              &&
              ( (j == child_coord_y && !is_cell_around_parent_cell_same_owner(0, 1)) || (mask_face_if_cell_bottom[l]) )
              &&
              ( (j != (child_coord_y + pattern-1) || !is_cell_around_parent_cell_same_owner(2, 1)) || mask_face_if_cell_top[l] )
              )
            {
              // Ici, la construction des conditions est la même à chaque fois.
              // Le premier booléen (i == child_coord_x) regarde si l'enfant se trouve
              // du bon côté de la maille parent.
              // Le second booléen (!mask_face_if_cell_left[l]) nous dit si la face l est bien
              // la face en commun avec la maille parent d'à côté.
              // Le troisième booléen (is_cell_around_parent_cell_different_owner(1, 0)) regarde s'il y a une
              // maille à côté qui prend la propriété de la face ou à qui on prend la propriété.

              // En outre, il y a deux cas différents selon les priorités définies au-dessus :
              // - soit nous ne sommes pas prioritaire, alors on attribue le propriétaire prioritaire à notre face,
              // - soit nous sommes prioritaire, alors on se positionne comme propriétaire de la face et on doit prévenir
              //   tous les autres processus (le processus ancien propriétaire mais aussi les processus qui peuvent
              //   avoir la face en fantôme).

              // Enfin, dans le cas du changement de propriétaire, seul le processus (re)prenant la propriété doit
              // faire une communication à ce propos. Les processus ne possédant que la face en fantôme ne doit pas
              // faire de communication (mais ils peuvent définir localement le bon propriétaire, TODO Optimisation possible ?).

              // À gauche, priorité 3 < 4 donc il prend la propriété de la face.
              if (i == child_coord_x && (!mask_face_if_cell_left[l]) && is_cell_around_parent_cell_different_owner(1, 0)) {
                child_face_owner = owner_cells_around_parent_cell(1, 0);
              }

              // En bas, priorité 1 < 4 donc il prend la propriété de la face.
              else if (j == child_coord_y && (!mask_face_if_cell_bottom[l]) && is_cell_around_parent_cell_different_owner(0, 1)) {
                child_face_owner = owner_cells_around_parent_cell(0, 1);
              }

              // Sinon, parent_cell est propriétaire de la face.
              else {

                // Seul le propriétaire de la parent_cell peut communiquer sur le changement de propriété.
                if (parent_cell_is_own) {

                  // À droite, priorité 5 > 4 donc parent_cell récupère la propriété de la face.
                  if (i == (child_coord_x + pattern - 1) && (!mask_face_if_cell_right[l]) && is_cell_around_parent_cell_different_owner(1, 2)) {
                    get_back_face_owner[owner_cells_around_parent_cell(1, 2)][1]++;
                    get_back_face_owner[owner_cells_around_parent_cell(1, 2)].add(child_faces_uids[l]);
                  }

                  // En haut, priorité 7 > 4 donc on récupère la propriété de la face.
                  else if (j == (child_coord_y + pattern - 1) && (!mask_face_if_cell_top[l]) && is_cell_around_parent_cell_different_owner(2, 1)) {
                    get_back_face_owner[owner_cells_around_parent_cell(2, 1)][1]++;
                    get_back_face_owner[owner_cells_around_parent_cell(2, 1)].add(child_faces_uids[l]);
                  }
                }

                // Sinon, c'est une face interne donc au parent_cell.
                child_face_owner = owner_cells_around_parent_cell(1, 1);
              }
            }

            // S'il y a une création de face et/ou un changement de propriétaire.
            if(child_face_owner != -1){
              face_uid_to_owner[child_faces_uids[l]] = child_face_owner;

              // Lorsqu'il y a un changement de propriétaire sans création de face,
              // on doit mettre de côté les uniqueIds de ces faces pour pouvoir
              // itérer dessus par la suite.
              if(!is_new_face){
                face_uid_change_owner_only.add(child_faces_uids[l]);
                debug() << "Child face (change owner) -- x : " << i
                        << " -- y : " << j
                        << " -- level : " << parent_cell_level + 1
                        << " -- face : " << l
                        << " -- uid_face : " << child_faces_uids[l]
                        << " -- owner : " << child_face_owner
                ;
              }
              else{
                debug() << "Child face (create face)  -- x : " << i
                        << " -- y : " << j
                        << " -- level : " << parent_cell_level + 1
                        << " -- face : " << l
                        << " -- uid_face : " << child_faces_uids[l]
                        << " -- owner : " << child_face_owner
                ;
              }
            }
          }

          // Partie Node.
          // Cette partie est assez ressemblante à la partie face, mis à part le fait qu'il peut y avoir
          // plus de propriétaires possibles.
          for(Integer l = 0; l < m_num_mng->getNbNode(); ++l) {
            Integer child_node_owner = -1;
            bool is_new_node = false;

            // Deux parties :
            // D'abord, on regarde si l'on doit créer le noeud l. Pour cela, on doit regarder s'il est présente sur la
            // maille à côté.
            // Pour gauche/bas, c'est le même principe. Si la maille enfant est tout à gauche/bas de la maille parente, on regarde
            // s'il y a une maille parente à gauche/bas. Sinon, on crée le noeud. Si oui, on regarde le masque pour savoir si l'on
            // doit créer le noeud.
            // Pour droite/haut, le principe est différent de gauche/bas. On ne suit le masque que si la maille enfant est toute à droite/haut
            // de la maille parente. Sinon on crée toujours les noeuds droites/hautes.
            // Enfin, on utilise le tableau "is_cell_around_parent_cell_in_subdomain". Si la maille parente d'à côté est sur
            // notre sous-domaine, alors il se peut que les noeuds en communs avec notre maille parente existent déjà, dans ce cas,
            // pas de doublon.
            if (
              ( (i == child_coord_x && !is_cell_around_parent_cell_in_subdomain(1, 0)) || (mask_node_if_cell_left[l]) )
              &&
              ( (i != (child_coord_x + pattern-1) || !is_cell_around_parent_cell_in_subdomain(1, 2)) || mask_node_if_cell_right[l] )
              &&
              ( (j == child_coord_y && !is_cell_around_parent_cell_in_subdomain(0, 1)) || (mask_node_if_cell_bottom[l]) )
              &&
              ( (j != (child_coord_y + pattern-1) || !is_cell_around_parent_cell_in_subdomain(2, 1)) || mask_node_if_cell_top[l] )
            )
            {
              is_new_node = true;
              nodes_infos.add(child_nodes_uids[l]);
              total_nb_nodes++;

              // Par défaut, parent_cell est propriétaire du nouveau noeud.
              child_node_owner = owner_cells_around_parent_cell(1, 1);
            }

            // Deuxième partie.
            // On doit maintenant trouver le bon propriétaire pour le noeud. Mis à part le tableau "is_cell_around_parent_cell_same_owner",
            // la condition est identique à celle au-dessus.
            // Le changement de tableau est important puisqu'à partir d'ici, on est sûr que le noeud qui nous intéresse existe.
            // Le nouveau tableau permet de savoir si la maille d'à côté est aussi à nous ou pas. Si ce n'est pas le cas, alors
            // un changement de propriétaire est possible, selon les priorités définies au-dessus. On n'a pas besoin de savoir
            // si la maille est présente sur le sous-domaine.
            if (
            ( (i == child_coord_x && !is_cell_around_parent_cell_same_owner(1, 0)) || (mask_node_if_cell_left[l]) )
            &&
            ( (i != (child_coord_x + pattern-1) || !is_cell_around_parent_cell_same_owner(1, 2)) || mask_node_if_cell_right[l] )
            &&
            ( (j == child_coord_y && !is_cell_around_parent_cell_same_owner(0, 1)) || (mask_node_if_cell_bottom[l]) )
            &&
            ( (j != (child_coord_y + pattern-1) || !is_cell_around_parent_cell_same_owner(2, 1)) || mask_node_if_cell_top[l] )
            )
            {
              // Par rapport aux faces qui n'ont que deux propriétaires possibles, un noeud peut
              // en avoir jusqu'à quatre.
              // (Et oui, en 3D, c'est encore plus amusant !)

              // Si le noeud est sur le côté gauche de la maille parente ("sur la face gauche").
              if (i == child_coord_x && (!mask_node_if_cell_left[l])) {

                // Si le noeud est sur le bas de la maille parente ("sur la face basse").
                // Donc noeud en bas à gauche (même position que le noeud de la maille parente).
                if (j == child_coord_y && (!mask_node_if_cell_bottom[l])) {

                  // Priorité 0 < 4.
                  if (is_cell_around_parent_cell_different_owner(0, 0)) {
                    child_node_owner = owner_cells_around_parent_cell(0, 0);
                  }

                  // Priorité 1 < 4.
                  else if (is_cell_around_parent_cell_different_owner(0, 1)) {
                    child_node_owner = owner_cells_around_parent_cell(0, 1);
                  }

                  // Priorité 3 < 4.
                  else if (is_cell_around_parent_cell_different_owner(1, 0)) {
                    child_node_owner = owner_cells_around_parent_cell(1, 0);
                  }

                  else {
                    child_node_owner = owner_cells_around_parent_cell(1, 1);
                  }
                }

                // Si le noeud est en haut de la maille parente ("sur la face haute").
                // Donc noeud en haut à gauche (même position que le noeud de la maille parente).
                else if (j == (child_coord_y + pattern - 1) && (!mask_node_if_cell_top[l])) {

                  // Priorité 3 < 4.
                  if (is_cell_around_parent_cell_different_owner(1, 0)) {
                    child_node_owner = owner_cells_around_parent_cell(1, 0);
                  }

                  // Sinon, parent_cell est propriétaire du noeud.
                  else {

                    // Seul le propriétaire de la parent_cell peut communiquer sur le changement de propriété.
                    if (parent_cell_is_own) {

                      // Priorité 6 > 4.
                      if (is_cell_around_parent_cell_different_owner(2, 0)) {
                        get_back_node_owner[owner_cells_around_parent_cell(2, 0)][1]++;
                        get_back_node_owner[owner_cells_around_parent_cell(2, 0)].add(child_nodes_uids[l]);
                      }

                      // Priorité 7 > 4.
                      if (is_cell_around_parent_cell_different_owner(2, 1)) {
                        get_back_node_owner[owner_cells_around_parent_cell(2, 1)][1]++;
                        get_back_node_owner[owner_cells_around_parent_cell(2, 1)].add(child_nodes_uids[l]);
                      }
                    }

                    child_node_owner = owner_cells_around_parent_cell(1, 1);
                  }
                }

                // Si le noeud est quelque part sur la face parente gauche...
                else {
                  // S'il y a une maille à gauche, elle est propriétaire du noeud.
                  if (is_cell_around_parent_cell_different_owner(1, 0)) {
                    child_node_owner = owner_cells_around_parent_cell(1, 0);
                  }

                  // Sinon parent_cell est propriétaire du noeud.
                  else {
                    child_node_owner = owner_cells_around_parent_cell(1, 1);
                  }
                }
              }

              // Si le noeud est sur le côté droit de la maille parente ("sur la face droite").
              else if (i == (child_coord_x + pattern - 1) && (!mask_node_if_cell_right[l])) {

                // Si le noeud est sur le bas de la maille parente ("sur la face basse").
                // Donc noeud en bas à droite (même position que le noeud de la maille parente).
                if (j == child_coord_y && (!mask_node_if_cell_bottom[l])) {

                  // Priorité 1 < 4.
                  if (is_cell_around_parent_cell_different_owner(0, 1)) {
                    child_node_owner = owner_cells_around_parent_cell(0, 1);
                  }

                  // Priorité 2 < 4.
                  else if (is_cell_around_parent_cell_different_owner(0, 2)) {
                    child_node_owner = owner_cells_around_parent_cell(0, 2);
                  }

                  // Sinon, parent_cell est propriétaire du noeud.
                  else {

                    // Seul le propriétaire de la parent_cell peut communiquer sur le changement de propriété.
                    // Priorité 5 > 4.
                    if (parent_cell_is_own && is_cell_around_parent_cell_different_owner(1, 2)) {
                      get_back_node_owner[owner_cells_around_parent_cell(1, 2)][1]++;
                      get_back_node_owner[owner_cells_around_parent_cell(1, 2)].add(child_nodes_uids[l]);
                    }
                    child_node_owner = owner_cells_around_parent_cell(1, 1);
                  }
                }

                // Si le noeud est en haut de la maille parente ("sur la face haute").
                // Donc noeud en haut à droite (même position que le noeud de la maille parente).
                else if (j == (child_coord_y + pattern - 1) && (!mask_node_if_cell_top[l])) {

                  // Seul le propriétaire de la parent_cell peut communiquer sur le changement de propriété.
                  if (parent_cell_is_own) {

                    // Priorité 5 > 4.
                    if (is_cell_around_parent_cell_different_owner(1, 2)) {
                      get_back_node_owner[owner_cells_around_parent_cell(1, 2)][1]++;
                      get_back_node_owner[owner_cells_around_parent_cell(1, 2)].add(child_nodes_uids[l]);
                    }

                    // Priorité 7 > 4.
                    if (is_cell_around_parent_cell_different_owner(2, 1)) {
                      get_back_node_owner[owner_cells_around_parent_cell(2, 1)][1]++;
                      get_back_node_owner[owner_cells_around_parent_cell(2, 1)].add(child_nodes_uids[l]);
                    }

                    // Priorité 8 > 4.
                    if (is_cell_around_parent_cell_different_owner(2, 2)) {
                      get_back_node_owner[owner_cells_around_parent_cell(2, 2)][1]++;
                      get_back_node_owner[owner_cells_around_parent_cell(2, 2)].add(child_nodes_uids[l]);
                    }
                  }

                  child_node_owner = owner_cells_around_parent_cell(1, 1);
                }

                // Si le noeud est quelque part sur la face parente droite...
                else {

                  // S'il y a une maille à droite, parent_cell est le propriétaire du noeud.
                  if (parent_cell_is_own && is_cell_around_parent_cell_different_owner(1, 2)) {
                    get_back_node_owner[owner_cells_around_parent_cell(1, 2)][1]++;
                    get_back_node_owner[owner_cells_around_parent_cell(1, 2)].add(child_nodes_uids[l]);
                  }

                  child_node_owner = owner_cells_around_parent_cell(1, 1);
                }
              }

              // Si le noeud est ni sur la face parente gauche, ni sur la face parente droite...
              else {

                // Si le noeud est sur le bas de la maille parente ("sur la face basse") et
                // qu'il y a une maille en bas de priorité 1 < 4, elle est propriétaire du noeud.
                if (j == child_coord_y && (!mask_node_if_cell_bottom[l]) && is_cell_around_parent_cell_different_owner(0, 1)) {
                  child_node_owner = owner_cells_around_parent_cell(0, 1);
                }

                // Si le noeud est sur le haut de la maille parente ("sur la face haute") et
                // qu'il y a une maille en haut de priorité 7 > 4, parent_cell est propriétaire du noeud.
                else if (parent_cell_is_own && j == (child_coord_y + pattern - 1) && (!mask_node_if_cell_top[l]) && is_cell_around_parent_cell_different_owner(2, 1)) {
                  get_back_node_owner[owner_cells_around_parent_cell(2, 1)][1]++;
                  get_back_node_owner[owner_cells_around_parent_cell(2, 1)].add(child_nodes_uids[l]);

                  child_node_owner = owner_cells_around_parent_cell(1, 1);
                }

                // Noeuds qui ne sont sur aucune face de la maille parente.
                else {
                  child_node_owner = owner_cells_around_parent_cell(1, 1);
                }
              }
            }

            // S'il y a une création de noeud et/ou un changement de propriétaire.
            if(child_node_owner != -1){
              node_uid_to_owner[child_nodes_uids[l]] = child_node_owner;

              // Lorsqu'il y a un changement de propriétaire sans création de noeud,
              // on doit mettre de côté les uniqueIds de ces noeuds pour pouvoir
              // itérer dessus par la suite.
              if(!is_new_node){
                node_uid_change_owner_only.add(child_nodes_uids[l]);
                debug() << "Child node (change owner) -- x : " << i
                        << " -- y : " << j
                        << " -- level : " << parent_cell_level + 1
                        << " -- node : " << l
                        << " -- uid_node : " << child_nodes_uids[l]
                        << " -- owner : " << child_node_owner;
              }
              else{
                debug() << "Child node (create node)  -- x : " << i
                        << " -- y : " << j
                        << " -- level : " << parent_cell_level + 1
                        << " -- node : " << l
                        << " -- uid_node : " << child_nodes_uids[l]
                        << " -- owner : " << child_node_owner;
              }
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
    // dans le tableau "child_nodes_uids" est toujours pareil (l et l+1, voir le 2D).
    // Pour le 3D, ce n'est pas le cas donc on a des tableaux pour avoir une correspondance
    // entre les noeuds de chaque face et la position des noeuds dans le tableau "child_nodes_uids".
    // (Exemple : pour la face 1 (même ordre d'énumération qu'Arcane), on doit prendre le
    // tableau "nodes_in_face_1" et donc les noeuds "child_nodes_uids[0]", "child_nodes_uids[3]",
    // "child_nodes_uids[7]" et "child_nodes_uids[4]").
    const Integer nodes_in_face_0[] = {0, 1, 2, 3};
    const Integer nodes_in_face_1[] = {0, 3, 7, 4};
    const Integer nodes_in_face_2[] = {0, 1, 5, 4};
    const Integer nodes_in_face_3[] = {4, 5, 6, 7};
    const Integer nodes_in_face_4[] = {1, 2, 6, 5};
    const Integer nodes_in_face_5[] = {3, 2, 6, 7};

    const Integer nb_nodes_in_face = 4;

    // Pour la taille :
    // - on a "cell_to_refine_internals.size() * 8" mailles enfants,
    // - pour chaque maille, on a 2 infos (type de maille et uniqueId de la maille)
    // - pour chaque maille, on a "m_num_mng->getNbNode()" uniqueIds (les uniqueId de chaque noeud de la maille).
    cells_infos.reserve((cell_to_refine_internals.size() * 8) * (2 + m_num_mng->getNbNode()));

    // Pour la taille, au maximum :
    // - on a "cell_to_refine_internals.size() * 36" faces enfants,
    // - pour chaque face, on a 2 infos (type de face et uniqueId de la face)
    // - pour chaque face, on a 4 uniqueIds de noeuds.
    faces_infos.reserve((cell_to_refine_internals.size() * 36) * (2 + 4));

    // Pour la taille, au maximum :
    // - on a (cell_to_refine_internals.size() * 27) uniqueIds de noeuds.
    nodes_infos.reserve(cell_to_refine_internals.size() * 27);

    for (Cell parent_cell : cell_to_refine_internals) {
      const Int64 parent_cell_uid = parent_cell.uniqueId();
      const Int32 parent_cell_level = parent_cell.level();
      const bool parent_cell_is_own = (parent_cell.owner() == my_rank);

      const Int64 parent_coord_x = m_num_mng->uidToCoordX(parent_cell_uid, parent_cell_level);
      const Int64 parent_coord_y = m_num_mng->uidToCoordY(parent_cell_uid, parent_cell_level);
      const Int64 parent_coord_z = m_num_mng->uidToCoordZ(parent_cell_uid, parent_cell_level);

      const Int64 child_coord_x = m_num_mng->getOffsetLevelToLevel(parent_coord_x, parent_cell_level, parent_cell_level + 1);
      const Int64 child_coord_y = m_num_mng->getOffsetLevelToLevel(parent_coord_y, parent_cell_level, parent_cell_level + 1);
      const Int64 child_coord_z = m_num_mng->getOffsetLevelToLevel(parent_coord_z, parent_cell_level, parent_cell_level + 1);

      const Integer pattern = m_num_mng->getPattern();


      UniqueArray<Int64> uid_cells_around_parent_cell_1d(27);
      UniqueArray<Int32> owner_cells_around_parent_cell_1d(27);
      UniqueArray<Int32> flags_cells_around_parent_cell_1d(27);

      m_num_mng->getCellUidsAround(uid_cells_around_parent_cell_1d, parent_cell);

      for(Integer i = 0; i < 27; ++i){
        Int64 uid_cell = uid_cells_around_parent_cell_1d[i];
        // Si uid_cell != -1 alors il y a peut-être une maille (mais on ne sait pas si elle est bien présente).
        // Si around_parent_cells_uid_to_owner[uid_cell] != -1 alors il y a bien une maille.
        if(uid_cell != -1 && around_parent_cells_uid_to_owner[uid_cell] != -1) {
          owner_cells_around_parent_cell_1d[i] = around_parent_cells_uid_to_owner[uid_cell];
          flags_cells_around_parent_cell_1d[i] = around_parent_cells_uid_to_flags[uid_cell];
        }
        else{
          uid_cells_around_parent_cell_1d[i] = -1;
          owner_cells_around_parent_cell_1d[i] = -1;
          flags_cells_around_parent_cell_1d[i] = 0;
        }
      }

      // Pour simplifier, on utilise des vues 3D. (array[Z][Y][X]).
      ConstArray3View uid_cells_around_parent_cell(uid_cells_around_parent_cell_1d.data(), 3, 3, 3);
      ConstArray3View owner_cells_around_parent_cell(owner_cells_around_parent_cell_1d.data(), 3, 3, 3);
      ConstArray3View flags_cells_around_parent_cell(flags_cells_around_parent_cell_1d.data(), 3, 3, 3);


      // Voici les priorités pour la propriété des noeuds et des faces :
      // ┌──────────┐ │ ┌──────────┐ │ ┌──────────┐
      // │ 6   7   8│ │ │15  16  17│ │ │24  25  26│
      // │          │ │ └───────┐  │ │ │          │
      // │          │ │ ┌──┐┌──┐│  │ │ │          │
      // │ 3   4   5│ │ │12││13││14│ │ │21  22  23│
      // │          │ │ │  │└──┘└──┘ │ │          │
      // │          │ │ │  └───────┐ │ │          │
      // │ 0   1   2│ │ │ 9  10  11│ │ │18  19  20│
      // └──────────┘ │ └──────────┘ │ └──────────┘
      // Z=0          │ Z=1          │ Z=2
      // ("arrière")  │              │ ("avant")

      // Chaque chiffre désigne une maille parente et une priorité (0 étant la priorité la plus forte).
      // 13 = parent_cell ("nous")

      // Exemple 1 :
      // On cherche à raffiner des mailles de niveau 0 (donc créer des mailles de niveau 1).
      // En bas, il n'y a pas de mailles.
      // À gauche (donc priorité 12), il y a une maille qui est déjà raffinée (flag "II_Inactive").
      // On est priorité 13 donc il est prioritaire. Donc les noeuds et des faces que l'on a en commun
      // lui appartiennent.

      // Exemple 2 :
      // On cherche à raffiner des mailles de niveau 0 (donc créer des mailles de niveau 1).
      // En haut, il y a des mailles déjà raffinées (flag "II_Inactive").
      // On est prioritaire sur elles, on récupère donc la propriété des noeuds et des faces que l'on a
      // en commun. Ce changement de propriété doit leur être signalé.

      // On simplifie avec un tableau de booléens.
      // Si true, alors on doit appliquer la priorité de propriété.
      // Si false, alors on considère qu'il n'y a pas de maille à la position définie.
      bool is_cell_around_parent_cell_present_and_useful[3][3][3] = {{{false}}};

      // Pour les mailles prioritaires sur nous, on doit regarder les deux flags.
      // Si une maille a le flag "II_Refine", on n'existe pas pour elle donc elle prend la propriété
      // des faces et des noeuds qu'on a en commun.
      // Si une maille a le flag "II_Inactive", elle a déjà les bons propriétaires.
      // Quoi qu'il en soit, si true alors les faces et noeuds qu'on a en commun leurs appartiennent.
      is_cell_around_parent_cell_present_and_useful[0][0][0] = ((uid_cells_around_parent_cell(0, 0, 0) != -1) && (flags_cells_around_parent_cell(0, 0, 0) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_cell_around_parent_cell_present_and_useful[0][0][1] = ((uid_cells_around_parent_cell(0, 0, 1) != -1) && (flags_cells_around_parent_cell(0, 0, 1) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_cell_around_parent_cell_present_and_useful[0][0][2] = ((uid_cells_around_parent_cell(0, 0, 2) != -1) && (flags_cells_around_parent_cell(0, 0, 2) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_cell_around_parent_cell_present_and_useful[0][1][0] = ((uid_cells_around_parent_cell(0, 1, 0) != -1) && (flags_cells_around_parent_cell(0, 1, 0) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_cell_around_parent_cell_present_and_useful[0][1][1] = ((uid_cells_around_parent_cell(0, 1, 1) != -1) && (flags_cells_around_parent_cell(0, 1, 1) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_cell_around_parent_cell_present_and_useful[0][1][2] = ((uid_cells_around_parent_cell(0, 1, 2) != -1) && (flags_cells_around_parent_cell(0, 1, 2) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_cell_around_parent_cell_present_and_useful[0][2][0] = ((uid_cells_around_parent_cell(0, 2, 0) != -1) && (flags_cells_around_parent_cell(0, 2, 0) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_cell_around_parent_cell_present_and_useful[0][2][1] = ((uid_cells_around_parent_cell(0, 2, 1) != -1) && (flags_cells_around_parent_cell(0, 2, 1) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_cell_around_parent_cell_present_and_useful[0][2][2] = ((uid_cells_around_parent_cell(0, 2, 2) != -1) && (flags_cells_around_parent_cell(0, 2, 2) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));

      is_cell_around_parent_cell_present_and_useful[1][0][0] = ((uid_cells_around_parent_cell(1, 0, 0) != -1) && (flags_cells_around_parent_cell(1, 0, 0) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_cell_around_parent_cell_present_and_useful[1][0][1] = ((uid_cells_around_parent_cell(1, 0, 1) != -1) && (flags_cells_around_parent_cell(1, 0, 1) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      is_cell_around_parent_cell_present_and_useful[1][0][2] = ((uid_cells_around_parent_cell(1, 0, 2) != -1) && (flags_cells_around_parent_cell(1, 0, 2) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));

      is_cell_around_parent_cell_present_and_useful[1][1][0] = ((uid_cells_around_parent_cell(1, 1, 0) != -1) && (flags_cells_around_parent_cell(1, 1, 0) & (ItemFlags::II_Refine | ItemFlags::II_Inactive)));
      // is_cell_around_parent_cell_present_and_useful[1][1][1] = parent_cell;

      // Pour les mailles non prioritaires, on doit regarder qu'un seul flag.
      // Si une maille a le flag "II_Inactive", alors elle doit être avertie qu'on récupère la propriété
      // des noeuds et des faces qu'on a en commun.
      // On ne regarde pas le flag "II_Refine" car, si ces mailles sont aussi en train d'être raffinée,
      // elles savent qu'on existe et qu'on obtient la propriété des noeuds et des faces qu'on a en commun.
      // En résumé, si true alors les faces et noeuds qu'on a en commun nous appartiennent.
      is_cell_around_parent_cell_present_and_useful[1][1][2] = ((uid_cells_around_parent_cell(1, 1, 2) != -1) && (flags_cells_around_parent_cell(1, 1, 2) & ItemFlags::II_Inactive));

      is_cell_around_parent_cell_present_and_useful[1][2][0] = ((uid_cells_around_parent_cell(1, 2, 0) != -1) && (flags_cells_around_parent_cell(1, 2, 0) & ItemFlags::II_Inactive));
      is_cell_around_parent_cell_present_and_useful[1][2][1] = ((uid_cells_around_parent_cell(1, 2, 1) != -1) && (flags_cells_around_parent_cell(1, 2, 1) & ItemFlags::II_Inactive));
      is_cell_around_parent_cell_present_and_useful[1][2][2] = ((uid_cells_around_parent_cell(1, 2, 2) != -1) && (flags_cells_around_parent_cell(1, 2, 2) & ItemFlags::II_Inactive));

      is_cell_around_parent_cell_present_and_useful[2][0][0] = ((uid_cells_around_parent_cell(2, 0, 0) != -1) && (flags_cells_around_parent_cell(2, 0, 0) & ItemFlags::II_Inactive));
      is_cell_around_parent_cell_present_and_useful[2][0][1] = ((uid_cells_around_parent_cell(2, 0, 1) != -1) && (flags_cells_around_parent_cell(2, 0, 1) & ItemFlags::II_Inactive));
      is_cell_around_parent_cell_present_and_useful[2][0][2] = ((uid_cells_around_parent_cell(2, 0, 2) != -1) && (flags_cells_around_parent_cell(2, 0, 2) & ItemFlags::II_Inactive));
      is_cell_around_parent_cell_present_and_useful[2][1][0] = ((uid_cells_around_parent_cell(2, 1, 0) != -1) && (flags_cells_around_parent_cell(2, 1, 0) & ItemFlags::II_Inactive));
      is_cell_around_parent_cell_present_and_useful[2][1][1] = ((uid_cells_around_parent_cell(2, 1, 1) != -1) && (flags_cells_around_parent_cell(2, 1, 1) & ItemFlags::II_Inactive));
      is_cell_around_parent_cell_present_and_useful[2][1][2] = ((uid_cells_around_parent_cell(2, 1, 2) != -1) && (flags_cells_around_parent_cell(2, 1, 2) & ItemFlags::II_Inactive));
      is_cell_around_parent_cell_present_and_useful[2][2][0] = ((uid_cells_around_parent_cell(2, 2, 0) != -1) && (flags_cells_around_parent_cell(2, 2, 0) & ItemFlags::II_Inactive));
      is_cell_around_parent_cell_present_and_useful[2][2][1] = ((uid_cells_around_parent_cell(2, 2, 1) != -1) && (flags_cells_around_parent_cell(2, 2, 1) & ItemFlags::II_Inactive));
      is_cell_around_parent_cell_present_and_useful[2][2][2] = ((uid_cells_around_parent_cell(2, 2, 2) != -1) && (flags_cells_around_parent_cell(2, 2, 2) & ItemFlags::II_Inactive));


      // En plus de regarder si chaque maille parent autour de notre maille parent existe et possède (II_Inactive) ou possédera (II_Refine) des enfants...
      // ... on regarde si chaque maille parent est présente sur notre sous-domaine, que ce soit une maille fantôme ou non.
      auto is_cell_around_parent_cell_in_subdomain = [&](Integer z, Integer y, Integer x){
        return is_cell_around_parent_cell_present_and_useful[z][y][x] && (flags_cells_around_parent_cell(z, y, x) & ItemFlags::II_UserMark1);
      };

      // ... on regarde si chaque maille parent est possédé par le même propriétaire que notre maille parent.
      auto is_cell_around_parent_cell_same_owner = [&](Integer z, Integer y, Integer x){
        return is_cell_around_parent_cell_present_and_useful[z][y][x] && (owner_cells_around_parent_cell(z, y, x) == owner_cells_around_parent_cell(1, 1, 1));
      };

      // ... on regarde si chaque maille parent a un propriétaire différent par rapport à notre maille parent.
      auto is_cell_around_parent_cell_different_owner = [&](Integer z, Integer y, Integer x){
        return is_cell_around_parent_cell_present_and_useful[z][y][x] && (owner_cells_around_parent_cell(z, y, x) != owner_cells_around_parent_cell(1, 1, 1));
      };


      // On itère sur toutes les mailles enfants.
      for (Int64 k = child_coord_z; k < child_coord_z + pattern; ++k) {
        for (Int64 j = child_coord_y; j < child_coord_y + pattern; ++j) {
          for (Int64 i = child_coord_x; i < child_coord_x + pattern; ++i) {
            parent_cells.add(parent_cell);
            total_nb_cells++;

            const Int64 child_cell_uid = m_num_mng->getCellUid(parent_cell_level + 1, i, j, k);
            debug() << "Child -- x : " << i << " -- y : " << j << " -- z : " << k << " -- level : " << parent_cell_level + 1 << " -- uid : " << child_cell_uid;

            m_num_mng->getNodeUids(child_nodes_uids, parent_cell_level + 1, i, j, k);
            m_num_mng->getFaceUids(child_faces_uids, parent_cell_level + 1, i, j, k);

            const Integer type_cell = IT_Hexaedron8;
            const Integer type_face = IT_Quad4;

            // Partie Cell.
            cells_infos.add(type_cell);
            cells_infos.add(child_cell_uid);
            for (Integer nc = 0; nc < m_num_mng->getNbNode(); nc++) {
              cells_infos.add(child_nodes_uids[nc]);
            }

            // Partie Face.
            for(Integer l = 0; l < m_num_mng->getNbFace(); ++l){
              Integer child_face_owner = -1;
              bool is_new_face = false;

              // Deux parties :
              // D'abord, on regarde si l'on doit créer la face l. Pour cela, on doit regarder si elle est présente sur la
              // maille à côté.
              // Pour gauche/bas/arrière, c'est le même principe. Si la maille enfant est tout à gauche/bas/arrière de la maille parente, on regarde
              // s'il y a une maille parente à gauche/bas/arrière. Sinon, on crée la face. Si oui, on regarde le masque pour savoir si l'on
              // doit créer la face.
              // Pour droite/haut/avant, le principe est différent de gauche/bas/arrière. On ne suit le masque que si on est tout à droite/haut/avant
              // de la maille parente. Sinon on crée toujours les faces droites/hautes/avant.
              // Enfin, on utilise le tableau "is_cell_around_parent_cell_in_subdomain". Si la maille parente d'à côté est sur
              // notre sous-domaine, alors il se peut que les faces en communes avec notre maille parente existent déjà, dans ce cas,
              // pas de doublon.
              if (
                ( (i == child_coord_x && !is_cell_around_parent_cell_in_subdomain(1, 1, 0)) || mask_face_if_cell_left[l] )
                &&
                ( (i != (child_coord_x + pattern-1) || !is_cell_around_parent_cell_in_subdomain(1, 1, 2)) || mask_face_if_cell_right[l] )
                &&
                ( (j == child_coord_y && !is_cell_around_parent_cell_in_subdomain(1, 0, 1)) || mask_face_if_cell_bottom[l] )
                &&
                ( (j != (child_coord_y + pattern-1) || !is_cell_around_parent_cell_in_subdomain(1, 2, 1)) || mask_face_if_cell_top[l] )
                &&
                ( (k == child_coord_z && !is_cell_around_parent_cell_in_subdomain(0, 1, 1)) || mask_face_if_cell_rear[l] )
                &&
                ( (k != (child_coord_z + pattern-1) || !is_cell_around_parent_cell_in_subdomain(2, 1, 1)) || mask_face_if_cell_front[l] )
              ){
                is_new_face = true;
                faces_infos.add(type_face);
                faces_infos.add(child_faces_uids[l]);

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
                  faces_infos.add(child_nodes_uids[nc]);
                }
                total_nb_faces++;

                // Par défaut, parent_cell est propriétaire de la nouvelle face.
                child_face_owner = owner_cells_around_parent_cell(1, 1, 1);
              }

              // Deuxième partie.
              // On doit maintenant trouver le bon propriétaire pour la face. Mis à part le tableau "is_cell_around_parent_cell_same_owner",
              // la condition est identique à celle au-dessus.
              // Le changement de tableau est important puisqu'à partir d'ici, on est sûr qu'il y a la face qui nous intéresse.
              // Le nouveau tableau permet de savoir si la maille d'à côté est aussi à nous ou pas. Si ce n'est pas le cas, alors
              // un changement de propriétaire est possible, selon les priorités définies au-dessus. On n'a pas besoin de savoir
              // si la maille est présente sur le sous-domaine.
              if (
                ( (i == child_coord_x && !is_cell_around_parent_cell_same_owner(1, 1, 0)) || mask_face_if_cell_left[l] )
                &&
                ( (i != (child_coord_x + pattern-1) || !is_cell_around_parent_cell_same_owner(1, 1, 2)) || mask_face_if_cell_right[l] )
                &&
                ( (j == child_coord_y && !is_cell_around_parent_cell_same_owner(1, 0, 1)) || mask_face_if_cell_bottom[l] )
                &&
                ( (j != (child_coord_y + pattern-1) || !is_cell_around_parent_cell_same_owner(1, 2, 1)) || mask_face_if_cell_top[l] )
                &&
                ( (k == child_coord_z && !is_cell_around_parent_cell_same_owner(0, 1, 1)) || mask_face_if_cell_rear[l] )
                &&
                ( (k != (child_coord_z + pattern-1) || !is_cell_around_parent_cell_same_owner(2, 1, 1)) || mask_face_if_cell_front[l] )
              )
              {
                // Ici, la construction des conditions est la même à chaque fois.
                // Le premier booléen (i == child_coord_x) regarde si l'enfant se trouve
                // du bon côté de la maille parent.
                // Le second booléen (!mask_face_if_cell_left[l]) nous dit si la face l est bien
                // la face en commun avec la maille parent d'à côté.
                // Le troisième booléen (is_cell_around_parent_cell_different_owner(1, 0)) regarde s'il y a une
                // maille à côté qui prend la propriété de la face ou à qui on prend la propriété.

                // En outre, il y a deux cas différents selon les priorités définies au-dessus :
                // - soit nous ne sommes pas prioritaire, alors on attribue le propriétaire prioritaire à notre face,
                // - soit nous sommes prioritaire, alors on se positionne comme propriétaire de la face et on doit prévenir
                //   tous les autres processus (le processus ancien propriétaire mais aussi les processus qui peuvent
                //   avoir la face en fantôme).

                // Enfin, dans le cas du changement de propriétaire, seul le processus (re)prenant la propriété doit
                // faire une communication à ce propos. Les processus ne possédant que la face en fantôme ne doit pas
                // faire de communication (mais ils peuvent définir localement le bon propriétaire, TODO Optimisation possible ?).

                // À gauche, priorité 12 < 13 donc il prend la propriété de la face.
                if(i == child_coord_x && (!mask_face_if_cell_left[l]) && is_cell_around_parent_cell_different_owner(1, 1, 0)){
                  child_face_owner = owner_cells_around_parent_cell(1, 1, 0);
                }

                // En bas, priorité 10 < 13 donc il prend la propriété de la face.
                else if(j == child_coord_y && (!mask_face_if_cell_bottom[l]) && is_cell_around_parent_cell_different_owner(1, 0, 1)){
                  child_face_owner = owner_cells_around_parent_cell(1, 0, 1);
                }

                // À l'arrière, priorité 4 < 13 donc il prend la propriété de la face.
                else if(k == child_coord_z && (!mask_face_if_cell_rear[l]) && is_cell_around_parent_cell_different_owner(0, 1, 1)){
                  child_face_owner = owner_cells_around_parent_cell(0, 1, 1);
                }

                // Sinon, parent_cell est propriétaire de la face.
                else {

                  // Seul le propriétaire de la parent_cell peut communiquer sur le changement de propriété.
                  if (parent_cell_is_own) {

                    // À droite, priorité 14 > 13 donc parent_cell récupère la propriété de la face.
                    if (i == (child_coord_x + pattern - 1) && (!mask_face_if_cell_right[l]) && is_cell_around_parent_cell_different_owner(1, 1, 2)) {
                      get_back_face_owner[owner_cells_around_parent_cell(1, 1, 2)][1]++;
                      get_back_face_owner[owner_cells_around_parent_cell(1, 1, 2)].add(child_faces_uids[l]);
                    }

                    // En haut, priorité 16 > 13 donc parent_cell récupère la propriété de la face.
                    else if (j == (child_coord_y + pattern - 1) && (!mask_face_if_cell_top[l]) && is_cell_around_parent_cell_different_owner(1, 2, 1)) {
                      get_back_face_owner[owner_cells_around_parent_cell(1, 2, 1)][1]++;
                      get_back_face_owner[owner_cells_around_parent_cell(1, 2, 1)].add(child_faces_uids[l]);
                    }

                    // À l'avant, priorité 22 > 13 donc parent_cell récupère la propriété de la face.
                    else if (k == (child_coord_z + pattern - 1) && (!mask_face_if_cell_front[l]) && is_cell_around_parent_cell_different_owner(2, 1, 1)) {
                      get_back_face_owner[owner_cells_around_parent_cell(2, 1, 1)][1]++;
                      get_back_face_owner[owner_cells_around_parent_cell(2, 1, 1)].add(child_faces_uids[l]);
                    }
                  }

                  // Sinon, c'est une face interne donc au parent_cell.
                  child_face_owner = owner_cells_around_parent_cell(1, 1, 1);
                }
              }

              // S'il y a une création de face et/ou un changement de propriétaire.
              if(child_face_owner != -1){
                face_uid_to_owner[child_faces_uids[l]] = child_face_owner;

                // Lorsqu'il y a un changement de propriétaire sans création de face,
                // on doit mettre de côté les uniqueIds de ces faces pour pouvoir
                // itérer dessus par la suite.
                if(!is_new_face){
                  face_uid_change_owner_only.add(child_faces_uids[l]);
                  debug() << "Child face (change owner) -- x : " << i
                          << " -- y : " << j
                          << " -- z : " << k
                          << " -- level : " << parent_cell_level + 1
                          << " -- face : " << l
                          << " -- uid_face : " << child_faces_uids[l]
                          << " -- owner : " << child_face_owner
                  ;
                }
                else{
                  debug() << "Child face (create face)  -- x : " << i
                          << " -- y : " << j
                          << " -- z : " << k
                          << " -- level : " << parent_cell_level + 1
                          << " -- face : " << l
                          << " -- uid_face : " << child_faces_uids[l]
                          << " -- owner : " << child_face_owner
                  ;
                }

              }
            }


            // Partie Node.
            // Cette partie est assez ressemblante à la partie face, mis à part le fait qu'il peut y avoir
            // plus de propriétaires possibles.
            for(Integer l = 0; l < m_num_mng->getNbNode(); ++l){
              Integer child_node_owner = -1;
              bool is_new_node = false;

              // Deux parties :
              // D'abord, on regarde si l'on doit créer le noeud l. Pour cela, on doit regarder s'il est présente sur la
              // maille à côté.
              // Pour gauche/bas/arrière, c'est le même principe. Si la maille enfant est tout à gauche/bas/arrière de la maille parente, on regarde
              // s'il y a une maille parente à gauche/bas/arrière. Sinon, on crée le noeud. Si oui, on regarde le masque pour savoir si l'on
              // doit créer le noeud.
              // Pour droite/haut/avant, le principe est différent de gauche/bas/arrière. On ne suit le masque que si la maille
              // enfant est toute à droite/haut/avant
              // de la maille parente. Sinon on crée toujours les noeuds droites/hautes/avant.
              // Enfin, on utilise le tableau "is_cell_around_parent_cell_in_subdomain". Si la maille parente d'à côté est sur
              // notre sous-domaine, alors il se peut que les noeuds en communs avec notre maille parente existent déjà, dans ce cas,
              // pas de doublon.
              if (
                ( (i == child_coord_x && !is_cell_around_parent_cell_in_subdomain(1, 1, 0)) || mask_node_if_cell_left[l] )
                &&
                ( (i != (child_coord_x + pattern-1) || !is_cell_around_parent_cell_in_subdomain(1, 1, 2)) || mask_node_if_cell_right[l] )
                &&
                ( (j == child_coord_y && !is_cell_around_parent_cell_in_subdomain(1, 0, 1)) || mask_node_if_cell_bottom[l] )
                &&
                ( (j != (child_coord_y + pattern-1) || !is_cell_around_parent_cell_in_subdomain(1, 2, 1)) || mask_node_if_cell_top[l] )
                &&
                ( (k == child_coord_z && !is_cell_around_parent_cell_in_subdomain(0, 1, 1)) || mask_node_if_cell_rear[l] )
                &&
                ( (k != (child_coord_z + pattern-1) || !is_cell_around_parent_cell_in_subdomain(2, 1, 1)) || mask_node_if_cell_front[l] )
              )
              {
                is_new_node = true;
                nodes_infos.add(child_nodes_uids[l]);
                total_nb_nodes++;

                // Par défaut, parent_cell est propriétaire du nouveau noeud.
                child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
              }

              // Deuxième partie.
              // On doit maintenant trouver le bon propriétaire pour le noeud. Mis à part le tableau "is_cell_around_parent_cell_same_owner",
              // la condition est identique à celle au-dessus.
              // Le changement de tableau est important puisqu'à partir d'ici, on est sûr que le noeud qui nous intéresse existe.
              // Le nouveau tableau permet de savoir si la maille d'à côté est aussi à nous ou pas. Si ce n'est pas le cas, alors
              // un changement de propriétaire est possible, selon les priorités définies au-dessus. On n'a pas besoin de savoir
              // si la maille est présente sur le sous-domaine.
              if (
                ( (i == child_coord_x && !is_cell_around_parent_cell_same_owner(1, 1, 0)) || mask_node_if_cell_left[l] )
                &&
                ( (i != (child_coord_x + pattern-1) || !is_cell_around_parent_cell_same_owner(1, 1, 2)) || mask_node_if_cell_right[l] )
                &&
                ( (j == child_coord_y && !is_cell_around_parent_cell_same_owner(1, 0, 1)) || mask_node_if_cell_bottom[l] )
                &&
                ( (j != (child_coord_y + pattern-1) || !is_cell_around_parent_cell_same_owner(1, 2, 1)) || mask_node_if_cell_top[l] )
                &&
                ( (k == child_coord_z && !is_cell_around_parent_cell_same_owner(0, 1, 1)) || mask_node_if_cell_rear[l] )
                &&
                ( (k != (child_coord_z + pattern-1) || !is_cell_around_parent_cell_same_owner(2, 1, 1)) || mask_node_if_cell_front[l] )
              )
              {

                // Par rapport aux faces qui n'ont que deux propriétaires possibles, un noeud peut
                // en avoir jusqu'à huit.

                // Si le noeud est sur la face gauche de la maille parente.
                if(i == child_coord_x && (!mask_node_if_cell_left[l])){

                  // Si le noeud est sur la face basse de la maille parente.
                  // Donc noeud sur l'arête à gauche en bas.
                  if(j == child_coord_y && (!mask_node_if_cell_bottom[l])) {

                    // Si le noeud est sur la face arrière de la maille parente.
                    // Donc noeud à gauche, en bas, en arrière (même position que le noeud de la maille parente).
                    if(k == child_coord_z && (!mask_node_if_cell_rear[l])) {

                      // Priorité 0 < 13.
                      if (is_cell_around_parent_cell_different_owner(0, 0, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 0, 0);
                      }

                      // Priorité 1 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 0, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 0, 1);
                      }

                      // Priorité 3 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 1, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 0);
                      }

                      // Priorité 4 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 1, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 1);
                      }

                      // Priorité 9 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 0, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 0);
                      }

                      // Priorité 10 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 0, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 1);
                      }

                      // Priorité 12 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 1, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                      }

                      // Pas de mailles autour.
                      else {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // Si le noeud est sur la face avant de la maille parente.
                    // Donc noeud à gauche, en bas, en avant (même position que le noeud de la maille parente).
                    else if(k == (child_coord_z + pattern-1) && (!mask_node_if_cell_front[l])) {

                      // Priorité 9 < 13.
                      if (is_cell_around_parent_cell_different_owner(1, 0, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 0);
                      }

                      // Priorité 10 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 0, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 1);
                      }

                      // Priorité 12 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 1, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                      }

                      // Sinon, parent_cell est propriétaire du noeud.
                      else {

                        // Seul le propriétaire de la parent_cell peut communiquer sur le changement de propriété.
                        if (parent_cell_is_own) {

                          // Priorité 18 > 13.
                          if (is_cell_around_parent_cell_different_owner(2, 0, 0)) {
                            get_back_node_owner[owner_cells_around_parent_cell(2, 0, 0)][1]++;
                            get_back_node_owner[owner_cells_around_parent_cell(2, 0, 0)].add(child_nodes_uids[l]);
                          }

                          // Priorité 19 > 13.
                          if (is_cell_around_parent_cell_different_owner(2, 0, 1)) {
                            get_back_node_owner[owner_cells_around_parent_cell(2, 0, 1)][1]++;
                            get_back_node_owner[owner_cells_around_parent_cell(2, 0, 1)].add(child_nodes_uids[l]);
                          }

                          // Priorité 21 > 13.
                          if (is_cell_around_parent_cell_different_owner(2, 1, 0)) {
                            get_back_node_owner[owner_cells_around_parent_cell(2, 1, 0)][1]++;
                            get_back_node_owner[owner_cells_around_parent_cell(2, 1, 0)].add(child_nodes_uids[l]);
                          }

                          // Priorité 22 > 13.
                          if (is_cell_around_parent_cell_different_owner(2, 1, 1)) {
                            get_back_node_owner[owner_cells_around_parent_cell(2, 1, 1)][1]++;
                            get_back_node_owner[owner_cells_around_parent_cell(2, 1, 1)].add(child_nodes_uids[l]);
                          }
                        }

                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // Sinon le noeud est quelque part sur l'arête à gauche en bas...
                    else{

                      // Priorité 9 < 13.
                      if (is_cell_around_parent_cell_different_owner(1, 0, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 0);
                      }

                      // Priorité 10 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 0, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 1);
                      }

                      // Priorité 12 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 1, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                      }

                      // Pas de mailles autour.
                      else {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }
                  }

                  // Si le noeud est sur la face haute de la maille parente.
                  // Donc noeud sur l'arête à gauche en haut.
                  else if(j == (child_coord_y + pattern-1) && (!mask_node_if_cell_top[l])) {

                    // Si le noeud est sur la face arrière de la maille parente.
                    // Donc noeud à gauche, en haut, en arrière (même position que le noeud de la maille parente).
                    if(k == child_coord_z && (!mask_node_if_cell_rear[l])) {

                      // Priorité 3 < 13.
                      if (is_cell_around_parent_cell_different_owner(0, 1, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 0);
                      }

                      // Priorité 4 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 1, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 1);
                      }

                      // Priorité 6 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 2, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 2, 0);
                      }

                      // Priorité 7 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 2, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 2, 1);
                      }

                      // Priorité 12 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 1, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                      }

                      // Sinon, parent_cell est propriétaire du noeud.
                      else {

                        // Seul le propriétaire de la parent_cell peut communiquer sur le changement de propriété.
                        if (parent_cell_is_own) {

                          // Priorité 15 > 13.
                          if (is_cell_around_parent_cell_different_owner(1, 2, 0)) {
                            get_back_node_owner[owner_cells_around_parent_cell(1, 2, 0)][1]++;
                            get_back_node_owner[owner_cells_around_parent_cell(1, 2, 0)].add(child_nodes_uids[l]);
                          }

                          // Priorité 16 > 13.
                          if (is_cell_around_parent_cell_different_owner(1, 2, 1)) {
                            get_back_node_owner[owner_cells_around_parent_cell(1, 2, 1)][1]++;
                            get_back_node_owner[owner_cells_around_parent_cell(1, 2, 1)].add(child_nodes_uids[l]);
                          }
                        }

                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // Si le noeud est sur la face avant de la maille parente.
                    // Donc noeud à gauche, en haut, en avant (même position que le noeud de la maille parente).
                    else if(k == (child_coord_z + pattern-1) && (!mask_node_if_cell_front[l])) {

                      // Priorité 4 < 13.
                      if (is_cell_around_parent_cell_different_owner(1, 1, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                      }

                      // Sinon, parent_cell est propriétaire du noeud.
                      else {

                        // Seul le propriétaire de la parent_cell peut communiquer sur le changement de propriété.
                        if (parent_cell_is_own) {

                          // Priorité 15 > 13.
                          if (is_cell_around_parent_cell_different_owner(1, 2, 0)) {
                            get_back_node_owner[owner_cells_around_parent_cell(1, 2, 0)][1]++;
                            get_back_node_owner[owner_cells_around_parent_cell(1, 2, 0)].add(child_nodes_uids[l]);
                          }

                          // Priorité 16 > 13.
                          if (is_cell_around_parent_cell_different_owner(1, 2, 1)) {
                            get_back_node_owner[owner_cells_around_parent_cell(1, 2, 1)][1]++;
                            get_back_node_owner[owner_cells_around_parent_cell(1, 2, 1)].add(child_nodes_uids[l]);
                          }

                          // Priorité 21 > 13.
                          if (is_cell_around_parent_cell_different_owner(2, 1, 0)) {
                            get_back_node_owner[owner_cells_around_parent_cell(2, 1, 0)][1]++;
                            get_back_node_owner[owner_cells_around_parent_cell(2, 1, 0)].add(child_nodes_uids[l]);
                          }

                          // Priorité 22 > 13.
                          if (is_cell_around_parent_cell_different_owner(2, 1, 1)) {
                            get_back_node_owner[owner_cells_around_parent_cell(2, 1, 1)][1]++;
                            get_back_node_owner[owner_cells_around_parent_cell(2, 1, 1)].add(child_nodes_uids[l]);
                          }

                          // Priorité 24 > 13.
                          if (is_cell_around_parent_cell_different_owner(2, 2, 0)) {
                            get_back_node_owner[owner_cells_around_parent_cell(2, 2, 0)][1]++;
                            get_back_node_owner[owner_cells_around_parent_cell(2, 2, 0)].add(child_nodes_uids[l]);
                          }

                          // Priorité 25 > 13.
                          if (is_cell_around_parent_cell_different_owner(2, 2, 1)) {
                            get_back_node_owner[owner_cells_around_parent_cell(2, 2, 1)][1]++;
                            get_back_node_owner[owner_cells_around_parent_cell(2, 2, 1)].add(child_nodes_uids[l]);
                          }
                        }

                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // Sinon le noeud est quelque part sur l'arête à gauche en haut...
                    else {

                      // Priorité 12 < 13.
                      if (is_cell_around_parent_cell_different_owner(1, 1, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                      }

                      // Sinon, parent_cell est propriétaire du noeud.
                      else {

                        // Seul le propriétaire de la parent_cell peut communiquer sur le changement de propriété.
                        if (parent_cell_is_own) {

                          // Priorité 15 > 13.
                          if (is_cell_around_parent_cell_different_owner(1, 2, 0)) {
                            get_back_node_owner[owner_cells_around_parent_cell(1, 2, 0)][1]++;
                            get_back_node_owner[owner_cells_around_parent_cell(1, 2, 0)].add(child_nodes_uids[l]);
                          }

                          // Priorité 16 > 13.
                          if (is_cell_around_parent_cell_different_owner(1, 2, 1)) {
                            get_back_node_owner[owner_cells_around_parent_cell(1, 2, 1)][1]++;
                            get_back_node_owner[owner_cells_around_parent_cell(1, 2, 1)].add(child_nodes_uids[l]);
                          }
                        }
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }
                  }

                  // Sinon le noeud est ni sur l'arête à gauche en bas, ni sur l'arête à gauche en haut.
                  else {

                    // Si le noeud est quelque part sur l'arête à gauche en arrière.
                    if (k == child_coord_z && (!mask_node_if_cell_rear[l])) {

                      // Priorité 3 < 13.
                      if (is_cell_around_parent_cell_different_owner(0, 1, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 0);
                      }

                      // Priorité 4 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 1, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 1);
                      }

                      // Priorité 12 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 1, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                      }

                      // Pas de mailles autour.
                      else {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // Si le noeud est quelque part sur l'arête à gauche en avant.
                    else if (k == (child_coord_z + pattern - 1) && (!mask_node_if_cell_front[l])) {

                      // Priorité 12 < 13.
                      if (is_cell_around_parent_cell_different_owner(1, 1, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                      }

                      // Sinon, parent_cell est propriétaire du noeud.
                      else {

                        // Seul le propriétaire de la parent_cell peut communiquer sur le changement de propriété.
                        if (parent_cell_is_own) {

                          // Priorité 21 > 13.
                          if (is_cell_around_parent_cell_different_owner(2, 1, 0)) {
                            get_back_node_owner[owner_cells_around_parent_cell(2, 1, 0)][1]++;
                            get_back_node_owner[owner_cells_around_parent_cell(2, 1, 0)].add(child_nodes_uids[l]);
                          }

                          // Priorité 22 > 13.
                          if (is_cell_around_parent_cell_different_owner(2, 1, 1)) {
                            get_back_node_owner[owner_cells_around_parent_cell(2, 1, 1)][1]++;
                            get_back_node_owner[owner_cells_around_parent_cell(2, 1, 1)].add(child_nodes_uids[l]);
                          }
                        }

                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // Sinon le noeud est quelque part sur la face gauche...
                    else {

                      // Priorité 12 < 13.
                      if (is_cell_around_parent_cell_different_owner(1, 1, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                      }

                      // Parent_cell est le proprio.
                      else {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }
                  }
                }

                // À partir de là, on a exploré tous les noeuds et toutes les arêtes de la face parente gauche.

                // Si le noeud est sur la face droite de la maille parente.
                else if(i == (child_coord_x + pattern-1) && (!mask_node_if_cell_right[l])){

                  // Si le noeud est sur la face basse de la maille parente.
                  // Donc noeud sur l'arête à droite en bas.
                  if(j == child_coord_y && (!mask_node_if_cell_bottom[l])){

                    // Si le noeud est sur la face arrière de la maille parente.
                    // Donc noeud à droite, en bas, en arrière (même position que le noeud de la maille parente).
                    if(k == child_coord_z && (!mask_node_if_cell_rear[l])){

                      // Priorité 1 < 13.
                      if (is_cell_around_parent_cell_different_owner(0, 0, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 0, 1);
                      }

                      // Priorité 2 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 0, 2)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 0, 2);
                      }

                      // Priorité 4 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 1, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 1);
                      }

                      // Priorité 5 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 1, 2)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 2);
                      }

                      // Priorité 10 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 0, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 1);
                      }

                      // Priorité 11 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 0, 2)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 2);
                      }

                      // Sinon, parent_cell est propriétaire du noeud.
                      else {

                        // Seul le propriétaire de la parent_cell peut communiquer sur le changement de propriété.
                        // Priorité 14 > 13.
                        if (is_cell_around_parent_cell_different_owner(1, 1, 2) && parent_cell_is_own) {
                          get_back_node_owner[owner_cells_around_parent_cell(1, 1, 2)][1]++;
                          get_back_node_owner[owner_cells_around_parent_cell(1, 1, 2)].add(child_nodes_uids[l]);
                        }

                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // Si le noeud est sur la face avant de la maille parente.
                    // Donc noeud à droite, en bas, en avant (même position que le noeud de la maille parente).
                    else if(k == (child_coord_z + pattern-1) && (!mask_node_if_cell_front[l])) {

                      // Priorité 10 < 13.
                      if (is_cell_around_parent_cell_different_owner(1, 0, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 1);
                      }

                      // Priorité 11 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 0, 2)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 2);
                      }

                      // Sinon, parent_cell est propriétaire du noeud.
                      else {

                        // Seul le propriétaire de la parent_cell peut communiquer sur le changement de propriété.
                        if (parent_cell_is_own) {

                          // Priorité 14 > 13.
                          if (is_cell_around_parent_cell_different_owner(1, 1, 2)) {
                            get_back_node_owner[owner_cells_around_parent_cell(1, 1, 2)][1]++;
                            get_back_node_owner[owner_cells_around_parent_cell(1, 1, 2)].add(child_nodes_uids[l]);
                          }

                          // Priorité 19 > 13.
                          if (is_cell_around_parent_cell_different_owner(2, 0, 1)) {
                            get_back_node_owner[owner_cells_around_parent_cell(2, 0, 1)][1]++;
                            get_back_node_owner[owner_cells_around_parent_cell(2, 0, 1)].add(child_nodes_uids[l]);
                          }

                          // Priorité 20 > 13.
                          if (is_cell_around_parent_cell_different_owner(2, 0, 2)) {
                            get_back_node_owner[owner_cells_around_parent_cell(2, 0, 2)][1]++;
                            get_back_node_owner[owner_cells_around_parent_cell(2, 0, 2)].add(child_nodes_uids[l]);
                          }

                          // Priorité 22 > 13.
                          if (is_cell_around_parent_cell_different_owner(2, 1, 1)) {
                            get_back_node_owner[owner_cells_around_parent_cell(2, 1, 1)][1]++;
                            get_back_node_owner[owner_cells_around_parent_cell(2, 1, 1)].add(child_nodes_uids[l]);
                          }

                          // Priorité 23 > 13.
                          if (is_cell_around_parent_cell_different_owner(2, 1, 2)) {
                            get_back_node_owner[owner_cells_around_parent_cell(2, 1, 2)][1]++;
                            get_back_node_owner[owner_cells_around_parent_cell(2, 1, 2)].add(child_nodes_uids[l]);
                          }
                        }

                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // Sinon le noeud est quelque part sur l'arête à droite en bas...
                    else {

                      // Priorité 10 < 13.
                      if (is_cell_around_parent_cell_different_owner(1, 0, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 1);
                      }

                      // Priorité 11 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 0, 2)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 2);
                      }

                      // Sinon, parent_cell est propriétaire du noeud.
                      else {

                        // Seul le propriétaire de la parent_cell peut communiquer sur le changement de propriété.
                        // Priorité 14 > 13.
                        if (is_cell_around_parent_cell_different_owner(1, 1, 2) && parent_cell_is_own) {
                          get_back_node_owner[owner_cells_around_parent_cell(1, 1, 2)][1]++;
                          get_back_node_owner[owner_cells_around_parent_cell(1, 1, 2)].add(child_nodes_uids[l]);
                        }

                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }
                  }

                  // Si le noeud est sur la face haute de la maille parente.
                  // Donc noeud sur l'arête à droite en haut.
                  else if(j == (child_coord_y + pattern-1) && (!mask_node_if_cell_top[l])) {

                    // Si le noeud est sur la face arrière de la maille parente.
                    // Donc noeud à droite, en haut, en arrière (même position que le noeud de la maille parente).
                    if(k == child_coord_z && (!mask_node_if_cell_rear[l])) {

                      // Priorité 4 < 13.
                      if (is_cell_around_parent_cell_different_owner(0, 1, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 1);
                      }

                      // Priorité 5 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 1, 2)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 2);
                      }

                      // Priorité 7 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 2, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 2, 1);
                      }

                      // Priorité 8 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 2, 2)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 2, 2);
                      }

                      // Sinon, parent_cell est propriétaire du noeud.
                      else {

                        // Seul le propriétaire de la parent_cell peut communiquer sur le changement de propriété.
                        if (parent_cell_is_own) {

                          // Priorité 14 > 13.
                          if (is_cell_around_parent_cell_different_owner(1, 1, 2)) {
                            get_back_node_owner[owner_cells_around_parent_cell(1, 1, 2)][1]++;
                            get_back_node_owner[owner_cells_around_parent_cell(1, 1, 2)].add(child_nodes_uids[l]);
                          }

                          // Priorité 16 > 13.
                          if (is_cell_around_parent_cell_different_owner(1, 2, 1)) {
                            get_back_node_owner[owner_cells_around_parent_cell(1, 2, 1)][1]++;
                            get_back_node_owner[owner_cells_around_parent_cell(1, 2, 1)].add(child_nodes_uids[l]);
                          }

                          // Priorité 17 > 13.
                          if (is_cell_around_parent_cell_different_owner(1, 2, 2)) {
                            get_back_node_owner[owner_cells_around_parent_cell(1, 2, 2)][1]++;
                            get_back_node_owner[owner_cells_around_parent_cell(1, 2, 2)].add(child_nodes_uids[l]);
                          }
                        }

                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // Si le noeud est sur la face avant de la maille parente.
                    // Donc noeud à droite, en haut, en avant (même position que le noeud de la maille parente).
                    else if(k == (child_coord_z + pattern-1) && (!mask_node_if_cell_front[l])) {

                      // Seul le propriétaire de la parent_cell peut communiquer sur le changement de propriété.
                      if (parent_cell_is_own) {

                        // Priorité 14 > 13.
                        if (is_cell_around_parent_cell_different_owner(1, 1, 2)) {
                          get_back_node_owner[owner_cells_around_parent_cell(1, 1, 2)][1]++;
                          get_back_node_owner[owner_cells_around_parent_cell(1, 1, 2)].add(child_nodes_uids[l]);
                        }

                        // Priorité 16 > 13.
                        if (is_cell_around_parent_cell_different_owner(1, 2, 1)) {
                          get_back_node_owner[owner_cells_around_parent_cell(1, 2, 1)][1]++;
                          get_back_node_owner[owner_cells_around_parent_cell(1, 2, 1)].add(child_nodes_uids[l]);
                        }

                        // Priorité 17 > 13.
                        if (is_cell_around_parent_cell_different_owner(1, 2, 2)) {
                          get_back_node_owner[owner_cells_around_parent_cell(1, 2, 2)][1]++;
                          get_back_node_owner[owner_cells_around_parent_cell(1, 2, 2)].add(child_nodes_uids[l]);
                        }

                        // Priorité 22 > 13.
                        if (is_cell_around_parent_cell_different_owner(2, 1, 1)) {
                          get_back_node_owner[owner_cells_around_parent_cell(2, 1, 1)][1]++;
                          get_back_node_owner[owner_cells_around_parent_cell(2, 1, 1)].add(child_nodes_uids[l]);
                        }

                        // Priorité 23 > 13.
                        if (is_cell_around_parent_cell_different_owner(2, 1, 2)) {
                          get_back_node_owner[owner_cells_around_parent_cell(2, 1, 2)][1]++;
                          get_back_node_owner[owner_cells_around_parent_cell(2, 1, 2)].add(child_nodes_uids[l]);
                        }

                        // Priorité 25 > 13.
                        if (is_cell_around_parent_cell_different_owner(2, 2, 1)) {
                          get_back_node_owner[owner_cells_around_parent_cell(2, 2, 1)][1]++;
                          get_back_node_owner[owner_cells_around_parent_cell(2, 2, 1)].add(child_nodes_uids[l]);
                        }

                        // Priorité 26 > 13.
                        if (is_cell_around_parent_cell_different_owner(2, 2, 2)) {
                          get_back_node_owner[owner_cells_around_parent_cell(2, 2, 2)][1]++;
                          get_back_node_owner[owner_cells_around_parent_cell(2, 2, 2)].add(child_nodes_uids[l]);
                        }
                      }

                      child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                    }

                    // Sinon le noeud est quelque part sur l'arête à droite en haut...
                    else {

                      // Seul le propriétaire de la parent_cell peut communiquer sur le changement de propriété.
                      if (parent_cell_is_own) {

                        // Priorité 14 > 13.
                        if (is_cell_around_parent_cell_different_owner(1, 1, 2)) {
                          get_back_node_owner[owner_cells_around_parent_cell(1, 1, 2)][1]++;
                          get_back_node_owner[owner_cells_around_parent_cell(1, 1, 2)].add(child_nodes_uids[l]);
                        }

                        // Priorité 16 > 13.
                        if (is_cell_around_parent_cell_different_owner(1, 2, 1)) {
                          get_back_node_owner[owner_cells_around_parent_cell(1, 2, 1)][1]++;
                          get_back_node_owner[owner_cells_around_parent_cell(1, 2, 1)].add(child_nodes_uids[l]);
                        }

                        // Priorité 17 > 13.
                        if (is_cell_around_parent_cell_different_owner(1, 2, 2)) {
                          get_back_node_owner[owner_cells_around_parent_cell(1, 2, 2)][1]++;
                          get_back_node_owner[owner_cells_around_parent_cell(1, 2, 2)].add(child_nodes_uids[l]);
                        }
                      }

                      child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                    }
                  }

                  // Sinon le noeud est ni sur l'arête à droite en bas, ni sur l'arête à droite en haut.
                  else {
                    // Si le noeud est quelque part sur l'arête à droite en arrière.
                    if (k == child_coord_z && (!mask_node_if_cell_rear[l])) {

                      // Priorité 4 < 13.
                      if (is_cell_around_parent_cell_different_owner(0, 1, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 1);
                      }

                      // Priorité 5 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 1, 2)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 2);
                      }

                      // Sinon, parent_cell est propriétaire du noeud.
                      else {

                        // Seul le propriétaire de la parent_cell peut communiquer sur le changement de propriété.
                        // Priorité 14 > 13.
                        if (is_cell_around_parent_cell_different_owner(1, 1, 2) && parent_cell_is_own) {
                          get_back_node_owner[owner_cells_around_parent_cell(1, 1, 2)][1]++;
                          get_back_node_owner[owner_cells_around_parent_cell(1, 1, 2)].add(child_nodes_uids[l]);
                        }

                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // Si le noeud est quelque part sur l'arête à droite en avant.
                    else if (k == (child_coord_z + pattern - 1) && (!mask_node_if_cell_front[l])) {

                      // Seul le propriétaire de la parent_cell peut communiquer sur le changement de propriété.
                      if (parent_cell_is_own) {

                        // Priorité 14 > 13.
                        if (is_cell_around_parent_cell_different_owner(1, 1, 2)) {
                          get_back_node_owner[owner_cells_around_parent_cell(1, 1, 2)][1]++;
                          get_back_node_owner[owner_cells_around_parent_cell(1, 1, 2)].add(child_nodes_uids[l]);
                        }

                        // Priorité 22 > 13.
                        if (is_cell_around_parent_cell_different_owner(2, 1, 1)) {
                          get_back_node_owner[owner_cells_around_parent_cell(2, 1, 1)][1]++;
                          get_back_node_owner[owner_cells_around_parent_cell(2, 1, 1)].add(child_nodes_uids[l]);
                        }

                        // Priorité 23 > 13.
                        if (is_cell_around_parent_cell_different_owner(2, 1, 2)) {
                          get_back_node_owner[owner_cells_around_parent_cell(2, 1, 2)][1]++;
                          get_back_node_owner[owner_cells_around_parent_cell(2, 1, 2)].add(child_nodes_uids[l]);
                        }
                      }

                      child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                    }

                    // Sinon le noeud est quelque part sur la face droite...
                    else {

                      // Seul le propriétaire de la parent_cell peut communiquer sur le changement de propriété.
                      // Priorité 14 > 13.
                      if (is_cell_around_parent_cell_different_owner(1, 1, 2) && parent_cell_is_own) {
                        get_back_node_owner[owner_cells_around_parent_cell(1, 1, 2)][1]++;
                        get_back_node_owner[owner_cells_around_parent_cell(1, 1, 2)].add(child_nodes_uids[l]);
                      }

                      child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                    }
                  }
                }

                // À partir de là, on a exploré tous les noeuds de la maille parente et toutes les arêtes
                // de la face parente droite (et gauche).
                // Donc il ne reste que quatre arêtes et quatre faces à explorer.

                // Sinon le noeud est ni sur la face gauche, ni sur la face droite.
                else {

                  // Si le noeud est sur la face basse de la maille parente.
                  if (j == child_coord_y && (!mask_node_if_cell_bottom[l])) {

                    // Si le noeud est sur la face arrière de la maille parente.
                    // Donc noeud sur l'arête en arrière en bas.
                    if (k == child_coord_z && (!mask_node_if_cell_rear[l])) {

                      // Priorité 1 < 13.
                      if (is_cell_around_parent_cell_different_owner(0, 0, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 0, 1);
                      }

                      // Priorité 4 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 1, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 1);
                      }

                      // Priorité 10 < 13.
                      else if (is_cell_around_parent_cell_different_owner(1, 0, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 1);
                      }

                      // Pas de mailles autour.
                      else {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // Si le noeud est sur la face avant de la maille parente.
                    // Donc noeud sur l'arête en avant en bas.
                    else if (k == (child_coord_z + pattern - 1) && (!mask_node_if_cell_front[l])) {

                      // Priorité 10 < 13.
                      if (is_cell_around_parent_cell_different_owner(1, 0, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 1);
                      }

                      // Sinon, parent_cell est propriétaire du noeud.
                      else {

                        // Seul le propriétaire de la parent_cell peut communiquer sur le changement de propriété.
                        if (parent_cell_is_own) {

                          // Priorité 19 > 13.
                          if (is_cell_around_parent_cell_different_owner(2, 0, 1)) {
                            get_back_node_owner[owner_cells_around_parent_cell(2, 0, 1)][1]++;
                            get_back_node_owner[owner_cells_around_parent_cell(2, 0, 1)].add(child_nodes_uids[l]);
                          }

                          // Priorité 22 > 13.
                          if (is_cell_around_parent_cell_different_owner(2, 1, 1)) {
                            get_back_node_owner[owner_cells_around_parent_cell(2, 1, 1)][1]++;
                            get_back_node_owner[owner_cells_around_parent_cell(2, 1, 1)].add(child_nodes_uids[l]);
                          }
                        }

                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // Sinon le noeud est quelque part sur la face en bas...
                    else {

                      // Priorité 10 < 13.
                      if (is_cell_around_parent_cell_different_owner(1, 0, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 0, 1);
                      }

                      // Parent_cell est le proprio.
                      else {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }
                  }

                  // Si le noeud est sur la face haute de la maille parente.
                  else if (j == (child_coord_y + pattern - 1) && (!mask_node_if_cell_top[l])) {

                    // Si le noeud est sur la face arrière de la maille parente.
                    // Donc noeud sur l'arête en arrière en haut.
                    if (k == child_coord_z && (!mask_node_if_cell_rear[l])) {

                      // Priorité 4 < 13.
                      if (is_cell_around_parent_cell_different_owner(0, 1, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 1);
                      }

                      // Priorité 7 < 13.
                      else if (is_cell_around_parent_cell_different_owner(0, 2, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 2, 1);
                      }

                      // Sinon, parent_cell est propriétaire du noeud.
                      else {

                        // Seul le propriétaire de la parent_cell peut communiquer sur le changement de propriété.
                        // Priorité 16 > 13.
                        if (is_cell_around_parent_cell_different_owner(1, 2, 1) && parent_cell_is_own) {
                          get_back_node_owner[owner_cells_around_parent_cell(1, 2, 1)][1]++;
                          get_back_node_owner[owner_cells_around_parent_cell(1, 2, 1)].add(child_nodes_uids[l]);
                        }

                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // Si le noeud est sur la face avant de la maille parente.
                    // Donc noeud sur l'arête en avant en haut.
                    else if (k == (child_coord_z + pattern - 1) && (!mask_node_if_cell_front[l])) {

                      // Seul le propriétaire de la parent_cell peut communiquer sur le changement de propriété.
                      if (parent_cell_is_own) {

                        // Priorité 16 > 13.
                        if (is_cell_around_parent_cell_different_owner(1, 2, 1)) {
                          get_back_node_owner[owner_cells_around_parent_cell(1, 2, 1)][1]++;
                          get_back_node_owner[owner_cells_around_parent_cell(1, 2, 1)].add(child_nodes_uids[l]);
                        }

                        // Priorité 22 > 13.
                        if (is_cell_around_parent_cell_different_owner(2, 1, 1)) {
                          get_back_node_owner[owner_cells_around_parent_cell(2, 1, 1)][1]++;
                          get_back_node_owner[owner_cells_around_parent_cell(2, 1, 1)].add(child_nodes_uids[l]);
                        }

                        // Priorité 25 > 13.
                        if (is_cell_around_parent_cell_different_owner(2, 2, 1)) {
                          get_back_node_owner[owner_cells_around_parent_cell(2, 2, 1)][1]++;
                          get_back_node_owner[owner_cells_around_parent_cell(2, 2, 1)].add(child_nodes_uids[l]);
                        }
                      }

                      child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                    }

                    // Sinon le noeud est quelque part sur la face en haut...
                    else {

                      // Seul le propriétaire de la parent_cell peut communiquer sur le changement de propriété.
                      // Priorité 16 > 13.
                      if (is_cell_around_parent_cell_different_owner(1, 2, 1) && parent_cell_is_own) {
                        get_back_node_owner[owner_cells_around_parent_cell(1, 2, 1)][1]++;
                        get_back_node_owner[owner_cells_around_parent_cell(1, 2, 1)].add(child_nodes_uids[l]);
                      }

                      child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                    }
                  }

                  // Il ne reste plus que deux faces, la face arrière et la face avant...
                  else {

                    // Si le noeud est quelque part sur la face arrière...
                    if (k == child_coord_z && (!mask_node_if_cell_rear[l])) {

                      // Priorité 4 < 13.
                      if (is_cell_around_parent_cell_different_owner(0, 1, 1)) {
                        child_node_owner = owner_cells_around_parent_cell(0, 1, 1);
                      }

                      // Parent_cell est le proprio.
                      else {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // Si le noeud est quelque part sur la face avant...
                    else if (k == (child_coord_z + pattern - 1) && (!mask_node_if_cell_front[l])) {

                      // Seul le propriétaire de la parent_cell peut communiquer sur le changement de propriété.
                      // Priorité 22 < 13.
                      if (is_cell_around_parent_cell_different_owner(2, 1, 1) && parent_cell_is_own) {
                        get_back_node_owner[owner_cells_around_parent_cell(2, 1, 1)][1]++;
                        get_back_node_owner[owner_cells_around_parent_cell(2, 1, 1)].add(child_nodes_uids[l]);
                      }

                      child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                    }

                    // Sinon, le noeud est à l'intérieur de la maille parente.
                    else {
                      child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                    }
                  }
                }
              }

              // S'il y a une création de noeud et/ou un changement de propriétaire.
              if(child_node_owner != -1){
                node_uid_to_owner[child_nodes_uids[l]] = child_node_owner;

                // Lorsqu'il y a un changement de propriétaire sans création de noeud,
                // on doit mettre de côté les uniqueIds de ces noeuds pour pouvoir
                // itérer dessus par la suite.
                if(!is_new_node){
                  node_uid_change_owner_only.add(child_nodes_uids[l]);
                  debug() << "Child node (change owner) -- x : " << i
                          << " -- y : " << j
                          << " -- z : " << k
                          << " -- level : " << parent_cell_level + 1
                          << " -- node : " << l
                          << " -- uid_node : " << child_nodes_uids[l]
                          << " -- owner : " << child_node_owner;
                }
                else{
                  debug() << "Child node (create node)  -- x : " << i
                          << " -- y : " << j
                          << " -- z : " << k
                          << " -- level : " << parent_cell_level + 1
                          << " -- node : " << l
                          << " -- uid_node : " << child_nodes_uids[l]
                          << " -- owner : " << child_node_owner;
                }
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

  // Nodes
  {
    debug() << "Nb new nodes in patch : " << total_nb_nodes;
    {
      const Integer nb_node_owner_change = node_uid_change_owner_only.size();

      // Ce tableau contiendra les localIds des nouveaux noeuds mais aussi les localIds
      // des noeuds qui changent juste de propriétaire.
      UniqueArray<Int32> nodes_lid(total_nb_nodes + nb_node_owner_change);

      // On crée les noeuds. On met les localIds des nouveaux noeuds au début du tableau.
      m_mesh->modifier()->addNodes(nodes_infos, nodes_lid.subView(0, total_nb_nodes));

      // On cherche les localIds des noeuds qui changent de proprio et on les met à la fin du tableau.
      m_mesh->nodeFamily()->itemsUniqueIdToLocalId(nodes_lid.subView(total_nb_nodes, nb_node_owner_change), node_uid_change_owner_only, true);

      // On attribue les bons propriétaires aux noeuds.
      ENUMERATE_ (Node, inode, m_mesh->nodeFamily()->view(nodes_lid)) {
        Node node = *inode;
        node.mutableItemBase().setOwner(node_uid_to_owner[node.uniqueId()], my_rank);

        if (node_uid_to_owner[node.uniqueId()] == my_rank) {
          node.mutableItemBase().addFlags(ItemFlags::II_Own);
        }
      }
    }

    if(pm->commSize() != 1) {
      // On distribue les nouveaux propriétaires aux processus.
      UniqueArray<Int64> recv_buffer;

      for (Integer rank = 0; rank < nb_rank; ++rank) {
        pm->gatherVariable(get_back_node_owner[rank], recv_buffer, rank);
      }

      Integer index = 0;

      // On explore le tableau reçu contenant les nouveaux propriétaires pour nos noeuds.
      while (index < recv_buffer.size()) {

        // Le nouveau propriétaire des noeuds de cette partie du tableau.
        auto rank = static_cast<Integer>(recv_buffer[index++]);

        // Le nombre de uid de noeuds à suivre dans le tableau.
        auto size = static_cast<Integer>(recv_buffer[index++]);

        // On récupère les uid des noeuds devant changer de proprio.
        ConstArrayView<Int64> nodes_uid = recv_buffer.subView(index, size);
        index += size;

        UniqueArray<Int32> nodes_lid(size);
        m_mesh->nodeFamily()->itemsUniqueIdToLocalId(nodes_lid, nodes_uid, true);

        ENUMERATE_ (Node, inode, m_mesh->nodeFamily()->view(nodes_lid)) {
          Node node = *inode;
          debug() << "Change node owner -- UniqueId : " << node.uniqueId()
                  << " -- Old Owner : " << node.owner()
                  << " -- New Owner : " << rank
          ;
          node.mutableItemBase().setOwner(rank, my_rank);
        }
      }
    }

    m_mesh->nodeFamily()->notifyItemsOwnerChanged();
  }

  // Faces
  {
    debug() << "Nb new faces in patch : " << total_nb_faces;
    {
      const Integer nb_face_owner_change = face_uid_change_owner_only.size();

      // Ce tableau contiendra les localIds des nouvelles faces mais aussi les localIds
      // des faces qui changent juste de propriétaire.
      UniqueArray<Int32> faces_lid(total_nb_faces + nb_face_owner_change);

      // On crée les faces. On met les localIds des nouvelles faces au début du tableau.
      m_mesh->modifier()->addFaces(total_nb_faces, faces_infos, faces_lid.subView(0, total_nb_faces));

      // On cherche les localIds des faces qui changent de proprio et on les met à la fin du tableau.
      m_mesh->faceFamily()->itemsUniqueIdToLocalId(faces_lid.subView(total_nb_faces, nb_face_owner_change), face_uid_change_owner_only, true);

      // On attribue les bons propriétaires aux faces.
      ENUMERATE_ (Face, iface, m_mesh->faceFamily()->view(faces_lid)) {
        Face face = *iface;
        face.mutableItemBase().setOwner(face_uid_to_owner[face.uniqueId()], my_rank);

        if (face_uid_to_owner[face.uniqueId()] == my_rank) {
          face.mutableItemBase().addFlags(ItemFlags::II_Own);
        }
      }
    }

    if(pm->commSize() != 1) {
      // On distribue les nouveaux propriétaires aux processus.
      UniqueArray<Int64> recv_buffer;

      for (Integer rank = 0; rank < nb_rank; ++rank) {
        pm->gatherVariable(get_back_face_owner[rank], recv_buffer, rank);
      }

      Integer index = 0;

      // On explore le tableau reçu contenant les nouveaux propriétaires pour nos faces.
      while (index < recv_buffer.size()) {

        // Le nouveau propriétaire des faces de cette partie du tableau.
        auto rank = static_cast<Integer>(recv_buffer[index++]);

        // Le nombre de uid de faces à suivre dans le tableau.
        auto size = static_cast<Integer>(recv_buffer[index++]);

        // On récupère les uid des faces devant changer de proprio.
        ConstArrayView<Int64> faces_uid = recv_buffer.subView(index, size);
        index += size;

        UniqueArray<Int32> faces_lid(size);
        m_mesh->faceFamily()->itemsUniqueIdToLocalId(faces_lid, faces_uid, true);

        ENUMERATE_ (Face, iface, m_mesh->faceFamily()->view(faces_lid)) {
          Face face = *iface;
          debug() << "Change face owner -- UniqueId : " << face.uniqueId()
                  << " -- Old Owner : " << face.owner()
                  << " -- New Owner : " << rank
          ;
          face.mutableItemBase().setOwner(rank, my_rank);
        }
      }
    }

    m_mesh->faceFamily()->notifyItemsOwnerChanged();
  }

  // Cells
  {
    debug() << "Nb new cells in patch : " << total_nb_cells;

    UniqueArray<Int32> cells_lid(total_nb_cells);
    m_mesh->modifier()->addCells(total_nb_cells, cells_infos, cells_lid);

    // Itération sur les nouvelles mailles.
    CellInfoListView cells(m_mesh->cellFamily());
    for (Integer i = 0; i < total_nb_cells; ++i){
      Cell child = cells[cells_lid[i]];

      child.mutableItemBase().setOwner(parent_cells[i].owner(), my_rank);

      child.mutableItemBase().addFlags(ItemFlags::II_JustAdded);

      if (parent_cells[i].owner() == my_rank) {
        child.mutableItemBase().addFlags(ItemFlags::II_Own);
      }

      if(parent_cells[i].itemBase().flags() & ItemFlags::II_Shared){
        child.mutableItemBase().addFlags(ItemFlags::II_Shared);
      }

      m_mesh->modifier()->addParentCellToCell(child, parent_cells[i]);
      m_mesh->modifier()->addChildCellToCell(parent_cells[i], child);
    }

    // Itération sur les mailles parentes.
    for(Cell cell : cell_to_refine_internals){
      cell.mutableItemBase().removeFlags(ItemFlags::II_Refine);
      cell.mutableItemBase().addFlags(ItemFlags::II_JustRefined | ItemFlags::II_Inactive);
    }
    m_mesh->cellFamily()->notifyItemsOwnerChanged();
  }

  m_mesh->modifier()->endUpdate();

  // On positionne les noeuds dans l'espace.
  for(Cell parent_cell : cell_to_refine_internals){
    m_num_mng->setChildNodeCoordinates(parent_cell);
  }

//  ENUMERATE_(Cell, icell, m_mesh->allCells()){
//    debug() << "\t" << *icell;
//    for(Node node : icell->nodes()){
//      debug() << "\t\t" << node;
//    }
//    for(Face face : icell->faces()){
//      debug() << "\t\t\t" << face;
//    }
//  }
//  info() << "Résumé :";
//  ENUMERATE_ (Cell, icell, m_mesh->allCells()){
//    debug() << "\tCell uniqueId : " << icell->uniqueId() << " -- level : " << icell->level() << " -- nbChildren : " << icell->nbHChildren();
//    for(Integer i = 0; i < icell->nbHChildren(); ++i){
//      debug() << "\t\tChild uniqueId : " << icell->hChild(i).uniqueId() << " -- level : " << icell->hChild(i).level() << " -- nbChildren : " << icell->hChild(i).nbHChildren();
//    }
//  }
}

void CartesianMeshAMRPatchMng::
createLevelDown()
{
  IParallelMng* pm = m_mesh->parallelMng();
  Int32 nb_rank = pm->commSize();
  Int32 my_rank = pm->commRank();

  UniqueArray<Int64> cell_uid_to_create;

  std::unordered_map<Int64, Int32> around_parent_cells_uid_to_owner;
  std::unordered_map<Int64, UniqueArray<Cell>> parent_to_child_cells;

  std::unordered_map<Int64, Int32> node_uid_to_owner;
  std::unordered_map<Int64, Int32> face_uid_to_owner;

  m_num_mng->prepareLevel(-1);

  // CartesianMeshCoarsening2::_doDoubleGhostLayers()
  IMeshModifier* mesh_modifier = m_mesh->modifier();
  IGhostLayerMng* gm = m_mesh->ghostLayerMng();
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
  // CartesianMeshCoarsening2::_doDoubleGhostLayers()

  ENUMERATE_ (Cell, icell, m_mesh->allLevelCells(0)) {
    Cell cell = *icell;

    Int64 parent_uid = m_num_mng->getParentCellUidOfCell(cell);
    info() << "Test : " << cell.uniqueId() << " " << cell.level() << " " << parent_uid << " " << cell.owner();
    if (!cell_uid_to_create.contains(parent_uid)) {
      cell_uid_to_create.add(parent_uid);
      around_parent_cells_uid_to_owner[parent_uid] = cell.owner();
    }
    else {
      if (around_parent_cells_uid_to_owner[parent_uid] != cell.owner()) {
        ARCANE_FATAL("Pb owner");
      }
    }
    parent_to_child_cells[parent_uid].add(cell);
  }

  info() << cell_uid_to_create;
  for (const auto& [key, value] : parent_to_child_cells) {
    info() << "Parent : " << key << " -- Children : " << value;
  }

  UniqueArray<Int64> cells_infos;
  UniqueArray<Int64> faces_infos;
  UniqueArray<Int64> nodes_infos;

  Integer total_nb_cells = 0;
  Integer total_nb_nodes = 0;
  Integer total_nb_faces = 0;

  // Deux tableaux permettant de récupérer les uniqueIds des noeuds et des faces
  // de chaque maille parent à chaque appel à getNodeUids()/getFaceUids().
  UniqueArray<Int64> parent_nodes_uids(m_num_mng->getNbNode());
  UniqueArray<Int64> parent_faces_uids(m_num_mng->getNbFace());

  // Partie échange d'informations sur les mailles autour du patch
  // (pour remplacer les mailles fantômes).
  {
    // Tableau qui contiendra les uids des mailles dont on a besoin des infos.
    UniqueArray<Int64> uid_of_cells_needed;
    {
      UniqueArray<Int64> cell_uids_around((m_mesh->dimension() == 2) ? 9 : 27);
      for (Int64 parent_cell : cell_uid_to_create) {
        m_num_mng->getCellUidsAround(cell_uids_around, parent_cell, -1);
        for (Int64 cell_uid : cell_uids_around) {
          // Si -1 alors il n'y a pas de mailles à cette position.
          if (cell_uid == -1)
            continue;

          // TODO C++20 : Mettre map.contains().
          // SI on a la maille, on n'a pas besoin de demander d'infos.
          if (around_parent_cells_uid_to_owner.find(cell_uid) != around_parent_cells_uid_to_owner.end())
            continue;

          uid_of_cells_needed.add(cell_uid);
        }
      }
    }

    UniqueArray<Int64> uid_of_cells_needed_all_procs;
    pm->allGatherVariable(uid_of_cells_needed, uid_of_cells_needed_all_procs);

    UniqueArray<Int32> owner_of_cells_needed_all_procs(uid_of_cells_needed_all_procs.size());

    {
      for (Integer i = 0; i < uid_of_cells_needed_all_procs.size(); ++i) {
        if (around_parent_cells_uid_to_owner.find(owner_of_cells_needed_all_procs[i]) != around_parent_cells_uid_to_owner.end()) {
          owner_of_cells_needed_all_procs[i] = around_parent_cells_uid_to_owner[owner_of_cells_needed_all_procs[i]];
        }
      }
    }

    pm->reduce(Parallel::eReduceType::ReduceMax, owner_of_cells_needed_all_procs);


    {
      Integer size_uid_of_cells_needed = uid_of_cells_needed.size();
      Integer my_pos_in_all_procs_arrays = 0;
      UniqueArray<Integer> size_uid_of_cells_needed_per_proc(nb_rank);
      ArrayView<Integer> av(1, &size_uid_of_cells_needed);
      pm->allGather(av, size_uid_of_cells_needed_per_proc);

      for (Integer i = 0; i < my_rank; ++i) {
        my_pos_in_all_procs_arrays += size_uid_of_cells_needed_per_proc[i];
      }

      ArrayView<Int32> owner_of_cells_needed = owner_of_cells_needed_all_procs.subView(my_pos_in_all_procs_arrays, size_uid_of_cells_needed);
      for (Integer i = 0; i < size_uid_of_cells_needed; ++i) {
        around_parent_cells_uid_to_owner[uid_of_cells_needed[i]] = owner_of_cells_needed[i];
      }
    }
  }

  if (m_mesh->dimension() == 2) {
    const bool mask_face_if_cell_left[] = { true, true, true, false };
    const bool mask_face_if_cell_bottom[] = { false, true, true, true };

    const bool mask_face_if_cell_right[] = { true, false, true, true };
    const bool mask_face_if_cell_top[] = { true, true, false, true };

    const bool mask_node_if_cell_left[] = { false, true, true, false };
    const bool mask_node_if_cell_bottom[] = { false, false, true, true };

    const bool mask_node_if_cell_right[] = { true, false, false, true };
    const bool mask_node_if_cell_top[] = { true, true, false, false };

    for (Int64 parent_cell_uid : cell_uid_to_create) {

      UniqueArray<Int64> cells_uid_around(9);
      m_num_mng->getCellUidsAround(cells_uid_around, parent_cell_uid, -1);

      UniqueArray<Int32> owner_cells_around_parent_cell_1d(9);
      ConstArray2View owner_cells_around_parent_cell(owner_cells_around_parent_cell_1d.data(), 3, 3);

      for (Integer i = 0; i < 9; ++i) {
        Int64 uid_cell = cells_uid_around[i];
        // Si uid_cell != -1 alors il y a peut-être une maille (mais on ne sait pas si elle est bien présente).
        // Si around_parent_cells_uid_to_owner[uid_cell] != -1 alors il y a bien une maille.
        if (uid_cell != -1 && around_parent_cells_uid_to_owner[uid_cell] != -1) {
          owner_cells_around_parent_cell_1d[i] = around_parent_cells_uid_to_owner[uid_cell];
        }
        else {
          cells_uid_around[i] = -1;
          owner_cells_around_parent_cell_1d[i] = -1;
        }
      }

      info() << cells_uid_around;
      info() << owner_cells_around_parent_cell_1d;

      auto is_cell_around_parent_cell_different_owner = [&](Integer y, Integer x) {
        return (owner_cells_around_parent_cell(y, x) != owner_cells_around_parent_cell(1, 1));
      };

      auto is_cell_around_parent_cell_exist_and_different_owner = [&](Integer y, Integer x) {
        return (owner_cells_around_parent_cell(y, x) != -1 && (owner_cells_around_parent_cell(y, x) != owner_cells_around_parent_cell(1, 1)));
      };

      total_nb_cells++;
      debug() << "Parent"
              << " -- x : " << m_num_mng->uidToCoordX(parent_cell_uid, -1)
              << " -- y : " << m_num_mng->uidToCoordY(parent_cell_uid, -1)
              << " -- level : " << -1
              << " -- uid : " << parent_cell_uid;

      m_num_mng->getNodeUids(parent_nodes_uids, -1, parent_cell_uid);
      m_num_mng->getFaceUids(parent_faces_uids, -1, parent_cell_uid);

      const Integer type_cell = IT_Quad4;
      const Integer type_face = IT_Line2;

      // Partie Cell.
      cells_infos.add(type_cell);
      cells_infos.add(parent_cell_uid);
      for (Integer nc = 0; nc < m_num_mng->getNbNode(); nc++) {
        cells_infos.add(parent_nodes_uids[nc]);
      }

      // Partie Face.
      for (Integer l = 0; l < m_num_mng->getNbFace(); ++l) {
        Integer parent_face_owner = -1;

        if (
        (mask_face_if_cell_left[l] || is_cell_around_parent_cell_different_owner(1, 0)) &&
        (mask_face_if_cell_bottom[l] || is_cell_around_parent_cell_different_owner(0, 1))) {
          faces_infos.add(type_face);
          faces_infos.add(parent_faces_uids[l]);

          // Les noeuds de la face sont toujours les noeuds l et l+1
          // car on utilise la même exploration pour les deux cas.
          for (Integer nc = l; nc < l + 2; nc++) {
            faces_infos.add(parent_nodes_uids[nc % m_num_mng->getNbNode()]);
          }
          total_nb_faces++;

          if ((!mask_face_if_cell_left[l]) && is_cell_around_parent_cell_exist_and_different_owner(1, 0)) {
            parent_face_owner = owner_cells_around_parent_cell(1, 0);
          }
          else if ((!mask_face_if_cell_bottom[l]) && is_cell_around_parent_cell_exist_and_different_owner(0, 1)) {
            parent_face_owner = owner_cells_around_parent_cell(0, 1);
          }
          else {
            parent_face_owner = owner_cells_around_parent_cell(1, 1);
          }
          face_uid_to_owner[parent_faces_uids[l]] = parent_face_owner;
          debug() << "Parent face (create face)  -- parent_cell_uid : " << parent_cell_uid
                  << " -- level : " << -1
                  << " -- face : " << l
                  << " -- uid_face : " << parent_faces_uids[l]
                  << " -- owner : " << parent_face_owner;
        }
      }

      // Partie Node.
      // Cette partie est assez ressemblante à la partie face, mis à part le fait qu'il peut y avoir
      // plus de propriétaires possibles.
      for (Integer l = 0; l < m_num_mng->getNbNode(); ++l) {
        Integer parent_node_owner = -1;
        if (
        (mask_node_if_cell_left[l] || is_cell_around_parent_cell_different_owner(1, 0)) &&
        (mask_node_if_cell_bottom[l] || is_cell_around_parent_cell_different_owner(0, 1))) {
          nodes_infos.add(parent_nodes_uids[l]);
          total_nb_nodes++;

          if ((!mask_node_if_cell_left[l])) {
            if ((!mask_node_if_cell_bottom[l])) {
              if (is_cell_around_parent_cell_exist_and_different_owner(0, 0)) {
                parent_node_owner = owner_cells_around_parent_cell(0, 0);
              }
              else if (is_cell_around_parent_cell_exist_and_different_owner(0, 1)) {
                parent_node_owner = owner_cells_around_parent_cell(0, 1);
              }
              else if (is_cell_around_parent_cell_exist_and_different_owner(1, 0)) {
                parent_node_owner = owner_cells_around_parent_cell(1, 0);
              }
              else {
                parent_node_owner = owner_cells_around_parent_cell(1, 1);
              }
            }
            else {
              if (is_cell_around_parent_cell_exist_and_different_owner(1, 0)) {
                parent_node_owner = owner_cells_around_parent_cell(1, 0);
              }
              else {
                parent_node_owner = owner_cells_around_parent_cell(1, 1);
              }
            }
          }
          else {
            if ((!mask_node_if_cell_bottom[l])) {
              if (is_cell_around_parent_cell_exist_and_different_owner(0, 1)) {
                parent_node_owner = owner_cells_around_parent_cell(0, 1);
              }
              else if (is_cell_around_parent_cell_exist_and_different_owner(0, 2)) {
                parent_node_owner = owner_cells_around_parent_cell(0, 2);
              }
              else {
                parent_node_owner = owner_cells_around_parent_cell(1, 1);
              }
            }
            else {
              parent_node_owner = owner_cells_around_parent_cell(1, 1);
            }
          }

          node_uid_to_owner[parent_nodes_uids[l]] = parent_node_owner;
          debug() << "Child node (create node)  -- parent_cell_uid : " << parent_cell_uid
                  << " -- level : " << -1
                  << " -- node : " << l
                  << " -- uid_node : " << parent_nodes_uids[l]
                  << " -- owner : " << parent_node_owner;
        }
      }
    }
  }
  else if (m_mesh->dimension() == 3) {
    const bool mask_node_if_cell_left[] = { false, true, true, false, false, true, true, false };
    const bool mask_node_if_cell_bottom[] = { false, false, true, true, false, false, true, true };
    const bool mask_node_if_cell_rear[] = { false, false, false, false, true, true, true, true };

    const bool mask_node_if_cell_right[] = { true, false, false, true, true, false, false, true };
    const bool mask_node_if_cell_top[] = { true, true, false, false, true, true, false, false };
    const bool mask_node_if_cell_front[] = { true, true, true, true, false, false, false, false };

    const bool mask_face_if_cell_left[] = { true, false, true, true, true, true };
    const bool mask_face_if_cell_bottom[] = { true, true, false, true, true, true };
    const bool mask_face_if_cell_rear[] = { false, true, true, true, true, true };

    const bool mask_face_if_cell_right[] = { true, true, true, true, false, true };
    const bool mask_face_if_cell_top[] = { true, true, true, true, true, false };
    const bool mask_face_if_cell_front[] = { true, true, true, false, true, true };

    // Petite différence par rapport au 2D. Pour le 2D, la position des noeuds des faces
    // dans le tableau "child_nodes_uids" est toujours pareil (l et l+1, voir le 2D).
    // Pour le 3D, ce n'est pas le cas donc on a des tableaux pour avoir une correspondance
    // entre les noeuds de chaque face et la position des noeuds dans le tableau "child_nodes_uids".
    // (Exemple : pour la face 1 (même ordre d'énumération qu'Arcane), on doit prendre le
    // tableau "nodes_in_face_1" et donc les noeuds "child_nodes_uids[0]", "child_nodes_uids[3]",
    // "child_nodes_uids[7]" et "child_nodes_uids[4]").
    const Integer nodes_in_face_0[] = { 0, 1, 2, 3 };
    const Integer nodes_in_face_1[] = { 0, 3, 7, 4 };
    const Integer nodes_in_face_2[] = { 0, 1, 5, 4 };
    const Integer nodes_in_face_3[] = { 4, 5, 6, 7 };
    const Integer nodes_in_face_4[] = { 1, 2, 6, 5 };
    const Integer nodes_in_face_5[] = { 3, 2, 6, 7 };

    const Integer nb_nodes_in_face = 4;

    for (Int64 parent_cell_uid : cell_uid_to_create) {
      UniqueArray<Int64> cells_uid_around(27);
      m_num_mng->getCellUidsAround(cells_uid_around, parent_cell_uid, -1);

      UniqueArray<Int32> owner_cells_around_parent_cell_1d(27);
      ConstArray3View owner_cells_around_parent_cell(owner_cells_around_parent_cell_1d.data(), 3, 3, 3);

      for (Integer i = 0; i < 27; ++i) {
        Int64 uid_cell = cells_uid_around[i];
        // Si uid_cell != -1 alors il y a peut-être une maille (mais on ne sait pas si elle est bien présente).
        // Si around_parent_cells_uid_to_owner[uid_cell] != -1 alors il y a bien une maille.
        if (uid_cell != -1 && around_parent_cells_uid_to_owner[uid_cell] != -1) {
          owner_cells_around_parent_cell_1d[i] = around_parent_cells_uid_to_owner[uid_cell];
        }
        else {
          cells_uid_around[i] = -1;
          owner_cells_around_parent_cell_1d[i] = -1;
        }
      }

      info() << cells_uid_around;
      info() << owner_cells_around_parent_cell_1d;

      auto is_cell_around_parent_cell_different_owner = [&](Integer z, Integer y, Integer x) {
        return (owner_cells_around_parent_cell(z, y, x) != owner_cells_around_parent_cell(1, 1, 1));
      };

      auto is_cell_around_parent_cell_exist_and_different_owner = [&](Integer z, Integer y, Integer x) {
        return (owner_cells_around_parent_cell(z, y, x) != -1 && (owner_cells_around_parent_cell(z, y, x) != owner_cells_around_parent_cell(1, 1, 1)));
      };

      total_nb_cells++;
      debug() << "Parent"
              << " -- x : " << m_num_mng->uidToCoordX(parent_cell_uid, -1)
              << " -- y : " << m_num_mng->uidToCoordY(parent_cell_uid, -1)
              << " -- z : " << m_num_mng->uidToCoordZ(parent_cell_uid, -1)
              << " -- level : " << -1
              << " -- uid : " << parent_cell_uid;

      m_num_mng->getNodeUids(parent_nodes_uids, -1, parent_cell_uid);
      m_num_mng->getFaceUids(parent_faces_uids, -1, parent_cell_uid);

      const Integer type_cell = IT_Hexaedron8;
      const Integer type_face = IT_Quad4;

      // Partie Cell.
      cells_infos.add(type_cell);
      cells_infos.add(parent_cell_uid);
      for (Integer nc = 0; nc < m_num_mng->getNbNode(); nc++) {
        cells_infos.add(parent_nodes_uids[nc]);
      }

      // Partie Face.
      for (Integer l = 0; l < m_num_mng->getNbFace(); ++l) {
        Integer parent_face_owner = -1;

        if (
        (mask_face_if_cell_left[l] || is_cell_around_parent_cell_different_owner(1, 1, 0)) &&
        (mask_face_if_cell_bottom[l] || is_cell_around_parent_cell_different_owner(1, 0, 1)) &&
        (mask_face_if_cell_rear[l] || is_cell_around_parent_cell_different_owner(0, 1, 1))) {
          faces_infos.add(type_face);
          faces_infos.add(parent_faces_uids[l]);

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
            faces_infos.add(parent_nodes_uids[nc]);
          }
          total_nb_faces++;

          if ((!mask_face_if_cell_left[l]) && is_cell_around_parent_cell_exist_and_different_owner(1, 1, 0)) {
            parent_face_owner = owner_cells_around_parent_cell(1, 1, 0);
          }
          else if ((!mask_face_if_cell_bottom[l]) && is_cell_around_parent_cell_exist_and_different_owner(1, 0, 1)) {
            parent_face_owner = owner_cells_around_parent_cell(1, 0, 1);
          }
          else if ((!mask_face_if_cell_rear[l]) && is_cell_around_parent_cell_exist_and_different_owner(0, 1, 1)) {
            parent_face_owner = owner_cells_around_parent_cell(0, 1, 1);
          }
          else {
            parent_face_owner = owner_cells_around_parent_cell(1, 1, 1);
          }
          face_uid_to_owner[parent_faces_uids[l]] = parent_face_owner;
          debug() << "Parent face (create face)  -- parent_cell_uid : " << parent_cell_uid
                  << " -- level : " << -1
                  << " -- face : " << l
                  << " -- uid_face : " << parent_faces_uids[l]
                  << " -- owner : " << parent_face_owner;
        }
      }

      // Partie Node.
      // Cette partie est assez ressemblante à la partie face, mis à part le fait qu'il peut y avoir
      // plus de propriétaires possibles.
      for (Integer l = 0; l < m_num_mng->getNbNode(); ++l) {
        Integer parent_node_owner = -1;
        if (
        (mask_node_if_cell_left[l] || is_cell_around_parent_cell_different_owner(1, 1, 0)) &&
        (mask_node_if_cell_bottom[l] || is_cell_around_parent_cell_different_owner(1, 0, 1)) &&
        (mask_node_if_cell_rear[l] || is_cell_around_parent_cell_different_owner(0, 1, 1))) {
          nodes_infos.add(parent_nodes_uids[l]);
          total_nb_nodes++;

          if ((!mask_node_if_cell_left[l])) {
            if ((!mask_node_if_cell_bottom[l])) {
              if ((!mask_node_if_cell_rear[l])) {

                if (is_cell_around_parent_cell_exist_and_different_owner(0, 0, 0)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 0, 0);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(0, 0, 1)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 0, 1);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(0, 1, 0)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 1, 0);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(0, 1, 1)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 1, 1);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(1, 0, 0)) {
                  parent_node_owner = owner_cells_around_parent_cell(1, 0, 0);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(1, 0, 1)) {
                  parent_node_owner = owner_cells_around_parent_cell(1, 0, 1);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(1, 1, 0)) {
                  parent_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                }
                else {
                  parent_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                }
              }
              else {
                if (is_cell_around_parent_cell_exist_and_different_owner(1, 0, 0)) {
                  parent_node_owner = owner_cells_around_parent_cell(1, 0, 0);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(1, 0, 1)) {
                  parent_node_owner = owner_cells_around_parent_cell(1, 0, 1);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(1, 1, 0)) {
                  parent_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                }
                else {
                  parent_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                }
              }
            }
            else {
              if ((!mask_node_if_cell_rear[l])) {
                if (is_cell_around_parent_cell_exist_and_different_owner(0, 1, 0)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 1, 0);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(0, 1, 1)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 1, 1);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(0, 2, 0)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 2, 0);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(0, 2, 1)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 2, 1);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(1, 1, 0)) {
                  parent_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                }
                else {
                  parent_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                }
              }
              else {
                if (is_cell_around_parent_cell_exist_and_different_owner(1, 1, 0)) {
                  parent_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                }
                else {
                  parent_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                }
              }
            }
          }
          else {
            if ((!mask_node_if_cell_bottom[l])) {
              if ((!mask_node_if_cell_rear[l])) {
                if (is_cell_around_parent_cell_exist_and_different_owner(0, 0, 1)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 0, 1);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(0, 0, 2)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 0, 2);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(0, 1, 1)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 1, 1);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(0, 1, 2)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 1, 2);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(1, 0, 1)) {
                  parent_node_owner = owner_cells_around_parent_cell(1, 0, 1);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(1, 0, 2)) {
                  parent_node_owner = owner_cells_around_parent_cell(1, 0, 2);
                }
                else {
                  parent_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                }
              }
              else {
                if (is_cell_around_parent_cell_exist_and_different_owner(1, 0, 1)) {
                  parent_node_owner = owner_cells_around_parent_cell(1, 0, 1);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(1, 0, 2)) {
                  parent_node_owner = owner_cells_around_parent_cell(1, 0, 2);
                }
                else {
                  parent_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                }
              }
            }
            else {
              if ((!mask_node_if_cell_rear[l])) {
                if (is_cell_around_parent_cell_exist_and_different_owner(0, 1, 1)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 1, 1);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(0, 1, 2)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 1, 2);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(0, 2, 1)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 2, 1);
                }
                else if (is_cell_around_parent_cell_exist_and_different_owner(0, 2, 2)) {
                  parent_node_owner = owner_cells_around_parent_cell(0, 2, 2);
                }
                else {
                  parent_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                }
              }
              else {
                parent_node_owner = owner_cells_around_parent_cell(1, 1, 1);
              }
            }
          }

          node_uid_to_owner[parent_nodes_uids[l]] = parent_node_owner;
          debug() << "Child node (create node)  -- parent_cell_uid : " << parent_cell_uid
                  << " -- level : " << -1
                  << " -- node : " << l
                  << " -- uid_node : " << parent_nodes_uids[l]
                  << " -- owner : " << parent_node_owner;
        }
      }
    }
  }
  else {
    ARCANE_FATAL("Bad dimension");
  }

  // Nodes
  {
    debug() << "Nb new nodes in patch : " << total_nb_nodes;
    {
      UniqueArray<Int32> nodes_lid(total_nb_nodes);

      // On crée les noeuds. On met les localIds des nouveaux noeuds au début du tableau.
      m_mesh->modifier()->addNodes(nodes_infos, nodes_lid);

      // On attribue les bons propriétaires aux noeuds.
      ENUMERATE_ (Node, inode, m_mesh->nodeFamily()->view(nodes_lid)) {
        Node node = *inode;
        node.mutableItemBase().setOwner(node_uid_to_owner[node.uniqueId()], my_rank);

        if (node_uid_to_owner[node.uniqueId()] == my_rank) {
          node.mutableItemBase().addFlags(ItemFlags::II_Own);
        }
      }
    }

    m_mesh->nodeFamily()->notifyItemsOwnerChanged();
  }

  // Faces
  {
    debug() << "Nb new faces in patch : " << total_nb_faces;
    {
      UniqueArray<Int32> faces_lid(total_nb_faces);

      m_mesh->modifier()->addFaces(total_nb_faces, faces_infos, faces_lid);

      // On attribue les bons propriétaires aux faces.
      ENUMERATE_ (Face, iface, m_mesh->faceFamily()->view(faces_lid)) {
        Face face = *iface;
        face.mutableItemBase().setOwner(face_uid_to_owner[face.uniqueId()], my_rank);

        if (face_uid_to_owner[face.uniqueId()] == my_rank) {
          face.mutableItemBase().addFlags(ItemFlags::II_Own);
        }
      }
    }

    m_mesh->faceFamily()->notifyItemsOwnerChanged();
  }

  // Cells
  UniqueArray<Int32> cells_lid(total_nb_cells);
  {
    debug() << "Nb new cells in patch : " << total_nb_cells;

    m_mesh->modifier()->addCells(total_nb_cells, cells_infos, cells_lid);

    // Itération sur les nouvelles mailles.
    CellInfoListView cells(m_mesh->cellFamily());
    for (Integer i = 0; i < total_nb_cells; ++i) {
      Cell parent = cells[cells_lid[i]];

      parent.mutableItemBase().setOwner(around_parent_cells_uid_to_owner[parent.uniqueId()], my_rank);

      parent.mutableItemBase().addFlags(ItemFlags::II_JustAdded);
      parent.mutableItemBase().addFlags(ItemFlags::II_JustCoarsened);
      parent.mutableItemBase().addFlags(ItemFlags::II_Inactive);

      if (around_parent_cells_uid_to_owner[parent.uniqueId()] == my_rank) {
        parent.mutableItemBase().addFlags(ItemFlags::II_Own);
      }
      else {
        parent.mutableItemBase().addFlags(ItemFlags::II_Shared);
      }

      for (Cell child : parent_to_child_cells[parent.uniqueId()]) {
        m_mesh->modifier()->addParentCellToCell(child, parent);
        m_mesh->modifier()->addChildCellToCell(parent, child);
      }
    }
    m_mesh->cellFamily()->notifyItemsOwnerChanged();
  }

  m_mesh->modifier()->endUpdate();
  m_num_mng->updateFirstLevel();

  ENUMERATE_ (Cell, icell, m_mesh->allCells()) {
    info() << *icell << icell->level();
  }

  // On positionne les noeuds dans l'espace.
  info() << "cells_lid : " << cells_lid.size();
  CellInfoListView cells(m_mesh->cellFamily());
  for (Integer i = 0; i < total_nb_cells; ++i) {
    Cell parent = cells[cells_lid[i]];
    m_num_mng->setParentNodeCoordinates(parent);
  }

  ARCANE_FATAL("Normal");
}

void CartesianMeshAMRPatchMng::
coarse()
{
  IParallelMng* pm = m_mesh->parallelMng();
  Int32 nb_rank = pm->commSize();
  Int32 my_rank = pm->commRank();
  Int32 min_level = 1;

  UniqueArray<Cell> cell_to_coarse_internals;
  ENUMERATE_ (Cell, icell, m_mesh->allActiveCells()) {
    Cell cell = *icell;
    if (cell.itemBase().flags() & ItemFlags::II_Coarsen) {
      cell_to_coarse_internals.add(cell);
      if (min_level == 1) {
        min_level = cell.level();
      }
      if (cell.level() != min_level)
        ARCANE_FATAL("Different levels not supported");
    }
  }

  if (min_level == 0) {
    createLevelDown();
  }

  //...
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
