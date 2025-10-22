// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshAMRPatchMng.cc                                 (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire de l'AMR par patch d'un maillage cartésien.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <arcane/cartesianmesh/CartesianMeshAMRPatchMng.h>

#include "arcane/cartesianmesh/CellDirectionMng.h"
#include "arcane/cartesianmesh/CartesianMeshNumberingMng.h"
#include "arcane/cartesianmesh/internal/ICartesianMeshInternal.h"

#include "arcane/utils/Array2View.h"
#include "arcane/utils/Array3View.h"
#include "arcane/utils/FixedArray.h"
#include "arcane/utils/Vector2.h"

#include "arcane/core/IGhostLayerMng.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/materials/IMeshMaterialMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshAMRPatchMng::
CartesianMeshAMRPatchMng(ICartesianMesh* cmesh, ICartesianMeshNumberingMng* numbering_mng)
: TraceAccessor(cmesh->mesh()->traceMng())
, m_mesh(cmesh->mesh())
, m_cmesh(cmesh)
, m_num_mng(numbering_mng)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAMRPatchMng::
flagCellToRefine(const Int32ConstArrayView cells_lids, const bool clear_old_flags)
{
  if (clear_old_flags) {
    constexpr ItemFlags::FlagType flags_to_remove = (
      ItemFlags::II_Coarsen | ItemFlags::II_Refine |
      ItemFlags::II_JustCoarsened | ItemFlags::II_JustRefined |
      ItemFlags::II_JustAdded | ItemFlags::II_CoarsenInactive
      );
    ENUMERATE_(Cell, icell, m_mesh->allCells()){
      icell->mutableItemBase().removeFlags(flags_to_remove);
    }
  }

  const ItemInfoListView cells(m_mesh->cellFamily());
  for (const int lid : cells_lids) {
    Item item = cells[lid];
    item.mutableItemBase().addFlags(ItemFlags::II_Refine);
  }
  _syncFlagCell();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAMRPatchMng::
flagCellToCoarsen(const Int32ConstArrayView cells_lids, const bool clear_old_flags)
{
  if (clear_old_flags) {
    constexpr ItemFlags::FlagType flags_to_remove = (
      ItemFlags::II_Coarsen | ItemFlags::II_Refine |
      ItemFlags::II_JustCoarsened | ItemFlags::II_JustRefined |
      ItemFlags::II_JustAdded | ItemFlags::II_CoarsenInactive
      );
    ENUMERATE_(Cell, icell, m_mesh->allCells()){
      icell->mutableItemBase().removeFlags(flags_to_remove);
    }
  }

  const ItemInfoListView cells(m_mesh->cellFamily());
  for (const Integer lid : cells_lids) {
    Item item = cells[lid];
    item.mutableItemBase().addFlags(ItemFlags::II_Coarsen);
  }
  _syncFlagCell();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAMRPatchMng::
_syncFlagCell() const
{
  if (!m_mesh->parallelMng()->isParallel())
    return;

  VariableCellInteger flag_cells_consistent(VariableBuildInfo(m_mesh, "FlagCellsConsistent"));
  ENUMERATE_ (Cell, icell, m_mesh->ownCells()) {
    Cell cell = *icell;
    flag_cells_consistent[cell] = cell.mutableItemBase().flags();
    //    debug() << "Send " << cell
    //            << " -- flag : " << cell.mutableItemBase().flags()
    //            << " -- II_Refine : " << (cell.itemBase().flags() & ItemFlags::II_Refine)
    //            << " -- II_Inactive : " << (cell.itemBase().flags() & ItemFlags::II_Inactive)
    //    ;
  }

  flag_cells_consistent.synchronize();

  ENUMERATE_ (Cell, icell, m_mesh->allCells().ghost()) {
    Cell cell = *icell;

    // On ajoute uniquement les flags qui nous interesse (pour éviter d'ajouter le flag "II_Own" par exemple).
    // On utilise set au lieu de add puisqu'une maille ne peut être à la fois II_Refine et II_Inactive.
    if (flag_cells_consistent[cell] & ItemFlags::II_Refine) {
      cell.mutableItemBase().setFlags(ItemFlags::II_Refine);
    }
    if (flag_cells_consistent[cell] & ItemFlags::II_Inactive) {
      cell.mutableItemBase().setFlags(ItemFlags::II_Inactive);
    }
    if (flag_cells_consistent[cell] & ItemFlags::II_Coarsen) {
      cell.mutableItemBase().setFlags(ItemFlags::II_Coarsen);
    }

    //    debug() << "After Compute " << cell
    //            << " -- flag : " << cell.mutableItemBase().flags()
    //            << " -- II_Refine : " << (cell.itemBase().flags() & ItemFlags::II_Refine)
    //            << " -- II_Inactive : " << (cell.itemBase().flags() & ItemFlags::II_Inactive)
    //    ;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * Pour les commentaires de cette méthode, on considère le repère suivant :
 *       (top)
 *         y    (front)
 *         ^    z
 *         |   /
 *         | /
 *  (left) ------->x (right)
 *  (rear)(bottom)
 */
void CartesianMeshAMRPatchMng::
refine()
{
  IParallelMng* pm = m_mesh->parallelMng();
  Int32 my_rank = pm->commRank();
  Int32 max_level = 0;

  UniqueArray<Cell> cell_to_refine_internals;
  ENUMERATE_ (Cell, icell, m_mesh->allActiveCells()) {
    Cell cell = *icell;
    if (cell.itemBase().flags() & ItemFlags::II_Refine) {
      cell_to_refine_internals.add(cell);
      if (cell.level() > max_level)
        max_level = cell.level();
    }
  }
  m_num_mng->prepareLevel(max_level + 1);

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

  // Deux tableaux permettant de récupérer les uniqueIds des noeuds et des faces
  // de chaque maille enfant à chaque appel à getNodeUids()/getFaceUids().
  UniqueArray<Int64> child_nodes_uids(m_num_mng->nbNodeByCell());
  UniqueArray<Int64> child_faces_uids(m_num_mng->nbFaceByCell());

  // On doit enregistrer les mailles parentes de chaque maille enfant pour mettre à jour les connectivités
  // lors de la création des mailles.
  UniqueArray<Cell> parent_cells;

  // Maps remplaçant les mailles fantômes.
  std::unordered_map<Int64, Integer> around_parent_cells_uid_to_owner;
  std::unordered_map<Int64, Int32> around_parent_cells_uid_to_flags;

  {
    // On a uniquement besoin de ses deux flags pour les mailles autour.
    // (II_Refine pour savoir si les mailles autour sont dans le même patch)
    // (II_Inactive pour savoir si les mailles autour sont déjà raffinées)
    Int32 useful_flags = ItemFlags::II_Refine + ItemFlags::II_Inactive;
    _shareInfosOfCellsAroundPatch(cell_to_refine_internals, around_parent_cells_uid_to_owner, around_parent_cells_uid_to_flags, useful_flags);
  }

  if (m_mesh->dimension() == 2) {

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
    constexpr bool mask_node_if_cell_left[] = { false, true, true, false };
    constexpr bool mask_node_if_cell_bottom[] = { false, false, true, true };

    constexpr bool mask_node_if_cell_right[] = { true, false, false, true };
    constexpr bool mask_node_if_cell_top[] = { true, true, false, false };

    constexpr bool mask_face_if_cell_left[] = { true, true, true, false };
    constexpr bool mask_face_if_cell_bottom[] = { false, true, true, true };

    constexpr bool mask_face_if_cell_right[] = { true, false, true, true };
    constexpr bool mask_face_if_cell_top[] = { true, true, false, true };

    // Pour la taille :
    // - on a "cell_to_refine_internals.size() * 4" mailles enfants,
    // - pour chaque maille, on a 2 infos (type de maille et uniqueId de la maille)
    // - pour chaque maille, on a "m_num_mng->getNbNode()" uniqueIds (les uniqueId de chaque noeud de la maille).
    cells_infos.reserve((cell_to_refine_internals.size() * 4) * (2 + m_num_mng->nbNodeByCell()));

    // Pour la taille, au maximum :
    // - on a "cell_to_refine_internals.size() * 12" faces
    // - pour chaque face, on a 2 infos (type de face et uniqueId de la face)
    // - pour chaque face, on a 2 uniqueIds de noeuds.
    faces_infos.reserve((cell_to_refine_internals.size() * 12) * (2 + 2));

    // Pour la taille, au maximum :
    // - on a (cell_to_refine_internals.size() * 9) uniqueIds de noeuds.
    nodes_infos.reserve(cell_to_refine_internals.size() * 9);

    FixedArray<Int64, 9> uid_cells_around_parent_cell_1d;
    FixedArray<Int32, 9> owner_cells_around_parent_cell_1d;
    FixedArray<Int32, 9> flags_cells_around_parent_cell_1d;

    for (Cell parent_cell : cell_to_refine_internals) {
      const Int64 parent_cell_uid = parent_cell.uniqueId();
      const Int32 parent_cell_level = parent_cell.level();
      const bool parent_cell_is_own = (parent_cell.owner() == my_rank);

      const Int64 parent_coord_x = m_num_mng->cellUniqueIdToCoordX(parent_cell_uid, parent_cell_level);
      const Int64 parent_coord_y = m_num_mng->cellUniqueIdToCoordY(parent_cell_uid, parent_cell_level);

      const Int64 child_coord_x = m_num_mng->offsetLevelToLevel(parent_coord_x, parent_cell_level, parent_cell_level + 1);
      const Int64 child_coord_y = m_num_mng->offsetLevelToLevel(parent_coord_y, parent_cell_level, parent_cell_level + 1);

      const Integer pattern = m_num_mng->pattern();

      m_num_mng->cellUniqueIdsAroundCell(uid_cells_around_parent_cell_1d.view(), parent_cell);

      for (Integer i = 0; i < 9; ++i) {
        Int64 uid_cell = uid_cells_around_parent_cell_1d[i];
        // Si uid_cell != -1 alors il y a peut-être une maille (mais on ne sait pas si elle est bien présente).
        // Si around_parent_cells_uid_to_owner[uid_cell] != -1 alors il y a bien une maille.
        if (uid_cell != -1 && around_parent_cells_uid_to_owner[uid_cell] != -1) {
          owner_cells_around_parent_cell_1d[i] = around_parent_cells_uid_to_owner[uid_cell];
          flags_cells_around_parent_cell_1d[i] = around_parent_cells_uid_to_flags[uid_cell];
        }
        else {
          uid_cells_around_parent_cell_1d[i] = -1;
          owner_cells_around_parent_cell_1d[i] = -1;
          flags_cells_around_parent_cell_1d[i] = 0;
        }
      }

      // Pour simplifier, on utilise des vues 2D. (array[Y][X]).
      ConstArray2View uid_cells_around_parent_cell(uid_cells_around_parent_cell_1d.data(), 3, 3);
      ConstArray2View owner_cells_around_parent_cell(owner_cells_around_parent_cell_1d.data(), 3, 3);
      ConstArray2View flags_cells_around_parent_cell(flags_cells_around_parent_cell_1d.data(), 3, 3);

      // #priority_owner_2d
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
      //
      // ^y
      // |
      // ->x

      // #arcane_order_to_around_2d
      // Note pour les maillages cartésiens 2D :
      // Les itérateurs sur les faces itèrent dans l'ordre (pour la maille 4 ici) :
      //  1. Face entre [4, 1],
      //  2. Face entre [4, 5],
      //  3. Face entre [4, 7],
      //  4. Face entre [4, 3],
      //
      // Les itérateurs sur les noeuds itèrent dans l'ordre (pour la maille 4 ici) :
      //  1. Noeud entre [4, 0]
      //  2. Noeud entre [4, 2]
      //  3. Noeud entre [4, 8]
      //  4. Noeud entre [4, 6]

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
      FixedArray<FixedArray<bool, 3>, 3> is_cell_around_parent_cell_present_and_useful;

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
      auto is_cell_around_parent_cell_in_subdomain = [&](const Integer y, const Integer x) {
        return is_cell_around_parent_cell_present_and_useful[y][x] && (flags_cells_around_parent_cell(y, x) & ItemFlags::II_UserMark1);
      };

      // ... on regarde si chaque maille parent est possédé par le même propriétaire que notre maille parent.
      auto is_cell_around_parent_cell_same_owner = [&](const Integer y, const Integer x) {
        return is_cell_around_parent_cell_present_and_useful[y][x] && (owner_cells_around_parent_cell(y, x) == owner_cells_around_parent_cell(1, 1));
      };

      // ... on regarde si chaque maille parent a un propriétaire différent par rapport à notre maille parent.
      auto is_cell_around_parent_cell_different_owner = [&](const Integer y, const Integer x) {
        return is_cell_around_parent_cell_present_and_useful[y][x] && (owner_cells_around_parent_cell(y, x) != owner_cells_around_parent_cell(1, 1));
      };

      // On itère sur toutes les mailles enfants.
      for (Int64 j = child_coord_y; j < child_coord_y + pattern; ++j) {
        for (Int64 i = child_coord_x; i < child_coord_x + pattern; ++i) {
          parent_cells.add(parent_cell);
          total_nb_cells++;

          const Int64 child_cell_uid = m_num_mng->cellUniqueId(parent_cell_level + 1, Int64x2(i, j));
          // debug() << "Child -- x : " << i << " -- y : " << j << " -- level : " << parent_cell_level + 1 << " -- uid : " << child_cell_uid;

          m_num_mng->cellNodeUniqueIds(child_nodes_uids, parent_cell_level + 1, Int64x2(i, j));
          m_num_mng->cellFaceUniqueIds(child_faces_uids, parent_cell_level + 1, Int64x2(i, j));

          constexpr Integer type_cell = IT_Quad4;
          constexpr Integer type_face = IT_Line2;

          // Partie Cell.
          cells_infos.add(type_cell);
          cells_infos.add(child_cell_uid);
          for (Integer nc = 0; nc < m_num_mng->nbNodeByCell(); nc++) {
            cells_infos.add(child_nodes_uids[nc]);
          }

          // Partie Face.
          for (Integer l = 0; l < m_num_mng->nbFaceByCell(); ++l) {
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
            ((i == child_coord_x && !is_cell_around_parent_cell_in_subdomain(1, 0)) || (mask_face_if_cell_left[l])) &&
            ((i != (child_coord_x + pattern - 1) || !is_cell_around_parent_cell_in_subdomain(1, 2)) || mask_face_if_cell_right[l]) &&
            ((j == child_coord_y && !is_cell_around_parent_cell_in_subdomain(0, 1)) || (mask_face_if_cell_bottom[l])) &&
            ((j != (child_coord_y + pattern - 1) || !is_cell_around_parent_cell_in_subdomain(2, 1)) || mask_face_if_cell_top[l])) {
              is_new_face = true;
              faces_infos.add(type_face);
              faces_infos.add(child_faces_uids[l]);

              // Les noeuds de la face sont toujours les noeuds l et l+1
              // car on utilise la même exploration pour les deux cas.
              for (Integer nc = l; nc < l + 2; nc++) {
                faces_infos.add(child_nodes_uids[nc % m_num_mng->nbNodeByCell()]);
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
            ((i == child_coord_x && !is_cell_around_parent_cell_same_owner(1, 0)) || (mask_face_if_cell_left[l])) &&
            ((i != (child_coord_x + pattern - 1) || !is_cell_around_parent_cell_same_owner(1, 2)) || mask_face_if_cell_right[l]) &&
            ((j == child_coord_y && !is_cell_around_parent_cell_same_owner(0, 1)) || (mask_face_if_cell_bottom[l])) &&
            ((j != (child_coord_y + pattern - 1) || !is_cell_around_parent_cell_same_owner(2, 1)) || mask_face_if_cell_top[l])) {
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

                // Sinon, c'est une face interne donc au parent_cell.
                child_face_owner = owner_cells_around_parent_cell(1, 1);
              }
            }

            // S'il y a une création de face et/ou un changement de propriétaire.
            if (child_face_owner != -1) {
              face_uid_to_owner[child_faces_uids[l]] = child_face_owner;

              // Lorsqu'il y a un changement de propriétaire sans création de face,
              // on doit mettre de côté les uniqueIds de ces faces pour pouvoir
              // itérer dessus par la suite.
              if (!is_new_face) {
                face_uid_change_owner_only.add(child_faces_uids[l]);
                // debug() << "Child face (change owner) -- x : " << i
                //         << " -- y : " << j
                //         << " -- level : " << parent_cell_level + 1
                //         << " -- face : " << l
                //         << " -- uid_face : " << child_faces_uids[l]
                //         << " -- owner : " << child_face_owner;
              }
              else {
                // debug() << "Child face (create face)  -- x : " << i
                //         << " -- y : " << j
                //         << " -- level : " << parent_cell_level + 1
                //         << " -- face : " << l
                //         << " -- uid_face : " << child_faces_uids[l]
                //         << " -- owner : " << child_face_owner;
              }
            }
          }

          // Partie Node.
          // Cette partie est assez ressemblante à la partie face, mis à part le fait qu'il peut y avoir
          // plus de propriétaires possibles.
          for (Integer l = 0; l < m_num_mng->nbNodeByCell(); ++l) {
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
            ((i == child_coord_x && !is_cell_around_parent_cell_in_subdomain(1, 0)) || (mask_node_if_cell_left[l])) &&
            ((i != (child_coord_x + pattern - 1) || !is_cell_around_parent_cell_in_subdomain(1, 2)) || mask_node_if_cell_right[l]) &&
            ((j == child_coord_y && !is_cell_around_parent_cell_in_subdomain(0, 1)) || (mask_node_if_cell_bottom[l])) &&
            ((j != (child_coord_y + pattern - 1) || !is_cell_around_parent_cell_in_subdomain(2, 1)) || mask_node_if_cell_top[l])) {
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
            ((i == child_coord_x && !is_cell_around_parent_cell_same_owner(1, 0)) || (mask_node_if_cell_left[l])) &&
            ((i != (child_coord_x + pattern - 1) || !is_cell_around_parent_cell_same_owner(1, 2)) || mask_node_if_cell_right[l]) &&
            ((j == child_coord_y && !is_cell_around_parent_cell_same_owner(0, 1)) || (mask_node_if_cell_bottom[l])) &&
            ((j != (child_coord_y + pattern - 1) || !is_cell_around_parent_cell_same_owner(2, 1)) || mask_node_if_cell_top[l])) {
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
                    child_node_owner = owner_cells_around_parent_cell(1, 1);
                  }
                }

                // Si le noeud est en haut de la maille parente ("sur la face haute").
                // Donc noeud en haut à droite (même position que le noeud de la maille parente).
                else if (j == (child_coord_y + pattern - 1) && (!mask_node_if_cell_top[l])) {
                  child_node_owner = owner_cells_around_parent_cell(1, 1);
                }

                // Si le noeud est quelque part sur la face parente droite...
                else {
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
                  child_node_owner = owner_cells_around_parent_cell(1, 1);
                }

                // Noeuds qui ne sont sur aucune face de la maille parente.
                else {
                  child_node_owner = owner_cells_around_parent_cell(1, 1);
                }
              }
            }

            // S'il y a une création de noeud et/ou un changement de propriétaire.
            if (child_node_owner != -1) {
              node_uid_to_owner[child_nodes_uids[l]] = child_node_owner;

              // Lorsqu'il y a un changement de propriétaire sans création de noeud,
              // on doit mettre de côté les uniqueIds de ces noeuds pour pouvoir
              // itérer dessus par la suite.
              if (!is_new_node) {
                node_uid_change_owner_only.add(child_nodes_uids[l]);
                // debug() << "Child node (change owner) -- x : " << i
                //         << " -- y : " << j
                //         << " -- level : " << parent_cell_level + 1
                //         << " -- node : " << l
                //         << " -- uid_node : " << child_nodes_uids[l]
                //         << " -- owner : " << child_node_owner;
              }
              else {
                // debug() << "Child node (create node)  -- x : " << i
                //         << " -- y : " << j
                //         << " -- level : " << parent_cell_level + 1
                //         << " -- node : " << l
                //         << " -- uid_node : " << child_nodes_uids[l]
                //         << " -- owner : " << child_node_owner;
              }
            }
          }
        }
      }
    }
  }

  // Pour le 3D, c'est très ressemblant, juste un peu plus long. Je recopie les commentaires, mais avec quelques adaptations.
  else if (m_mesh->dimension() == 3) {

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
    constexpr bool mask_node_if_cell_left[] = { false, true, true, false, false, true, true, false };
    constexpr bool mask_node_if_cell_bottom[] = { false, false, true, true, false, false, true, true };
    constexpr bool mask_node_if_cell_rear[] = { false, false, false, false, true, true, true, true };

    constexpr bool mask_node_if_cell_right[] = { true, false, false, true, true, false, false, true };
    constexpr bool mask_node_if_cell_top[] = { true, true, false, false, true, true, false, false };
    constexpr bool mask_node_if_cell_front[] = { true, true, true, true, false, false, false, false };

    constexpr bool mask_face_if_cell_left[] = { true, false, true, true, true, true };
    constexpr bool mask_face_if_cell_bottom[] = { true, true, false, true, true, true };
    constexpr bool mask_face_if_cell_rear[] = { false, true, true, true, true, true };

    constexpr bool mask_face_if_cell_right[] = { true, true, true, true, false, true };
    constexpr bool mask_face_if_cell_top[] = { true, true, true, true, true, false };
    constexpr bool mask_face_if_cell_front[] = { true, true, true, false, true, true };

    // Petite différence par rapport au 2D. Pour le 2D, la position des noeuds des faces
    // dans le tableau "child_nodes_uids" est toujours pareil (l et l+1, voir le 2D).
    // Pour le 3D, ce n'est pas le cas donc on a des tableaux pour avoir une correspondance
    // entre les noeuds de chaque face et la position des noeuds dans le tableau "child_nodes_uids".
    // (Exemple : pour la face 1 (même ordre d'énumération qu'Arcane), on doit prendre le
    // tableau "nodes_in_face_1" et donc les noeuds "child_nodes_uids[0]", "child_nodes_uids[3]",
    // "child_nodes_uids[7]" et "child_nodes_uids[4]").
    constexpr Integer nodes_in_face_0[] = { 0, 1, 2, 3 };
    constexpr Integer nodes_in_face_1[] = { 0, 3, 7, 4 };
    constexpr Integer nodes_in_face_2[] = { 0, 1, 5, 4 };
    constexpr Integer nodes_in_face_3[] = { 4, 5, 6, 7 };
    constexpr Integer nodes_in_face_4[] = { 1, 2, 6, 5 };
    constexpr Integer nodes_in_face_5[] = { 3, 2, 6, 7 };

    constexpr Integer nb_nodes_in_face = 4;

    // Pour la taille :
    // - on a "cell_to_refine_internals.size() * 8" mailles enfants,
    // - pour chaque maille, on a 2 infos (type de maille et uniqueId de la maille)
    // - pour chaque maille, on a "m_num_mng->getNbNode()" uniqueIds (les uniqueId de chaque noeud de la maille).
    cells_infos.reserve((cell_to_refine_internals.size() * 8) * (2 + m_num_mng->nbNodeByCell()));

    // Pour la taille, au maximum :
    // - on a "cell_to_refine_internals.size() * 36" faces enfants,
    // - pour chaque face, on a 2 infos (type de face et uniqueId de la face)
    // - pour chaque face, on a 4 uniqueIds de noeuds.
    faces_infos.reserve((cell_to_refine_internals.size() * 36) * (2 + 4));

    // Pour la taille, au maximum :
    // - on a (cell_to_refine_internals.size() * 27) uniqueIds de noeuds.
    nodes_infos.reserve(cell_to_refine_internals.size() * 27);

    FixedArray<Int64, 27> uid_cells_around_parent_cell_1d;
    FixedArray<Int32, 27> owner_cells_around_parent_cell_1d;
    FixedArray<Int32, 27> flags_cells_around_parent_cell_1d;

    for (Cell parent_cell : cell_to_refine_internals) {
      const Int64 parent_cell_uid = parent_cell.uniqueId();
      const Int32 parent_cell_level = parent_cell.level();

      const Int64 parent_coord_x = m_num_mng->cellUniqueIdToCoordX(parent_cell_uid, parent_cell_level);
      const Int64 parent_coord_y = m_num_mng->cellUniqueIdToCoordY(parent_cell_uid, parent_cell_level);
      const Int64 parent_coord_z = m_num_mng->cellUniqueIdToCoordZ(parent_cell_uid, parent_cell_level);

      const Int64 child_coord_x = m_num_mng->offsetLevelToLevel(parent_coord_x, parent_cell_level, parent_cell_level + 1);
      const Int64 child_coord_y = m_num_mng->offsetLevelToLevel(parent_coord_y, parent_cell_level, parent_cell_level + 1);
      const Int64 child_coord_z = m_num_mng->offsetLevelToLevel(parent_coord_z, parent_cell_level, parent_cell_level + 1);

      const Integer pattern = m_num_mng->pattern();

      m_num_mng->cellUniqueIdsAroundCell(uid_cells_around_parent_cell_1d.view(), parent_cell);

      for (Integer i = 0; i < 27; ++i) {
        Int64 uid_cell = uid_cells_around_parent_cell_1d[i];
        // Si uid_cell != -1 alors il y a peut-être une maille (mais on ne sait pas si elle est bien présente).
        // Si around_parent_cells_uid_to_owner[uid_cell] != -1 alors il y a bien une maille.
        if (uid_cell != -1 && around_parent_cells_uid_to_owner[uid_cell] != -1) {
          owner_cells_around_parent_cell_1d[i] = around_parent_cells_uid_to_owner[uid_cell];
          flags_cells_around_parent_cell_1d[i] = around_parent_cells_uid_to_flags[uid_cell];
        }
        else {
          uid_cells_around_parent_cell_1d[i] = -1;
          owner_cells_around_parent_cell_1d[i] = -1;
          flags_cells_around_parent_cell_1d[i] = 0;
        }
      }

      // Pour simplifier, on utilise des vues 3D. (array[Z][Y][X]).
      ConstArray3View uid_cells_around_parent_cell(uid_cells_around_parent_cell_1d.data(), 3, 3, 3);
      ConstArray3View owner_cells_around_parent_cell(owner_cells_around_parent_cell_1d.data(), 3, 3, 3);
      ConstArray3View flags_cells_around_parent_cell(flags_cells_around_parent_cell_1d.data(), 3, 3, 3);

      // #priority_owner_3d
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
      //
      // ^y
      // |
      // ->x

      // #arcane_order_to_around_3d
      // Note pour les maillages cartésiens 3D :
      // Les itérateurs sur les faces itèrent dans l'ordre (pour la maille 13 ici) :
      //  1. Face entre [13, 4],
      //  2. Face entre [13, 12],
      //  3. Face entre [13, 10],
      //  4. Face entre [13, 22],
      //  5. Face entre [13, 14],
      //  6. Face entre [13, 16],
      //
      // Les itérateurs sur les noeuds itèrent dans l'ordre (pour la maille 13 ici) :
      //  1. Noeud entre [13, 0]
      //  2. Noeud entre [13, 2]
      //  3. Noeud entre [13, 8]
      //  4. Noeud entre [13, 6]
      //  5. Noeud entre [13, 18]
      //  6. Noeud entre [13, 20]
      //  7. Noeud entre [13, 26]
      //  8. Noeud entre [13, 24]

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
      FixedArray<FixedArray<FixedArray<bool, 3>, 3>, 3> is_cell_around_parent_cell_present_and_useful;

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
      auto is_cell_around_parent_cell_in_subdomain = [&](const Integer z, const Integer y, const Integer x) {
        return is_cell_around_parent_cell_present_and_useful[z][y][x] && (flags_cells_around_parent_cell(z, y, x) & ItemFlags::II_UserMark1);
      };

      // ... on regarde si chaque maille parent est possédé par le même propriétaire que notre maille parent.
      auto is_cell_around_parent_cell_same_owner = [&](const Integer z, const Integer y, const Integer x) {
        return is_cell_around_parent_cell_present_and_useful[z][y][x] && (owner_cells_around_parent_cell(z, y, x) == owner_cells_around_parent_cell(1, 1, 1));
      };

      // ... on regarde si chaque maille parent a un propriétaire différent par rapport à notre maille parent.
      auto is_cell_around_parent_cell_different_owner = [&](const Integer z, const Integer y, const Integer x) {
        return is_cell_around_parent_cell_present_and_useful[z][y][x] && (owner_cells_around_parent_cell(z, y, x) != owner_cells_around_parent_cell(1, 1, 1));
      };

      // On itère sur toutes les mailles enfants.
      for (Int64 k = child_coord_z; k < child_coord_z + pattern; ++k) {
        for (Int64 j = child_coord_y; j < child_coord_y + pattern; ++j) {
          for (Int64 i = child_coord_x; i < child_coord_x + pattern; ++i) {
            parent_cells.add(parent_cell);
            total_nb_cells++;

            const Int64 child_cell_uid = m_num_mng->cellUniqueId(parent_cell_level + 1, Int64x3(i, j, k));
            // debug() << "Child -- x : " << i << " -- y : " << j << " -- z : " << k << " -- level : " << parent_cell_level + 1 << " -- uid : " << child_cell_uid;

            m_num_mng->cellNodeUniqueIds(child_nodes_uids, parent_cell_level + 1, Int64x3(i, j, k));
            m_num_mng->cellFaceUniqueIds(child_faces_uids, parent_cell_level + 1, Int64x3(i, j, k));

            constexpr Integer type_cell = IT_Hexaedron8;
            constexpr Integer type_face = IT_Quad4;

            // Partie Cell.
            cells_infos.add(type_cell);
            cells_infos.add(child_cell_uid);
            for (Integer nc = 0; nc < m_num_mng->nbNodeByCell(); nc++) {
              cells_infos.add(child_nodes_uids[nc]);
            }

            // Partie Face.
            for (Integer l = 0; l < m_num_mng->nbFaceByCell(); ++l) {
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
              ((i == child_coord_x && !is_cell_around_parent_cell_in_subdomain(1, 1, 0)) || mask_face_if_cell_left[l]) &&
              ((i != (child_coord_x + pattern - 1) || !is_cell_around_parent_cell_in_subdomain(1, 1, 2)) || mask_face_if_cell_right[l]) &&
              ((j == child_coord_y && !is_cell_around_parent_cell_in_subdomain(1, 0, 1)) || mask_face_if_cell_bottom[l]) &&
              ((j != (child_coord_y + pattern - 1) || !is_cell_around_parent_cell_in_subdomain(1, 2, 1)) || mask_face_if_cell_top[l]) &&
              ((k == child_coord_z && !is_cell_around_parent_cell_in_subdomain(0, 1, 1)) || mask_face_if_cell_rear[l]) &&
              ((k != (child_coord_z + pattern - 1) || !is_cell_around_parent_cell_in_subdomain(2, 1, 1)) || mask_face_if_cell_front[l])) {
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
              ((i == child_coord_x && !is_cell_around_parent_cell_same_owner(1, 1, 0)) || mask_face_if_cell_left[l]) &&
              ((i != (child_coord_x + pattern - 1) || !is_cell_around_parent_cell_same_owner(1, 1, 2)) || mask_face_if_cell_right[l]) &&
              ((j == child_coord_y && !is_cell_around_parent_cell_same_owner(1, 0, 1)) || mask_face_if_cell_bottom[l]) &&
              ((j != (child_coord_y + pattern - 1) || !is_cell_around_parent_cell_same_owner(1, 2, 1)) || mask_face_if_cell_top[l]) &&
              ((k == child_coord_z && !is_cell_around_parent_cell_same_owner(0, 1, 1)) || mask_face_if_cell_rear[l]) &&
              ((k != (child_coord_z + pattern - 1) || !is_cell_around_parent_cell_same_owner(2, 1, 1)) || mask_face_if_cell_front[l])) {
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
                if (i == child_coord_x && (!mask_face_if_cell_left[l]) && is_cell_around_parent_cell_different_owner(1, 1, 0)) {
                  child_face_owner = owner_cells_around_parent_cell(1, 1, 0);
                }

                // En bas, priorité 10 < 13 donc il prend la propriété de la face.
                else if (j == child_coord_y && (!mask_face_if_cell_bottom[l]) && is_cell_around_parent_cell_different_owner(1, 0, 1)) {
                  child_face_owner = owner_cells_around_parent_cell(1, 0, 1);
                }

                // À l'arrière, priorité 4 < 13 donc il prend la propriété de la face.
                else if (k == child_coord_z && (!mask_face_if_cell_rear[l]) && is_cell_around_parent_cell_different_owner(0, 1, 1)) {
                  child_face_owner = owner_cells_around_parent_cell(0, 1, 1);
                }

                // Sinon, parent_cell est propriétaire de la face.
                else {

                  // Sinon, c'est une face interne donc au parent_cell.
                  child_face_owner = owner_cells_around_parent_cell(1, 1, 1);
                }
              }

              // S'il y a une création de face et/ou un changement de propriétaire.
              if (child_face_owner != -1) {
                face_uid_to_owner[child_faces_uids[l]] = child_face_owner;

                // Lorsqu'il y a un changement de propriétaire sans création de face,
                // on doit mettre de côté les uniqueIds de ces faces pour pouvoir
                // itérer dessus par la suite.
                if (!is_new_face) {
                  face_uid_change_owner_only.add(child_faces_uids[l]);
                  // debug() << "Child face (change owner) -- x : " << i
                  //         << " -- y : " << j
                  //         << " -- z : " << k
                  //         << " -- level : " << parent_cell_level + 1
                  //         << " -- face : " << l
                  //         << " -- uid_face : " << child_faces_uids[l]
                  //         << " -- owner : " << child_face_owner;
                }
                else {
                  // debug() << "Child face (create face)  -- x : " << i
                  //         << " -- y : " << j
                  //         << " -- z : " << k
                  //         << " -- level : " << parent_cell_level + 1
                  //         << " -- face : " << l
                  //         << " -- uid_face : " << child_faces_uids[l]
                  //         << " -- owner : " << child_face_owner;
                }
              }
            }

            // Partie Node.
            // Cette partie est assez ressemblante à la partie face, mis à part le fait qu'il peut y avoir
            // plus de propriétaires possibles.
            for (Integer l = 0; l < m_num_mng->nbNodeByCell(); ++l) {
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
              ((i == child_coord_x && !is_cell_around_parent_cell_in_subdomain(1, 1, 0)) || mask_node_if_cell_left[l]) &&
              ((i != (child_coord_x + pattern - 1) || !is_cell_around_parent_cell_in_subdomain(1, 1, 2)) || mask_node_if_cell_right[l]) &&
              ((j == child_coord_y && !is_cell_around_parent_cell_in_subdomain(1, 0, 1)) || mask_node_if_cell_bottom[l]) &&
              ((j != (child_coord_y + pattern - 1) || !is_cell_around_parent_cell_in_subdomain(1, 2, 1)) || mask_node_if_cell_top[l]) &&
              ((k == child_coord_z && !is_cell_around_parent_cell_in_subdomain(0, 1, 1)) || mask_node_if_cell_rear[l]) &&
              ((k != (child_coord_z + pattern - 1) || !is_cell_around_parent_cell_in_subdomain(2, 1, 1)) || mask_node_if_cell_front[l])) {
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
              ((i == child_coord_x && !is_cell_around_parent_cell_same_owner(1, 1, 0)) || mask_node_if_cell_left[l]) &&
              ((i != (child_coord_x + pattern - 1) || !is_cell_around_parent_cell_same_owner(1, 1, 2)) || mask_node_if_cell_right[l]) &&
              ((j == child_coord_y && !is_cell_around_parent_cell_same_owner(1, 0, 1)) || mask_node_if_cell_bottom[l]) &&
              ((j != (child_coord_y + pattern - 1) || !is_cell_around_parent_cell_same_owner(1, 2, 1)) || mask_node_if_cell_top[l]) &&
              ((k == child_coord_z && !is_cell_around_parent_cell_same_owner(0, 1, 1)) || mask_node_if_cell_rear[l]) &&
              ((k != (child_coord_z + pattern - 1) || !is_cell_around_parent_cell_same_owner(2, 1, 1)) || mask_node_if_cell_front[l])) {

                // Par rapport aux faces qui n'ont que deux propriétaires possibles, un noeud peut
                // en avoir jusqu'à huit.

                // Si le noeud est sur la face gauche de la maille parente.
                if (i == child_coord_x && (!mask_node_if_cell_left[l])) {

                  // Si le noeud est sur la face basse de la maille parente.
                  // Donc noeud sur l'arête à gauche en bas.
                  if (j == child_coord_y && (!mask_node_if_cell_bottom[l])) {

                    // Si le noeud est sur la face arrière de la maille parente.
                    // Donc noeud à gauche, en bas, en arrière (même position que le noeud de la maille parente).
                    if (k == child_coord_z && (!mask_node_if_cell_rear[l])) {

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
                    else if (k == (child_coord_z + pattern - 1) && (!mask_node_if_cell_front[l])) {

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
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // Sinon le noeud est quelque part sur l'arête à gauche en bas...
                    else {

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
                  else if (j == (child_coord_y + pattern - 1) && (!mask_node_if_cell_top[l])) {

                    // Si le noeud est sur la face arrière de la maille parente.
                    // Donc noeud à gauche, en haut, en arrière (même position que le noeud de la maille parente).
                    if (k == child_coord_z && (!mask_node_if_cell_rear[l])) {

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
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // Si le noeud est sur la face avant de la maille parente.
                    // Donc noeud à gauche, en haut, en avant (même position que le noeud de la maille parente).
                    else if (k == (child_coord_z + pattern - 1) && (!mask_node_if_cell_front[l])) {

                      // Priorité 4 < 13.
                      if (is_cell_around_parent_cell_different_owner(1, 1, 0)) {
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 0);
                      }

                      // Sinon, parent_cell est propriétaire du noeud.
                      else {
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
                else if (i == (child_coord_x + pattern - 1) && (!mask_node_if_cell_right[l])) {

                  // Si le noeud est sur la face basse de la maille parente.
                  // Donc noeud sur l'arête à droite en bas.
                  if (j == child_coord_y && (!mask_node_if_cell_bottom[l])) {

                    // Si le noeud est sur la face arrière de la maille parente.
                    // Donc noeud à droite, en bas, en arrière (même position que le noeud de la maille parente).
                    if (k == child_coord_z && (!mask_node_if_cell_rear[l])) {

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
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // Si le noeud est sur la face avant de la maille parente.
                    // Donc noeud à droite, en bas, en avant (même position que le noeud de la maille parente).
                    else if (k == (child_coord_z + pattern - 1) && (!mask_node_if_cell_front[l])) {

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
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }
                  }

                  // Si le noeud est sur la face haute de la maille parente.
                  // Donc noeud sur l'arête à droite en haut.
                  else if (j == (child_coord_y + pattern - 1) && (!mask_node_if_cell_top[l])) {

                    // Si le noeud est sur la face arrière de la maille parente.
                    // Donc noeud à droite, en haut, en arrière (même position que le noeud de la maille parente).
                    if (k == child_coord_z && (!mask_node_if_cell_rear[l])) {

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
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // Si le noeud est sur la face avant de la maille parente.
                    // Donc noeud à droite, en haut, en avant (même position que le noeud de la maille parente).
                    else if (k == (child_coord_z + pattern - 1) && (!mask_node_if_cell_front[l])) {
                      child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                    }

                    // Sinon le noeud est quelque part sur l'arête à droite en haut...
                    else {
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
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // Si le noeud est quelque part sur l'arête à droite en avant.
                    else if (k == (child_coord_z + pattern - 1) && (!mask_node_if_cell_front[l])) {
                      child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                    }

                    // Sinon le noeud est quelque part sur la face droite...
                    else {
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
                        child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                      }
                    }

                    // Si le noeud est sur la face avant de la maille parente.
                    // Donc noeud sur l'arête en avant en haut.
                    else if (k == (child_coord_z + pattern - 1) && (!mask_node_if_cell_front[l])) {
                      child_node_owner = owner_cells_around_parent_cell(1, 1, 1);
                    }

                    // Sinon le noeud est quelque part sur la face en haut...
                    else {
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
              if (child_node_owner != -1) {
                node_uid_to_owner[child_nodes_uids[l]] = child_node_owner;

                // Lorsqu'il y a un changement de propriétaire sans création de noeud,
                // on doit mettre de côté les uniqueIds de ces noeuds pour pouvoir
                // itérer dessus par la suite.
                if (!is_new_node) {
                  node_uid_change_owner_only.add(child_nodes_uids[l]);
                  // debug() << "Child node (change owner) -- x : " << i
                  //         << " -- y : " << j
                  //         << " -- z : " << k
                  //         << " -- level : " << parent_cell_level + 1
                  //         << " -- node : " << l
                  //         << " -- uid_node : " << child_nodes_uids[l]
                  //         << " -- owner : " << child_node_owner;
                }
                else {
                  // debug() << "Child node (create node)  -- x : " << i
                  //         << " -- y : " << j
                  //         << " -- z : " << k
                  //         << " -- level : " << parent_cell_level + 1
                  //         << " -- node : " << l
                  //         << " -- uid_node : " << child_nodes_uids[l]
                  //         << " -- owner : " << child_node_owner;
                }
              }
            }
          }
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
      const Integer nb_node_owner_change = node_uid_change_owner_only.size();

      // Ce tableau contiendra les localIds des nouveaux noeuds mais aussi les localIds
      // des noeuds qui changent juste de propriétaire.
      UniqueArray<Int32> nodes_lid(total_nb_nodes + nb_node_owner_change);

      // On crée les noeuds. On met les localIds des nouveaux noeuds au début du tableau.
      m_mesh->modifier()->addNodes(nodes_infos, nodes_lid.subView(0, total_nb_nodes));

      // On cherche les localIds des noeuds qui changent de proprio et on les met à la fin du tableau.
      m_mesh->nodeFamily()->itemsUniqueIdToLocalId(nodes_lid.subView(total_nb_nodes, nb_node_owner_change), node_uid_change_owner_only, true);

      UniqueArray<Int64> uid_child_nodes(total_nb_nodes + nb_node_owner_change);
      UniqueArray<Int32> lid_child_nodes(total_nb_nodes + nb_node_owner_change);
      Integer index = 0;

      // On attribue les bons propriétaires aux noeuds.
      ENUMERATE_ (Node, inode, m_mesh->nodeFamily()->view(nodes_lid)) {
        Node node = *inode;
        node.mutableItemBase().setOwner(node_uid_to_owner[node.uniqueId()], my_rank);

        if (node_uid_to_owner[node.uniqueId()] == my_rank) {
          node.mutableItemBase().addFlags(ItemFlags::II_Own);
        }
        // TODO : Corriger ça dans la partie concerné directement.
        else {
          node.mutableItemBase().removeFlags(ItemFlags::II_Shared);
        }
        // Attention, node.level() == -1 ici.
        uid_child_nodes[index++] = m_num_mng->parentNodeUniqueIdOfNode(node.uniqueId(), max_level + 1, false);
      }

      m_mesh->nodeFamily()->itemsUniqueIdToLocalId(lid_child_nodes, uid_child_nodes, false);
      NodeInfoListView nodes(m_mesh->nodeFamily());

      index = 0;
      ENUMERATE_ (Node, inode, m_mesh->nodeFamily()->view(nodes_lid)) {
        const Int32 child_lid = lid_child_nodes[index++];
        if (child_lid == NULL_ITEM_ID) {
          continue;
        }

        Node parent = nodes[child_lid];
        Node child = *inode;

        m_mesh->modifier()->addParentNodeToNode(child, parent);
        m_mesh->modifier()->addChildNodeToNode(parent, child);
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

      UniqueArray<Int64> uid_parent_faces(total_nb_faces + nb_face_owner_change);
      UniqueArray<Int32> lid_parent_faces(total_nb_faces + nb_face_owner_change);
      Integer index = 0;

      // On attribue les bons propriétaires aux faces.
      ENUMERATE_ (Face, iface, m_mesh->faceFamily()->view(faces_lid)) {
        Face face = *iface;
        face.mutableItemBase().setOwner(face_uid_to_owner[face.uniqueId()], my_rank);

        if (face_uid_to_owner[face.uniqueId()] == my_rank) {
          face.mutableItemBase().addFlags(ItemFlags::II_Own);
        }
        // TODO : Corriger ça dans la partie concerné directement.
        else {
          face.mutableItemBase().removeFlags(ItemFlags::II_Shared);
        }
        // Attention, face.level() == -1 ici.
        uid_parent_faces[index++] = m_num_mng->parentFaceUniqueIdOfFace(face.uniqueId(), max_level + 1, false);
        // debug() << "Parent of : " << face.uniqueId() << " is : " << uid_child_faces[index - 1];
      }

      m_mesh->faceFamily()->itemsUniqueIdToLocalId(lid_parent_faces, uid_parent_faces, false);
      FaceInfoListView faces(m_mesh->faceFamily());

      index = 0;
      ENUMERATE_ (Face, iface, m_mesh->faceFamily()->view(faces_lid)) {
        const Int32 child_lid = lid_parent_faces[index++];
        if (child_lid == NULL_ITEM_ID) {
          continue;
        }

        Face parent = faces[child_lid];
        Face child = *iface;

        m_mesh->modifier()->addParentFaceToFace(child, parent);
        m_mesh->modifier()->addChildFaceToFace(parent, child);
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
    for (Integer i = 0; i < total_nb_cells; ++i) {
      Cell child = cells[cells_lid[i]];

      child.mutableItemBase().setOwner(parent_cells[i].owner(), my_rank);

      child.mutableItemBase().addFlags(ItemFlags::II_JustAdded);

      if (parent_cells[i].owner() == my_rank) {
        child.mutableItemBase().addFlags(ItemFlags::II_Own);
      }

      if (parent_cells[i].itemBase().flags() & ItemFlags::II_Shared) {
        child.mutableItemBase().addFlags(ItemFlags::II_Shared);
      }

      m_mesh->modifier()->addParentCellToCell(child, parent_cells[i]);
      m_mesh->modifier()->addChildCellToCell(parent_cells[i], child);
    }

    // Itération sur les mailles parentes.
    for (Cell cell : cell_to_refine_internals) {
      cell.mutableItemBase().removeFlags(ItemFlags::II_Refine);
      cell.mutableItemBase().addFlags(ItemFlags::II_JustRefined | ItemFlags::II_Inactive);
    }
    m_mesh->cellFamily()->notifyItemsOwnerChanged();
  }

  m_mesh->modifier()->endUpdate();

  // On positionne les noeuds dans l'espace.
  for (Cell parent_cell : cell_to_refine_internals) {
    m_num_mng->setChildNodeCoordinates(parent_cell);
    // On ajoute le flag "II_Shared" aux noeuds et aux faces des mailles partagées.
    if (parent_cell.mutableItemBase().flags() & ItemFlags::II_Shared) {
      for (Integer i = 0; i < parent_cell.nbHChildren(); ++i) {
        Cell child_cell = parent_cell.hChild(i);
        for (Node node : child_cell.nodes()) {
          if (node.mutableItemBase().flags() & ItemFlags::II_Own) {
            node.mutableItemBase().addFlags(ItemFlags::II_Shared);
          }
        }

        for (Face face : child_cell.faces()) {
          if (face.mutableItemBase().flags() & ItemFlags::II_Own) {
            face.mutableItemBase().addFlags(ItemFlags::II_Shared);
          }
        }
      }
    }
  }

  // Recalcule les informations de synchronisation.
  m_mesh->computeSynchronizeInfos();

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
  //  ENUMERATE_ (Cell, icell, m_mesh->allCells()) {
  //    debug() << "\tCell uniqueId : " << icell->uniqueId() << " -- level : " << icell->level() << " -- nbChildren : " << icell->nbHChildren();
  //    for (Integer i = 0; i < icell->nbHChildren(); ++i) {
  //      debug() << "\t\tChild uniqueId : " << icell->hChild(i).uniqueId() << " -- level : " << icell->hChild(i).level() << " -- nbChildren : " << icell->hChild(i).nbHChildren();
  //    }
  //  }
  //  info() << "Résumé node:";
  //  ENUMERATE_ (Node, inode, m_mesh->allNodes()) {
  //    debug() << "\tNode uniqueId : " << inode->uniqueId() << " -- level : " << inode->level() << " -- nbChildren : " << inode->nbHChildren();
  //    for (Integer i = 0; i < inode->nbHChildren(); ++i) {
  //      debug() << "\t\tNode Child uniqueId : " << inode->hChild(i).uniqueId() << " -- level : " << inode->hChild(i).level() << " -- nbChildren : " << inode->hChild(i).nbHChildren();
  //    }
  //  }
  //
  //  info() << "Résumé :";
  //  ENUMERATE_ (Face, iface, m_mesh->allFaces()) {
  //    debug() << "\tFace uniqueId : " << iface->uniqueId() << " -- level : " << iface->level() << " -- nbChildren : " << iface->nbHChildren();
  //    for (Integer i = 0; i < iface->nbHChildren(); ++i) {
  //      debug() << "\t\tChild uniqueId : " << iface->hChild(i).uniqueId() << " -- level : " << iface->hChild(i).level() << " -- nbChildren : " << iface->hChild(i).nbHChildren();
  //    }
  //  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAMRPatchMng::
createSubLevel()
{
  IParallelMng* pm = m_mesh->parallelMng();
  Int32 nb_rank = pm->commSize();
  Int32 my_rank = pm->commRank();

  UniqueArray<Int64> cell_uid_to_create;

  // TODO : Remplacer around_parent_cells_uid_to_owner par parent_to_child_cells ?
  std::unordered_map<Int64, Int32> around_parent_cells_uid_to_owner;
  std::unordered_map<Int64, bool> around_parent_cells_uid_is_in_subdomain;
  std::unordered_map<Int64, UniqueArray<Cell>> parent_to_child_cells;

  std::unordered_map<Int64, Int32> node_uid_to_owner;
  std::unordered_map<Int64, Int32> face_uid_to_owner;

  // On va créer le niveau -1.
  // À noter qu'à la fin de la méthode, on replacera ce niveau
  // à 0.
  m_num_mng->prepareLevel(-1);

  // On crée une ou plusieurs couches de mailles fantômes
  // pour éviter qu'une maille parente n'ai pas le même
  // nombre de mailles enfant.
  // ----------
  // CartesianMeshCoarsening2::_doDoubleGhostLayers()
  IMeshModifier* mesh_modifier = m_mesh->modifier();
  IGhostLayerMng* gm = m_mesh->ghostLayerMng();
  // Il faut au moins utiliser la version 3 pour pouvoir supporter
  // plusieurs couches de mailles fantômes
  Int32 version = gm->builderVersion();
  if (version < 3)
    gm->setBuilderVersion(3);
  Int32 nb_ghost_layer = gm->nbGhostLayer();
  // TODO AH : Cette ligne permettrait d'avoir moins de mailles fantômes et
  // d'éviter leurs suppressions en cas d'inutilité. Mais le comportement
  // serait différent de l'AMR historique.
  //gm->setNbGhostLayer(nb_ghost_layer + (nb_ghost_layer % m_num_mng->pattern()));
  // Comportement de l'AMR historique.
  gm->setNbGhostLayer(nb_ghost_layer * 2);
  mesh_modifier->setDynamic(true);
  mesh_modifier->updateGhostLayers();
  // Remet le nombre initial de couches de mailles fantômes
  gm->setNbGhostLayer(nb_ghost_layer);
  // CartesianMeshCoarsening2::_doDoubleGhostLayers()
  // ----------

  // On récupère les uniqueIds des parents à créer.
  ENUMERATE_ (Cell, icell, m_mesh->allLevelCells(0)) {
    Cell cell = *icell;

    Int64 parent_uid = m_num_mng->parentCellUniqueIdOfCell(cell);

    // On évite les doublons.
    if (!cell_uid_to_create.contains(parent_uid)) {
      cell_uid_to_create.add(parent_uid);
      // On en profite pour sauvegarder les owners des futures mailles
      // qui seront les mêmes owners que les mailles enfants.
      around_parent_cells_uid_to_owner[parent_uid] = cell.owner();
      around_parent_cells_uid_is_in_subdomain[parent_uid] = true;
    }
    else {
      // Ça peut arriver si le partitionnement n'est pas adapté.
      if (around_parent_cells_uid_to_owner[parent_uid] != cell.owner()) {
        ARCANE_FATAL("Pb owner -- Two+ children, two+ different owners, same parent\n"
                     "The ground patch size in x, y (and z if 3D) must be a multiple of four (need partitionner update to support multiple of two)\n"
                     "CellUID : {0} -- CellOwner : {1} -- OtherChildOwner : {2}",
                     cell.uniqueId(), cell.owner(), around_parent_cells_uid_to_owner[parent_uid]);
      }
    }

    // On doit sauvegarder les enfants des parents pour créer les connectivités
    // à la fin.
    parent_to_child_cells[parent_uid].add(cell);
  }

  //  info() << cell_uid_to_create;
  //  for (const auto& [key, value] : parent_to_child_cells) {
  //    info() << "Parent : " << key << " -- Children : " << value;
  //  }

  UniqueArray<Int64> cells_infos;
  UniqueArray<Int64> faces_infos;
  UniqueArray<Int64> nodes_infos;

  Integer total_nb_cells = 0;
  Integer total_nb_nodes = 0;
  Integer total_nb_faces = 0;

  // Deux tableaux permettant de récupérer les uniqueIds des noeuds et des faces
  // de chaque maille parent à chaque appel à getNodeUids()/getFaceUids().
  UniqueArray<Int64> parent_nodes_uids(m_num_mng->nbNodeByCell());
  UniqueArray<Int64> parent_faces_uids(m_num_mng->nbFaceByCell());

  // Partie échange d'informations sur les mailles autour du patch
  // (pour remplacer les mailles fantômes).
  {
    // Tableau qui contiendra les uids des mailles dont on a besoin des infos.
    UniqueArray<Int64> uid_of_cells_needed;
    {
      UniqueArray<Int64> cell_uids_around((m_mesh->dimension() == 2) ? 9 : 27);
      for (Int64 parent_cell : cell_uid_to_create) {
        m_num_mng->cellUniqueIdsAroundCell(cell_uids_around, parent_cell, -1);
        for (Int64 cell_uid : cell_uids_around) {
          // Si -1 alors il n'y a pas de mailles à cette position.
          if (cell_uid == -1)
            continue;

          // TODO C++20 : Mettre map.contains().
          // SI on a la maille, on n'a pas besoin de demander d'infos.
          if (around_parent_cells_uid_to_owner.find(cell_uid) != around_parent_cells_uid_to_owner.end())
            continue;

          // TODO : Bof
          if (!uid_of_cells_needed.contains(cell_uid)) {
            uid_of_cells_needed.add(cell_uid);

            // Si on a besoin des infos, c'est que l'on ne les possèdent pas :-)
            // On en profite pour enregistrer cette information pour distinguer les
            // mailles fantômes dont on possède les items (faces/noeuds) de celle dont
            // on ne possède rien.
            around_parent_cells_uid_is_in_subdomain[cell_uid] = false;
          }
        }
      }
    }

    // On partage les cell uid nécessaires de tout le monde.
    UniqueArray<Int64> uid_of_cells_needed_all_procs;
    pm->allGatherVariable(uid_of_cells_needed, uid_of_cells_needed_all_procs);

    UniqueArray<Int32> owner_of_cells_needed_all_procs(uid_of_cells_needed_all_procs.size());

    {
      // On enregistre le propriétaire des mailles que l'on possède.
      for (Integer i = 0; i < uid_of_cells_needed_all_procs.size(); ++i) {
        if (around_parent_cells_uid_to_owner.find(uid_of_cells_needed_all_procs[i]) != around_parent_cells_uid_to_owner.end()) {
          owner_of_cells_needed_all_procs[i] = around_parent_cells_uid_to_owner[uid_of_cells_needed_all_procs[i]];
        }
        else {
          // Le ReduceMax fera disparaitre ce -1.
          owner_of_cells_needed_all_procs[i] = -1;
        }
      }
    }

    // On récupère les owners de toutes les mailles nécessaires.
    pm->reduce(Parallel::eReduceType::ReduceMax, owner_of_cells_needed_all_procs);

    // On ne traite que les owners des mailles nécessaires pour nous.
    {
      Integer size_uid_of_cells_needed = uid_of_cells_needed.size();
      Integer my_pos_in_all_procs_arrays = 0;
      UniqueArray<Integer> size_uid_of_cells_needed_per_proc(nb_rank);
      ArrayView<Integer> av(1, &size_uid_of_cells_needed);
      pm->allGather(av, size_uid_of_cells_needed_per_proc);

      // On zap les mailles de tous les procs avant nous.
      for (Integer i = 0; i < my_rank; ++i) {
        my_pos_in_all_procs_arrays += size_uid_of_cells_needed_per_proc[i];
      }

      // On enregistre les owners nécessaires.
      ArrayView<Int32> owner_of_cells_needed = owner_of_cells_needed_all_procs.subView(my_pos_in_all_procs_arrays, size_uid_of_cells_needed);
      for (Integer i = 0; i < size_uid_of_cells_needed; ++i) {
        around_parent_cells_uid_to_owner[uid_of_cells_needed[i]] = owner_of_cells_needed[i];

        // En rafinnement, il peut y avoir plusieurs niveaux d'écarts entre les patchs.
        // En déraffinement, c'est impossible vu que le niveau 0 n'a pas de "trous".
        if (owner_of_cells_needed[i] == -1) {
          ARCANE_FATAL("En déraffinement, c'est normalement impossible");
        }
      }
    }
  }

  if (m_mesh->dimension() == 2) {

    // Masques permettant de savoir si on doit créer une faces/noeuds (true)
    // ou si on doit regarder la maille d'à côté avant (false).
    // Rappel que le parcours des faces par Arcane est dans l'ordre NumPad{2, 6, 8, 4}.
    constexpr bool mask_face_if_cell_left[] = { true, true, true, false };
    constexpr bool mask_face_if_cell_bottom[] = { false, true, true, true };

    // Rappel que le parcours des nodes par Arcane est dans l'ordre NumPad{1, 3, 9, 7}.
    constexpr bool mask_node_if_cell_left[] = { false, true, true, false };
    constexpr bool mask_node_if_cell_bottom[] = { false, false, true, true };

    FixedArray<Int64, 9> cells_uid_around;
    FixedArray<Int32, 9> owner_cells_around_parent_cell_1d;
    FixedArray<bool, 9> is_not_in_subdomain_cells_around_parent_cell_1d;

    // Pour le raffinement, on parcourait les mailles parents existantes.
    // Ici, les mailles parents n'existent pas encore, donc on parcours les uid.
    for (Int64 parent_cell_uid : cell_uid_to_create) {

      m_num_mng->cellUniqueIdsAroundCell(cells_uid_around.view(), parent_cell_uid, -1);

      ConstArray2View owner_cells_around_parent_cell(owner_cells_around_parent_cell_1d.data(), 3, 3);
      // Attention au "not" dans le nom de la variable.
      ConstArray2View is_not_in_subdomain_cells_around_parent_cell(is_not_in_subdomain_cells_around_parent_cell_1d.data(), 3, 3);

      for (Integer i = 0; i < 9; ++i) {
        Int64 uid_cell = cells_uid_around[i];
        // Si uid_cell != -1 alors il y a peut-être une maille (mais on ne sait pas si elle est bien présente).
        // Si around_parent_cells_uid_to_owner[uid_cell] != -1 alors il y a bien une maille.
        if (uid_cell != -1 && around_parent_cells_uid_to_owner[uid_cell] != -1) {
          owner_cells_around_parent_cell_1d[i] = around_parent_cells_uid_to_owner[uid_cell];
          is_not_in_subdomain_cells_around_parent_cell_1d[i] = !around_parent_cells_uid_is_in_subdomain[uid_cell];
        }
        else {
          cells_uid_around[i] = -1;
          owner_cells_around_parent_cell_1d[i] = -1;
          is_not_in_subdomain_cells_around_parent_cell_1d[i] = true;
        }
      }

      // Ces deux lambdas sont différentes.
      // Quand une parent_cell n'existe pas, il y a -1 dans le tableau adéquat,
      // la première lambda répondra donc forcément true alors que la seconde false.
      auto is_cell_around_parent_cell_different_owner = [&](const Integer y, const Integer x) {
        return (owner_cells_around_parent_cell(y, x) != owner_cells_around_parent_cell(1, 1));
      };

      auto is_cell_around_parent_cell_exist_and_different_owner = [&](const Integer y, const Integer x) {
        return (owner_cells_around_parent_cell(y, x) != -1 && (owner_cells_around_parent_cell(y, x) != owner_cells_around_parent_cell(1, 1)));
      };

      total_nb_cells++;
      // debug() << "Parent"
      //         << " -- x : " << m_num_mng->cellUniqueIdToCoordX(parent_cell_uid, -1)
      //         << " -- y : " << m_num_mng->cellUniqueIdToCoordY(parent_cell_uid, -1)
      //         << " -- level : " << -1
      //         << " -- uid : " << parent_cell_uid;

      // On récupère les uniqueIds des nodes et faces à créer.
      m_num_mng->cellNodeUniqueIds(parent_nodes_uids, -1, parent_cell_uid);
      m_num_mng->cellFaceUniqueIds(parent_faces_uids, -1, parent_cell_uid);

      constexpr Integer type_cell = IT_Quad4;
      constexpr Integer type_face = IT_Line2;

      // Partie Cell.
      cells_infos.add(type_cell);
      cells_infos.add(parent_cell_uid);
      for (Integer nc = 0; nc < m_num_mng->nbNodeByCell(); nc++) {
        cells_infos.add(parent_nodes_uids[nc]);
      }

      // Partie Face.
      for (Integer l = 0; l < m_num_mng->nbFaceByCell(); ++l) {
        // On regarde si l'on doit traiter la face.
        // Si mask_face_if_cell_left[l] == false, on doit regarder si la maille à gauche est à nous ou non
        // ou si la maille à gauche est dans notre sous-domaine ou non.
        // Si cette maille n'est pas à nous et/ou n'est pas sur notre sous-domaine,
        // on doit créer la face en tant que face fantôme.
        if (
        (mask_face_if_cell_left[l] || is_cell_around_parent_cell_different_owner(1, 0) || is_not_in_subdomain_cells_around_parent_cell(1, 0)) &&
        (mask_face_if_cell_bottom[l] || is_cell_around_parent_cell_different_owner(0, 1) || is_not_in_subdomain_cells_around_parent_cell(0, 1))) {
          Integer parent_face_owner = -1;
          faces_infos.add(type_face);
          faces_infos.add(parent_faces_uids[l]);

          // Les noeuds de la face sont toujours les noeuds l et l+1
          // car on utilise la même exploration pour les deux cas.
          for (Integer nc = l; nc < l + 2; nc++) {
            faces_infos.add(parent_nodes_uids[nc % m_num_mng->nbNodeByCell()]);
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
          // debug() << "Parent face (create face)  -- parent_cell_uid : " << parent_cell_uid
          //         << " -- level : " << -1
          //         << " -- face : " << l
          //         << " -- uid_face : " << parent_faces_uids[l]
          //         << " -- owner : " << parent_face_owner;
        }
      }

      // Partie Node.
      // Cette partie est assez ressemblante à la partie face, mis à part le fait qu'il peut y avoir
      // plus de propriétaires possibles.
      for (Integer l = 0; l < m_num_mng->nbNodeByCell(); ++l) {
        if (
        (mask_node_if_cell_left[l] || is_cell_around_parent_cell_different_owner(1, 0) || is_not_in_subdomain_cells_around_parent_cell(1, 0)) &&
        (mask_node_if_cell_bottom[l] || is_cell_around_parent_cell_different_owner(0, 1) || is_not_in_subdomain_cells_around_parent_cell(0, 1))) {
          Integer parent_node_owner = -1;
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
          // debug() << "Parent node (create node)  -- parent_cell_uid : " << parent_cell_uid
          //         << " -- level : " << -1
          //         << " -- node : " << l
          //         << " -- uid_node : " << parent_nodes_uids[l]
          //         << " -- owner : " << parent_node_owner;
        }
      }
    }
  }
  else if (m_mesh->dimension() == 3) {

    // Masques permettant de savoir si on doit créer une faces/noeuds (true)
    // ou si on doit regarder la maille d'à côté avant (false).
    constexpr bool mask_node_if_cell_left[] = { false, true, true, false, false, true, true, false };
    constexpr bool mask_node_if_cell_bottom[] = { false, false, true, true, false, false, true, true };
    constexpr bool mask_node_if_cell_rear[] = { false, false, false, false, true, true, true, true };

    constexpr bool mask_face_if_cell_left[] = { true, false, true, true, true, true };
    constexpr bool mask_face_if_cell_bottom[] = { true, true, false, true, true, true };
    constexpr bool mask_face_if_cell_rear[] = { false, true, true, true, true, true };

    // Petite différence par rapport au 2D. Pour le 2D, la position des noeuds des faces
    // dans le tableau "parent_nodes_uids" est toujours pareil (l et l+1, voir le 2D).
    // Pour le 3D, ce n'est pas le cas donc on a des tableaux pour avoir une correspondance
    // entre les noeuds de chaque face et la position des noeuds dans le tableau "parent_nodes_uids".
    // (Exemple : pour la face 1 (même ordre d'énumération qu'Arcane), on doit prendre le
    // tableau "nodes_in_face_1" et donc les noeuds "parent_nodes_uids[0]", "parent_nodes_uids[3]",
    // "parent_nodes_uids[7]" et "parent_nodes_uids[4]").
    constexpr Integer nodes_in_face_0[] = { 0, 1, 2, 3 };
    constexpr Integer nodes_in_face_1[] = { 0, 3, 7, 4 };
    constexpr Integer nodes_in_face_2[] = { 0, 1, 5, 4 };
    constexpr Integer nodes_in_face_3[] = { 4, 5, 6, 7 };
    constexpr Integer nodes_in_face_4[] = { 1, 2, 6, 5 };
    constexpr Integer nodes_in_face_5[] = { 3, 2, 6, 7 };

    constexpr Integer nb_nodes_in_face = 4;
    FixedArray<Int64, 27> cells_uid_around;
    FixedArray<Int32, 27> owner_cells_around_parent_cell_1d;
    FixedArray<bool, 27> is_not_in_subdomain_cells_around_parent_cell_1d;

    // Pour le raffinement, on parcourait les mailles parents existantes.
    // Ici, les mailles parents n'existent pas encore, donc on parcours les uid.
    for (Int64 parent_cell_uid : cell_uid_to_create) {

      m_num_mng->cellUniqueIdsAroundCell(cells_uid_around.view(), parent_cell_uid, -1);

      ConstArray3View owner_cells_around_parent_cell(owner_cells_around_parent_cell_1d.data(), 3, 3, 3);
      // Attention au "not" dans le nom de la variable.
      ConstArray3View is_not_in_subdomain_cells_around_parent_cell(is_not_in_subdomain_cells_around_parent_cell_1d.data(), 3, 3, 3);

      for (Integer i = 0; i < 27; ++i) {
        Int64 uid_cell = cells_uid_around[i];
        // Si uid_cell != -1 alors il y a peut-être une maille (mais on ne sait pas si elle est bien présente).
        // Si around_parent_cells_uid_to_owner[uid_cell] != -1 alors il y a bien une maille.
        if (uid_cell != -1 && around_parent_cells_uid_to_owner[uid_cell] != -1) {
          owner_cells_around_parent_cell_1d[i] = around_parent_cells_uid_to_owner[uid_cell];
          is_not_in_subdomain_cells_around_parent_cell_1d[i] = !around_parent_cells_uid_is_in_subdomain[uid_cell];
        }
        else {
          cells_uid_around[i] = -1;
          owner_cells_around_parent_cell_1d[i] = -1;
          is_not_in_subdomain_cells_around_parent_cell_1d[i] = true;
        }
      }

      // Ces deux lambdas sont différentes.
      // Quand une parent_cell n'existe pas, il y a -1 dans le tableau adéquat,
      // la première lambda répondra donc forcément true alors que la seconde false.
      auto is_cell_around_parent_cell_different_owner = [&](const Integer z, const Integer y, const Integer x) {
        return (owner_cells_around_parent_cell(z, y, x) != owner_cells_around_parent_cell(1, 1, 1));
      };

      auto is_cell_around_parent_cell_exist_and_different_owner = [&](const Integer z, const Integer y, const Integer x) {
        return (owner_cells_around_parent_cell(z, y, x) != -1 && (owner_cells_around_parent_cell(z, y, x) != owner_cells_around_parent_cell(1, 1, 1)));
      };

      total_nb_cells++;
      // debug() << "Parent"
      //         << " -- x : " << m_num_mng->cellUniqueIdToCoordX(parent_cell_uid, -1)
      //         << " -- y : " << m_num_mng->cellUniqueIdToCoordY(parent_cell_uid, -1)
      //         << " -- z : " << m_num_mng->cellUniqueIdToCoordZ(parent_cell_uid, -1)
      //         << " -- level : " << -1
      //         << " -- uid : " << parent_cell_uid;

      // On récupère les uniqueIds des nodes et faces à créer.
      m_num_mng->cellNodeUniqueIds(parent_nodes_uids, -1, parent_cell_uid);
      m_num_mng->cellFaceUniqueIds(parent_faces_uids, -1, parent_cell_uid);

      constexpr Integer type_cell = IT_Hexaedron8;
      constexpr Integer type_face = IT_Quad4;

      // Partie Cell.
      cells_infos.add(type_cell);
      cells_infos.add(parent_cell_uid);
      for (Integer nc = 0; nc < m_num_mng->nbNodeByCell(); nc++) {
        cells_infos.add(parent_nodes_uids[nc]);
      }

      // Partie Face.
      for (Integer l = 0; l < m_num_mng->nbFaceByCell(); ++l) {
        // On regarde si l'on doit traiter la face.
        // Si mask_face_if_cell_left[l] == false, on doit regarder si la maille à gauche est à nous ou non
        // ou si la maille à gauche est dans notre sous-domaine ou non.
        // Si cette maille n'est pas à nous et/ou n'est pas sur notre sous-domaine,
        // on doit créer la face en tant que face fantôme.
        if (
        (mask_face_if_cell_left[l] || is_cell_around_parent_cell_different_owner(1, 1, 0) || is_not_in_subdomain_cells_around_parent_cell(1, 1, 0)) &&
        (mask_face_if_cell_bottom[l] || is_cell_around_parent_cell_different_owner(1, 0, 1) || is_not_in_subdomain_cells_around_parent_cell(1, 0, 1)) &&
        (mask_face_if_cell_rear[l] || is_cell_around_parent_cell_different_owner(0, 1, 1) || is_not_in_subdomain_cells_around_parent_cell(0, 1, 1))) {
          Integer parent_face_owner = -1;
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
          // debug() << "Parent face (create face)  -- parent_cell_uid : " << parent_cell_uid
          //         << " -- level : " << -1
          //         << " -- face : " << l
          //         << " -- uid_face : " << parent_faces_uids[l]
          //         << " -- owner : " << parent_face_owner;
        }
      }

      // Partie Node.
      // Cette partie est assez ressemblante à la partie face, mis à part le fait qu'il peut y avoir
      // plus de propriétaires possibles.
      for (Integer l = 0; l < m_num_mng->nbNodeByCell(); ++l) {
        if (
        (mask_node_if_cell_left[l] || is_cell_around_parent_cell_different_owner(1, 1, 0) || is_not_in_subdomain_cells_around_parent_cell(1, 1, 0)) &&
        (mask_node_if_cell_bottom[l] || is_cell_around_parent_cell_different_owner(1, 0, 1) || is_not_in_subdomain_cells_around_parent_cell(1, 0, 1)) &&
        (mask_node_if_cell_rear[l] || is_cell_around_parent_cell_different_owner(0, 1, 1) || is_not_in_subdomain_cells_around_parent_cell(0, 1, 1))) {
          Integer parent_node_owner = -1;
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
          // debug() << "Parent node (create node)  -- parent_cell_uid : " << parent_cell_uid
          //         << " -- level : " << -1
          //         << " -- node : " << l
          //         << " -- uid_node : " << parent_nodes_uids[l]
          //         << " -- owner : " << parent_node_owner;
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
      // Ce tableau contiendra les localIds des nouveaux noeuds.
      UniqueArray<Int32> nodes_lid(total_nb_nodes);

      // On crée les noeuds. On met les localIds des nouveaux noeuds au début du tableau.
      m_mesh->modifier()->addNodes(nodes_infos, nodes_lid);

      UniqueArray<Int64> uid_child_nodes(total_nb_nodes);
      UniqueArray<Int32> lid_child_nodes(total_nb_nodes);
      Integer index = 0;

      // On attribue les bons propriétaires aux noeuds.
      ENUMERATE_ (Node, inode, m_mesh->nodeFamily()->view(nodes_lid)) {
        Node node = *inode;

        ARCANE_ASSERT((node_uid_to_owner.find(node.uniqueId()) != node_uid_to_owner.end()), ("No owner found for node"));
        ARCANE_ASSERT((node_uid_to_owner[node.uniqueId()] < nb_rank && node_uid_to_owner[node.uniqueId()] >= 0), ("Bad owner found for node"));

        node.mutableItemBase().setOwner(node_uid_to_owner[node.uniqueId()], my_rank);

        if (node_uid_to_owner[node.uniqueId()] == my_rank) {
          node.mutableItemBase().addFlags(ItemFlags::II_Own);
        }

        uid_child_nodes[index++] = m_num_mng->childNodeUniqueIdOfNode(node.uniqueId(), -1);
      }
      m_mesh->nodeFamily()->itemsUniqueIdToLocalId(lid_child_nodes, uid_child_nodes, false);
      NodeInfoListView nodes(m_mesh->nodeFamily());

      index = 0;
      ENUMERATE_ (Node, inode, m_mesh->nodeFamily()->view(nodes_lid)) {
        const Int32 child_lid = lid_child_nodes[index++];
        if (child_lid == NULL_ITEM_ID) {
          continue;
        }

        Node child = nodes[child_lid];
        Node parent = *inode;

        m_mesh->modifier()->addParentNodeToNode(child, parent);
        m_mesh->modifier()->addChildNodeToNode(parent, child);
      }
    }

    m_mesh->nodeFamily()->notifyItemsOwnerChanged();
  }

  // Faces
  {
    debug() << "Nb new faces in patch : " << total_nb_faces;
    {
      Integer nb_child = (m_mesh->dimension() == 2 ? 2 : 4);
      UniqueArray<Int32> faces_lid(total_nb_faces);

      m_mesh->modifier()->addFaces(total_nb_faces, faces_infos, faces_lid);

      UniqueArray<Int64> uid_child_faces(total_nb_faces * m_num_mng->nbFaceByCell());
      UniqueArray<Int32> lid_child_faces(total_nb_faces * m_num_mng->nbFaceByCell());
      Integer index = 0;

      // On attribue les bons propriétaires aux faces.
      ENUMERATE_ (Face, iface, m_mesh->faceFamily()->view(faces_lid)) {
        Face face = *iface;

        ARCANE_ASSERT((face_uid_to_owner.find(face.uniqueId()) != face_uid_to_owner.end()), ("No owner found for face"));
        ARCANE_ASSERT((face_uid_to_owner[face.uniqueId()] < nb_rank && face_uid_to_owner[face.uniqueId()] >= 0), ("Bad owner found for face"));

        face.mutableItemBase().setOwner(face_uid_to_owner[face.uniqueId()], my_rank);

        if (face_uid_to_owner[face.uniqueId()] == my_rank) {
          face.mutableItemBase().addFlags(ItemFlags::II_Own);
        }

        for (Integer i = 0; i < nb_child; ++i) {
          uid_child_faces[index++] = m_num_mng->childFaceUniqueIdOfFace(face.uniqueId(), -1, i);
        }
      }

      m_mesh->faceFamily()->itemsUniqueIdToLocalId(lid_child_faces, uid_child_faces, false);
      FaceInfoListView faces(m_mesh->faceFamily());

      index = 0;
      ENUMERATE_ (Face, iface, m_mesh->faceFamily()->view(faces_lid)) {
        for (Integer i = 0; i < nb_child; ++i) {
          const Int32 child_lid = lid_child_faces[index++];
          if (child_lid == NULL_ITEM_ID) {
            continue;
          }

          Face child = faces[child_lid];
          Face parent = *iface;

          m_mesh->modifier()->addParentFaceToFace(child, parent);
          m_mesh->modifier()->addChildFaceToFace(parent, child);
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
      if (parent_to_child_cells[parent.uniqueId()][0].itemBase().flags() & ItemFlags::II_Shared) {
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

  // On positionne les noeuds dans l'espace.
  CellInfoListView cells(m_mesh->cellFamily());
  for (Integer i = 0; i < total_nb_cells; ++i) {
    Cell parent_cell = cells[cells_lid[i]];
    m_num_mng->setParentNodeCoordinates(parent_cell);

    // On ajoute le flag "II_Shared" aux noeuds et aux faces des mailles partagées.
    if (parent_cell.mutableItemBase().flags() & ItemFlags::II_Shared) {
      for (Node node : parent_cell.nodes()) {
        if (node.mutableItemBase().flags() & ItemFlags::II_Own) {
          node.mutableItemBase().addFlags(ItemFlags::II_Shared);
        }
      }
      for (Face face : parent_cell.faces()) {
        if (face.mutableItemBase().flags() & ItemFlags::II_Own) {
          face.mutableItemBase().addFlags(ItemFlags::II_Shared);
        }
      }
    }
  }

  //! Créé le patch avec les mailles filles
  {
    CellGroup parent_cells = m_mesh->allLevelCells(0);
    m_cmesh->_internalApi()->addPatchFromExistingChildren(parent_cells.view().localIds());
  }

  // Recalcule les informations de synchronisation
  // Cela n'est pas nécessaire pour l'AMR car ces informations seront recalculées
  // lors du raffinement mais comme on ne sais pas si on va faire du raffinement
  // après il est préférable de calculer ces informations dans tous les cas.
  m_mesh->computeSynchronizeInfos();

  // Il faut recalculer les nouvelles directions après les modifications
  // et l'ajout de patch.
  m_cmesh->computeDirections();

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
  //  ENUMERATE_ (Cell, icell, m_mesh->allCells()) {
  //    debug() << "\tCell uniqueId : " << icell->uniqueId() << " -- level : " << icell->level() << " -- nbChildren : " << icell->nbHChildren();
  //    for (Integer i = 0; i < icell->nbHChildren(); ++i) {
  //      debug() << "\t\tChild uniqueId : " << icell->hChild(i).uniqueId() << " -- level : " << icell->hChild(i).level() << " -- nbChildren : " << icell->hChild(i).nbHChildren();
  //    }
  //  }
  //  info() << "Résumé node:";
  //  ENUMERATE_ (Node, inode, m_mesh->allNodes()) {
  //    debug() << "\tNode uniqueId : " << inode->uniqueId() << " -- level : " << inode->level() << " -- nbChildren : " << inode->nbHChildren();
  //    for (Integer i = 0; i < inode->nbHChildren(); ++i) {
  //      debug() << "\t\tNode Child uniqueId : " << inode->hChild(i).uniqueId() << " -- level : " << inode->hChild(i).level() << " -- nbChildren : " << inode->hChild(i).nbHChildren();
  //    }
  //  }
  //
  //  info() << "Résumé face:";
  //  ENUMERATE_ (Face, iface, m_mesh->allFaces()) {
  //    debug() << "\tFace uniqueId : " << iface->uniqueId() << " -- level : " << iface->level() << " -- nbChildren : " << iface->nbHChildren();
  //    for (Integer i = 0; i < iface->nbHChildren(); ++i) {
  //      debug() << "\t\tChild uniqueId : " << iface->hChild(i).uniqueId() << " -- level : " << iface->hChild(i).level() << " -- nbChildren : " << iface->hChild(i).nbHChildren();
  //    }
  //  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAMRPatchMng::
coarsen(bool update_parent_flag)
{
  // On commence par lister les mailles à ré-raffiner.
  UniqueArray<Cell> cells_to_coarsen_internal;
  ENUMERATE_ (Cell, icell, m_mesh->allActiveCells()) {
    Cell cell = *icell;
    if (cell.itemBase().flags() & ItemFlags::II_Coarsen) {
      if (cell.level() == 0) {
        ARCANE_FATAL("Cannot coarse level-0 cell");
      }

      Cell parent = cell.hParent();

      if (update_parent_flag) {
        parent.mutableItemBase().addFlags(ItemFlags::II_JustCoarsened);
        parent.mutableItemBase().removeFlags(ItemFlags::II_Inactive);
        parent.mutableItemBase().removeFlags(ItemFlags::II_CoarsenInactive);
      }

      // Pour une maille de niveau n-1, si une de ses mailles filles doit être dé-raffinée,
      // alors toutes ses mailles filles doivent être dé-raffinées.
      for (Integer i = 0; i < parent.nbHChildren(); ++i) {
        Cell child = parent.hChild(i);
        if (!(child.mutableItemBase().flags() & ItemFlags::II_Coarsen)) {
          ARCANE_FATAL("Parent cannot have children with coarse flag and children without coarse flag -- Parent uid: {0} -- Child uid: {1}", parent.uniqueId(), child.uniqueId());
        }
      }
      if (parent.mutableItemBase().flags() & ItemFlags::II_Coarsen) {
        ARCANE_FATAL("Cannot coarse parent and child in same time");
      }
      if (cell.nbHChildren() != 0) {
        ARCANE_FATAL("For now, cannot coarse cell with children");
      }
      cells_to_coarsen_internal.add(cell);
    }
  }

  // Maps remplaçant les mailles fantômes.
  std::unordered_map<Int64, Integer> around_cells_uid_to_owner;
  std::unordered_map<Int64, Int32> around_cells_uid_to_flags;

  {
    // On a uniquement besoin de ses deux flags pour les mailles autour.
    // (II_Coarsen pour savoir si les mailles autour sont aussi à dé-raffinées)
    // (II_Inactive pour savoir si les mailles autour sont déjà raffinées(pour vérifier qu'il n'y a pas plus d'un niveau d'écart))
    Int32 useful_flags = ItemFlags::II_Coarsen + ItemFlags::II_Inactive;
    _shareInfosOfCellsAroundPatch(cells_to_coarsen_internal, around_cells_uid_to_owner, around_cells_uid_to_flags, useful_flags);
  }

  // Avant de supprimer les mailles, on doit changer les propriétaires des faces/noeuds entre les mailles
  // à supprimer et les mailles restantes.
  if (m_mesh->dimension() == 2) {
    FixedArray<Int64, 9> uid_cells_around_cell_1d;
    FixedArray<Int32, 9> owner_cells_around_cell_1d;
    FixedArray<Int32, 9> flags_cells_around_cell_1d;

    for (Cell cell_to_coarsen : cells_to_coarsen_internal) {
      const Int64 cell_to_coarsen_uid = cell_to_coarsen.uniqueId();
      m_num_mng->cellUniqueIdsAroundCell(uid_cells_around_cell_1d.view(), cell_to_coarsen);

      {
        Integer nb_cells_to_coarsen_or_empty_around = 0;

        for (Integer i = 0; i < 9; ++i) {
          Int64 uid_cell = uid_cells_around_cell_1d[i];
          // Si uid_cell != -1 alors il y a peut-être une maille (mais on ne sait pas si elle est bien présente).
          // Si around_cells_uid_to_owner[uid_cell] != -1 alors il y a bien une maille.
          if (uid_cell != -1 && around_cells_uid_to_owner[uid_cell] != -1) {
            owner_cells_around_cell_1d[i] = around_cells_uid_to_owner[uid_cell];
            flags_cells_around_cell_1d[i] = around_cells_uid_to_flags[uid_cell];

            if (flags_cells_around_cell_1d[i] & ItemFlags::II_Coarsen) {
              nb_cells_to_coarsen_or_empty_around++;
            }
          }
          else {
            uid_cells_around_cell_1d[i] = -1;
            owner_cells_around_cell_1d[i] = -1;
            flags_cells_around_cell_1d[i] = 0;

            nb_cells_to_coarsen_or_empty_around++;
          }
        }

        // Si toutes les mailles autours de nous sont soit inexistantes, soit
        // à supprimer, inutile de chercher de nouveaux propriétaires pour nos items.
        // Notre maille est à dé-raffiner, donc nb_cells_to_coarsen_or_empty_around >= 1.
        if (nb_cells_to_coarsen_or_empty_around == 9) {
          continue;
        }
      }

      // Le propriétaire de notre maille.
      Int32 cell_to_coarsen_owner = owner_cells_around_cell_1d[4];

      {
        // On donne la position de la face dans le tableau des mailles autours
        // selon l'index de la face attribué par Arcane.
        // (Voir commentaire tagué "arcane_order_to_around_2d").
        // cell_to_coarsen.face(0) = uid_cells_around_cell_1d[1]
        // cell_to_coarsen.face(1) = uid_cells_around_cell_1d[5]
        // ...
        constexpr Integer arcane_order_to_pos_around[] = { 1, 5, 7, 3 };

        Integer count = -1;

        for (Face face : cell_to_coarsen.faces()) {
          count++;
          Int64 other_cell_uid = uid_cells_around_cell_1d[arcane_order_to_pos_around[count]];
          if (other_cell_uid == -1) {
            // On est au bord du maillage ou il n'y a pas de maille du même niveau à côté,
            // pas besoin de changer le owner de la face (elle sera supprimée).
            continue;
          }
          Int32 other_cell_flag = flags_cells_around_cell_1d[arcane_order_to_pos_around[count]];

          if (other_cell_flag & ItemFlags::II_Coarsen) {
            // La maille d'à côté sera aussi supprimée, pas besoin de changer le owner de la face.
            continue;
          }
          if (other_cell_flag & ItemFlags::II_Inactive) {
            // La maille d'à côté a des enfants. Il y aura donc au moins deux niveaux de
            // raffinements de différences.
            ARCANE_FATAL("Max one level diff between two cells is allowed -- Uid of Cell to be coarseing: {0} -- Uid of Opposite cell with children: {1}", cell_to_coarsen_uid, other_cell_uid);
          }
          Int32 other_cell_owner = owner_cells_around_cell_1d[arcane_order_to_pos_around[count]];
          if (other_cell_owner != cell_to_coarsen_owner) {
            // La maille d'à côté existe et appartient à quelqu'un d'autre. On lui donne la face.
            face.mutableItemBase().setOwner(other_cell_owner, cell_to_coarsen_owner);
          }
        }
      }

      {
        Integer count = -1;

        // Ici, plus compliqué.
        // Chaque élement du tableau de niveau 0 désigne un noeud.
        // Comme pour les faces, l'ordre est décrit au commentaire tagué "arcane_order_to_around_2d".
        //
        // Ensuite, par rapport aux faces, on a l'aspect priorité (comme notre maille
        // sera supprimé, pour chaque face, soit elle prendra la propriété de la maille d'à côté
        // si elle est "survivante", soit elle sera supprimée).
        // Un noeud est présent dans quatre mailles. Trois mailles seront potentiellement "survivantes".
        // On doit déterminer à qui appartiendra ce noeud parmi ces trois mailles.
        // Pour le savoir, on utilise les priorités décrites dans le commentaire
        // tagué "priority_owner_2d".
        //
        // Exemple : Le noeud n°0 est présent sur quatre mailles autour avec les priorités : P0, P1, P3, P4.
        //           La maille P4 (la "notre") sera supprimé.
        //           Dans le tableau n°0, on met les trois priorités (de la plus faible à la plus forte).
        //           Ensuite, on itére sur ces trois mailles (toujours de la plus faible à la plus forte).
        //           Si la maille i est "survivante", elle prend la propriété.
        //           Au final, la maille ayant la priorité la plus forte aura le noeud.
        //           Si aucune maille n'a pris la propriété, alors le noeud sera supprimée.
        //
        constexpr Integer priority_and_pos_of_cells_around_node[4][3] = { { 3, 1, 0 }, { 5, 2, 1 }, { 8, 7, 5 }, { 7, 6, 3 } };

        for (Node node : cell_to_coarsen.nodes()) {
          count++;
          Integer final_owner = -1;
          for (Integer other_cell = 0; other_cell < 3; ++other_cell) {
            Int64 other_cell_uid = uid_cells_around_cell_1d[priority_and_pos_of_cells_around_node[count][other_cell]];
            if (other_cell_uid == -1) {
              // On est au bord du maillage ou il n'y a pas de maille du même niveau à côté,
              // le noeud ne prendra pas son propriétaire.
              continue;
            }
            Int32 other_cell_flag = flags_cells_around_cell_1d[priority_and_pos_of_cells_around_node[count][other_cell]];

            if (other_cell_flag & ItemFlags::II_Coarsen) {
              // La maille d'à côté sera aussi supprimée, elle ne pourra pas prendre
              // la propriété de notre noeud.
              continue;
            }
            if (other_cell_flag & ItemFlags::II_Inactive) {
              // La maille d'à côté a des enfants. Il y aura donc au moins deux niveaux de
              // raffinements de différences.
              ARCANE_FATAL("Max one level diff between two cells is allowed -- Uid of Cell to be coarseing: {0} -- Uid of Opposite cell with children: {1}", cell_to_coarsen_uid, other_cell_uid);
            }
            Int32 other_cell_owner = owner_cells_around_cell_1d[priority_and_pos_of_cells_around_node[count][other_cell]];
            if (other_cell_owner != cell_to_coarsen_owner) {
              // La maille d'à côté existe et appartient à quelqu'un d'autre. On lui donne le noeud.
              final_owner = other_cell_owner;
            }
          }
          if (final_owner != -1) {
            node.mutableItemBase().setOwner(final_owner, cell_to_coarsen_owner);
          }
        }
      }
    }
  }
  else if (m_mesh->dimension() == 3) {
    FixedArray<Int64, 27> uid_cells_around_cell_1d;
    FixedArray<Int32, 27> owner_cells_around_cell_1d;
    FixedArray<Int32, 27> flags_cells_around_cell_1d;

    for (Cell cell_to_coarsen : cells_to_coarsen_internal) {
      const Int64 cell_to_coarsen_uid = cell_to_coarsen.uniqueId();
      m_num_mng->cellUniqueIdsAroundCell(uid_cells_around_cell_1d.view(), cell_to_coarsen);

      {
        Integer nb_cells_to_coarsen_or_empty_around = 0;

        for (Integer i = 0; i < 27; ++i) {
          Int64 uid_cell = uid_cells_around_cell_1d[i];
          // Si uid_cell != -1 alors il y a peut-être une maille (mais on ne sait pas si elle est bien présente).
          // Si around_cells_uid_to_owner[uid_cell] != -1 alors il y a bien une maille.
          if (uid_cell != -1 && around_cells_uid_to_owner[uid_cell] != -1) {
            owner_cells_around_cell_1d[i] = around_cells_uid_to_owner[uid_cell];
            flags_cells_around_cell_1d[i] = around_cells_uid_to_flags[uid_cell];

            if (flags_cells_around_cell_1d[i] & ItemFlags::II_Coarsen) {
              nb_cells_to_coarsen_or_empty_around++;
            }
          }
          else {
            uid_cells_around_cell_1d[i] = -1;
            owner_cells_around_cell_1d[i] = -1;
            flags_cells_around_cell_1d[i] = 0;

            nb_cells_to_coarsen_or_empty_around++;
          }
        }

        // Si toutes les mailles autours de nous sont soit inexistantes, soit
        // à supprimer, inutile de chercher de nouveaux propriétaires pour nos items.
        // Notre maille est à dé-raffiner, donc nb_cells_to_coarsen_or_empty_around >= 1.
        if (nb_cells_to_coarsen_or_empty_around == 27) {
          continue;
        }
      }

      // Le propriétaire de notre maille.
      Int32 cell_to_coarsen_owner = owner_cells_around_cell_1d[13];

      {
        // On donne la position de la face dans le tableau des mailles autours
        // selon l'index de la face attribué par Arcane.
        // (Voir commentaire tagué "arcane_order_to_around_3d").
        // cell_to_coarsen.face(0) = uid_cells_around_cell_1d[4]
        // cell_to_coarsen.face(1) = uid_cells_around_cell_1d[12]
        // ...
        constexpr Integer arcane_order_to_pos_around[] = { 4, 12, 10, 22, 14, 16 };

        Integer count = -1;

        for (Face face : cell_to_coarsen.faces()) {
          count++;
          Int64 other_cell_uid = uid_cells_around_cell_1d[arcane_order_to_pos_around[count]];
          if (other_cell_uid == -1) {
            // On est au bord du maillage ou il n'y a pas de maille du même niveau à côté,
            // pas besoin de changer le owner de la face (elle sera supprimée).
            continue;
          }
          Int32 other_cell_flag = flags_cells_around_cell_1d[arcane_order_to_pos_around[count]];

          if (other_cell_flag & ItemFlags::II_Coarsen) {
            // La maille d'à côté sera aussi supprimée, pas besoin de changer le owner de la face.
            continue;
          }
          if (other_cell_flag & ItemFlags::II_Inactive) {
            // La maille d'à côté a des enfants. Il y aura donc au moins deux niveaux de
            // raffinements de différences.
            ARCANE_FATAL("Max one level diff between two cells is allowed -- Uid of Cell to be coarseing: {0} -- Uid of Opposite cell with children: {1}", cell_to_coarsen_uid, other_cell_uid);
          }
          Int32 other_cell_owner = owner_cells_around_cell_1d[arcane_order_to_pos_around[count]];
          if (other_cell_owner != cell_to_coarsen_owner) {
            // La maille d'à côté existe et appartient à quelqu'un d'autre. On lui donne la face.
            face.mutableItemBase().setOwner(other_cell_owner, cell_to_coarsen_owner);
          }
        }
      }

      {
        Integer count = -1;

        // Chaque élement du tableau de niveau 0 désigne un noeud.
        // Comme pour les faces, l'ordre est décrit au commentaire tagué "arcane_order_to_around_3d".
        //
        // Ensuite, on a l'aspect priorité.
        // Un noeud est présent dans huit mailles. Sept mailles seront potentiellement "survivantes".
        // On doit déterminer à qui appartiendra ce noeud parmi ces sept mailles.
        // Pour le savoir, on utilise les priorités décrites dans le commentaire
        // tagué "priority_owner_3d".
        //
        // Exemple : Le noeud n°0 est présent sur huit mailles autour avec les priorités : P12, P10, P9, ...
        //           La maille P13 (la "notre") sera supprimé.
        //           Dans le tableau n°0, on met les sept priorités (de la plus faible à la plus forte).
        //           Ensuite, on itére sur ces sept mailles (toujours de la plus faible à la plus forte).
        //           Si la maille i est "survivante", elle prend la propriété.
        //           Au final, la maille ayant la priorité la plus forte aura le noeud.
        //           Si aucune maille n'a pris la propriété, alors le noeud sera supprimée.
        //
        constexpr Integer priority_and_pos_of_cells_around_node[8][7] = {
          {12, 10, 9, 4, 3, 1, 0},
          {14, 11, 10, 5, 4, 2, 1},
          {17, 16, 14, 8, 7, 5, 4},
          {16, 15, 12, 7, 6, 4, 3},
          {22, 21, 19, 18, 12, 10, 9},
          {23, 22, 20, 19, 14, 11, 10},
          {26, 25, 23, 22, 17, 16, 14},
          {25, 24, 22, 21, 16, 15, 12}
        };

        for (Node node : cell_to_coarsen.nodes()) {
          count++;
          Integer final_owner = -1;
          for (Integer other_cell = 0; other_cell < 7; ++other_cell) {
            Int64 other_cell_uid = uid_cells_around_cell_1d[priority_and_pos_of_cells_around_node[count][other_cell]];
            if (other_cell_uid == -1) {
              // On est au bord du maillage ou il n'y a pas de maille du même niveau à côté,
              // le noeud ne prendra pas son propriétaire.
              continue;
            }
            Int32 other_cell_flag = flags_cells_around_cell_1d[priority_and_pos_of_cells_around_node[count][other_cell]];

            if (other_cell_flag & ItemFlags::II_Coarsen) {
              // La maille d'à côté sera aussi supprimée, elle ne pourra pas prendre
              // la propriété de notre noeud.
              continue;
            }
            if (other_cell_flag & ItemFlags::II_Inactive) {
              // La maille d'à côté a des enfants. Il y aura donc au moins deux niveaux de
              // raffinements de différences.
              ARCANE_FATAL("Max one level diff between two cells is allowed -- Uid of Cell to be coarseing: {0} -- Uid of Opposite cell with children: {1}", cell_to_coarsen_uid, other_cell_uid);
            }
            Int32 other_cell_owner = owner_cells_around_cell_1d[priority_and_pos_of_cells_around_node[count][other_cell]];
            if (other_cell_owner != cell_to_coarsen_owner) {
              // La maille d'à côté existe et appartient à quelqu'un d'autre. On lui donne le noeud.
              node.mutableItemBase().setOwner(other_cell_owner, cell_to_coarsen_owner);
            }
          }
          if (final_owner != -1) {
            node.mutableItemBase().setOwner(final_owner, cell_to_coarsen_owner);
          }
        }
      }
    }
  }

  else {
    ARCANE_FATAL("Bad dimension");
  }

  UniqueArray<Int32> local_ids;
  for (Cell cell : cells_to_coarsen_internal) {
    local_ids.add(cell.localId());
  }
  m_mesh->modifier()->removeCells(local_ids);
  m_mesh->nodeFamily()->notifyItemsOwnerChanged();
  m_mesh->faceFamily()->notifyItemsOwnerChanged();
  m_mesh->modifier()->endUpdate();
  m_mesh->cellFamily()->computeSynchronizeInfos();
  m_mesh->nodeFamily()->computeSynchronizeInfos();
  m_mesh->faceFamily()->computeSynchronizeInfos();
  m_mesh->modifier()->setDynamic(true);

  UniqueArray<Int64> ghost_cell_to_refine;
  UniqueArray<Int64> ghost_cell_to_coarsen;

  if (!update_parent_flag) {
    // Si les matériaux sont actifs, il faut forcer un recalcul des matériaux car les groupes
    // de mailles ont été modifiés et donc la liste des constituants aussi
    Materials::IMeshMaterialMng* mm = Materials::IMeshMaterialMng::getReference(m_mesh, false);
    if (mm)
      mm->forceRecompute();
  }

  m_mesh->modifier()->updateGhostLayerFromParent(ghost_cell_to_refine, ghost_cell_to_coarsen, true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Méthode permettant d'obtenir les propriétaires et les flags des
 * mailles autour des mailles de patch_cells.
 *
 * \param patch_cells Les mailles autour desquelles on a besoin des infos des mailles autour.
 * \param around_cells_uid_to_owner Les propriétaires des mailles de patch_cells et des mailles autours des mailles de patch_cells.
 * \param around_cells_uid_to_flags Les flags des mailles de patch_cells et des mailles autours des mailles de patch_cells.
 * \param useful_flags Les flags qu'il est nécessaire de récupérer.
 */
void CartesianMeshAMRPatchMng::
_shareInfosOfCellsAroundPatch(ConstArrayView<Cell> patch_cells, std::unordered_map<Int64, Integer>& around_cells_uid_to_owner, std::unordered_map<Int64, Int32>& around_cells_uid_to_flags, Int32 useful_flags) const
{
  IParallelMng* pm = m_mesh->parallelMng();
  Int32 my_rank = pm->commRank();
  Int32 nb_rank = pm->commSize();

  // Partie échange d'informations sur les mailles autour du patch
  // (pour remplacer les mailles fantômes).

  // On remplit le tableau avec nos infos, pour les autres processus.
  ENUMERATE_ (Cell, icell, m_mesh->ownCells()) {
    Cell cell = *icell;
    around_cells_uid_to_owner[cell.uniqueId()] = my_rank;
    around_cells_uid_to_flags[cell.uniqueId()] = ((cell.itemBase().flags() & useful_flags) + ItemFlags::II_UserMark1);
  }

  ENUMERATE_ (Cell, icell, m_mesh->allCells().ghost()) {
    Cell cell = *icell;
    around_cells_uid_to_owner[cell.uniqueId()] = cell.owner();
    around_cells_uid_to_flags[cell.uniqueId()] = ((cell.itemBase().flags() & useful_flags) + ItemFlags::II_UserMark1);
  }

  // Tableau qui contiendra les uids des mailles dont on a besoin des infos.
  UniqueArray<Int64> uid_of_cells_needed;
  {
    UniqueArray<Int64> cell_uids_around((m_mesh->dimension() == 2) ? 9 : 27);
    for (Cell cell : patch_cells) {
      m_num_mng->cellUniqueIdsAroundCell(cell_uids_around, cell);
      for (Int64 cell_uid : cell_uids_around) {
        // Si -1 alors il n'y a pas de mailles à cette position.
        if (cell_uid == -1)
          continue;

        // TODO C++20 : Mettre map.contains().
        // SI on a la maille, on n'a pas besoin de demander d'infos.
        if (around_cells_uid_to_owner.find(cell_uid) != around_cells_uid_to_owner.end())
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
        flags_of_cells_needed_all_procs[compt] = (icell->itemBase().flags() & useful_flags);
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
      around_cells_uid_to_owner[uid_of_cells_needed[i]] = owner_of_cells_needed[i];
      around_cells_uid_to_flags[uid_of_cells_needed[i]] = flags_of_cells_needed[i];
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
