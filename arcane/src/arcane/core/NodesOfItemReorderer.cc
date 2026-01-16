// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NodesOfItemReorderer.cc                                     (C) 2000-2025 */
/*                                                                           */
/* Classe utilitaire pour réordonner les noeuds d'une entité.                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/NodesOfItemReorderer.h"

#include "arcane/utils/NotImplementedException.h"

#include "arcane/core/ItemTypeId.h"
#include "arcane/core/MeshUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// TODO: fusionner avec la version d'ordre 1
// Seulement implémenté pour les arêtes
bool NodesOfItemReorderer::
_reorderOrder3(ConstArrayView<Int64> nodes_uid,
               ArrayView<Int64> sorted_nodes_uid,
               [[maybe_unused]] bool has_center_node)
{
  // \a true s'il faut réorienter les faces pour que leur orientation
  // soit indépendante du partitionnement du maillage initial.

  Int32 nb_node = nodes_uid.size();

  // Traite uniquement le cas des arêtes d'ordre 3 qui ont donc 4 noeuds
  if (nb_node != 4)
    ARCANE_THROW(NotImplementedException, "Node reordering for 2D type of order 3 or more");

  if (nodes_uid[0] < nodes_uid[1]) {
    // Rien à faire
    sorted_nodes_uid[0] = nodes_uid[0];
    sorted_nodes_uid[1] = nodes_uid[1];
    sorted_nodes_uid[2] = nodes_uid[2];
    sorted_nodes_uid[3] = nodes_uid[3];
    return false;
  }
  sorted_nodes_uid[0] = nodes_uid[1];
  sorted_nodes_uid[1] = nodes_uid[0];
  sorted_nodes_uid[2] = nodes_uid[3];
  sorted_nodes_uid[3] = nodes_uid[2];
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// TODO: fusionner avec la version d'ordre 1
bool NodesOfItemReorderer::
_reorderOrder2(ConstArrayView<Int64> nodes_uid,
               ArrayView<Int64> sorted_nodes_uid, bool has_center_node)
{
  // \a true s'il faut réorienter les faces pour que leur orientation
  // soit indépendante du partitionnement du maillage initial.
  bool need_swap_orientation = false;
  Int32 min_node_index = 0;

  Int32 nb_node = nodes_uid.size();

  // Traite directement le cas des arêtes d'ordre 2
  if (nb_node == 3) {
    if (nodes_uid[0] < nodes_uid[1]) {
      sorted_nodes_uid[0] = nodes_uid[0];
      sorted_nodes_uid[1] = nodes_uid[1];
      sorted_nodes_uid[2] = nodes_uid[2];
      return false;
    }
    sorted_nodes_uid[0] = nodes_uid[1];
    sorted_nodes_uid[1] = nodes_uid[0];
    sorted_nodes_uid[2] = nodes_uid[2];
    return true;
  }
  // S'il y a un nœud central, c'est le dernier nœud de la liste
  // et il ne faut pas le trier
  // NOTE : Dans ce cas le nombre de noeuds de l'entité est impair.
  if (has_center_node)
    sorted_nodes_uid[nb_node - 1] = nodes_uid[nb_node - 1];

  // A l'ordre 2, si on a N noeuds, il ne faut tester les N/2 premiers noeuds
  // TODO: utiliser les informations de type.
  nb_node = nb_node / 2;

  // L'algorithme suivant oriente les faces en tenant compte uniquement
  // de l'ordre de la numérotation de ces noeuds. Si cet ordre est
  // conservé lors du partitionnement, alors l'orientation des faces
  // sera aussi conservée.

  // L'algorithme est le suivant:
  // - Recherche le noeud n de plus petit indice.
  // - Recherche n-1 et n+1 les indices de ses 2 noeuds voisins.
  // - Si (n+1) est inférieur à (n-1), l'orientation n'est pas modifiée.
  // - Si (n+1) est supérieur à (n-1), l'orientation est inversée.

  // Recherche le noeud de plus petit indice

  Int64 min_node = INT64_MAX;
  for (Integer k = 0; k < nb_node; ++k) {
    Int64 id = nodes_uid[k];
    if (id < min_node) {
      min_node = id;
      min_node_index = k;
    }
  }
  Int64 next_node = nodes_uid[(min_node_index + 1) % nb_node];
  Int64 prev_node = nodes_uid[(min_node_index + (nb_node - 1)) % nb_node];
  Integer incr = 0;
  Integer incr2 = 0;
  // Teste le cas où les noeuds précédents ou suivant
  // sont les mêmes que le noeud de plus petit uniqueId().
  // (dans ce cas l'entité est semi-dégénérée)
  {
    if (next_node == min_node) {
      next_node = nodes_uid[(min_node_index + (nb_node + 2)) % nb_node];
      incr = 1;
    }
    if (prev_node == min_node) {
      prev_node = nodes_uid[(min_node_index + (nb_node - 2)) % nb_node];
      incr2 = nb_node - 1;
    }
  }
  if (next_node > prev_node)
    need_swap_orientation = true;
  if (need_swap_orientation) {
    for (Integer k = 0; k < nb_node; ++k) {
      Integer index = (nb_node - k + min_node_index + incr) % nb_node;
      Int32 index2 = ((2*nb_node-1) + incr + min_node_index - k) % nb_node;
      sorted_nodes_uid[k] = nodes_uid[index];
      sorted_nodes_uid[k + nb_node] = nodes_uid[index2 + nb_node];
    }
  }
  else {
    for (Integer k = 0; k < nb_node; ++k) {
      Integer index = (k + min_node_index + incr2) % nb_node;
      sorted_nodes_uid[k] = nodes_uid[index];
      sorted_nodes_uid[k + nb_node] = nodes_uid[index + nb_node];
    }
  }
  return need_swap_orientation;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool NodesOfItemReorderer::
reorder(ItemTypeId type_id, ConstArrayView<Int64> nodes_uids)
{
  ItemTypeInfo* iti = m_item_type_mng->typeFromId(type_id);
  Int32 order = iti->order();
  Int32 nb_node = nodes_uids.size();
  m_work_sorted_nodes.resize(nb_node);
  if (order > 3)
    ARCANE_THROW(NotImplementedException, "Node reordering for type of order 4 or more");
  if (order == 3)
    return _reorderOrder3(nodes_uids, m_work_sorted_nodes, iti->hasCenterNode());
  if (order == 2)
    return _reorderOrder2(nodes_uids, m_work_sorted_nodes, iti->hasCenterNode());
  return MeshUtils::reorderNodesOfFace(nodes_uids, m_work_sorted_nodes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
