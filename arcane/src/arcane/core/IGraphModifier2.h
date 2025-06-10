// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IGraphModifier2.h                                           (C) 2000-2025 */
/*                                                                           */
/* Interface d'un outil de modification du graphe d'un maillage              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IGRAPHMODIFIER2_H
#define ARCANE_CORE_IGRAPHMODIFIER2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un graphe du maillage.
 */
class ARCANE_CORE_EXPORT IGraphModifier2
{
 public:

  virtual ~IGraphModifier2() = default; //!< Libère les ressources

 public:

  //! Ajout de liaisons dans le graphe avec un nombre fixe de noeuds dual par liaison
  virtual void addLinks(Integer nb_link,
                        Integer nb_dual_nodes_per_link,
                        Int64ConstArrayView links_infos) = 0;

  //! Ajout de noeuds duaux dans le graphe avec un type fixe d'item dual par noeud
  virtual void addDualNodes(Integer graph_nb_dual_node,
                            Integer dual_node_kind,
                            Int64ConstArrayView dual_nodes_infos) = 0;

  //! Ajout de noeuds duaux dans le graphe avec le type du noeud dans le tableau infos
  virtual void addDualNodes(Integer graph_nb_dual_node,
                            Int64ConstArrayView dual_nodes_infos) = 0;

  //! Suppression de noeuds duaux dans le graphe
  virtual void removeDualNodes(Int32ConstArrayView dual_node_local_ids) = 0;

  //! Suppression de liaisons duaux dans le graphe
  virtual void removeLinks(Int32ConstArrayView link_local_ids) = 0;

  //! Suppression des DualNodes et Links connectés aux mailles qui vont être supprimees
  virtual void removeConnectedItemsFromCells(Int32ConstArrayView cell_local_ids) = 0;

  virtual void endUpdate() = 0;

  virtual void updateAfterMeshChanged() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
