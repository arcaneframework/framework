﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IGraphModifier2.h                                            (C) 2011-2011 */
/*                                                                           */
/* Interface d'un outil de modification du graphe d'un maillage              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IGRAPHMODIFIER2_H
#define ARCANE_IGRAPHMODIFIER2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un graphe du maillage
 */
class ARCANE_CORE_EXPORT IGraphModifier2
{
 public:

  virtual ~IGraphModifier2() {} //<! Lib�re les ressources
    
 public:
  
 
  //! Ajout de liaisons dans le graphe avec un nombre fixe de noeuds dual par liaison
  virtual void addLinks(Integer nb_link,
                        Integer nb_dual_nodes_per_link,
                        Int64ConstArrayView links_infos) =0;

  //! Ajout de noeuds duaux dans le graphe avec un type fixe d'item dual par noeud
  virtual void addDualNodes(Integer graph_nb_dual_node,
                            Integer dual_node_kind,
                            Int64ConstArrayView dual_nodes_infos) = 0;

  //! Ajout de noeuds duaux dans le graphe avec le type du noeud dans le tableau infos
  virtual void addDualNodes(Integer graph_nb_dual_node,
                            Int64ConstArrayView dual_nodes_infos) = 0;

  //! Suppression de noeuds duaux dans le graphe
  virtual void removeDualNodes(Int32ConstArrayView dual_node_local_ids) =0;

  //! Suppression de liaisons duaux dans le graphe
  virtual void removeLinks(Int32ConstArrayView link_local_ids) =0;
  

  virtual void endUpdate() =0;

  virtual void updateAfterMeshChanged() =0;

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#endif

