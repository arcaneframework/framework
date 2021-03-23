// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IGraph.h                                                    (C) 2011-2011 */
/*                                                                           */
/* Interface d'un graphe d'un maillage    .                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IGRAPH_H
#define ARCANE_IGRAPH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"
#include "arcane/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IGraphModifier;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un graphe du maillage
 */
class IGraph
{
public:

  virtual ~IGraph() {} //<! Libère les ressources
  
public:
  
  virtual IGraphModifier* modifier() =0;

public:
  
  //! Nombre de noeuds duaux du graphe
  virtual Integer nbDualNode() =0;
  
  //! Nombre de liaisons du graphe
  virtual Integer nbLink() =0;
  
public:
  
  //! Groupe de tous les noeuds duaux
  virtual DualNodeGroup allDualNodes() =0;

  //! Groupe de toutes les liaisons
  virtual LinkGroup allLinks() =0;

  //! Groupe de tous les noeuds duaux propres au domaine
  virtual DualNodeGroup ownDualNodes() =0;

  //! Groupe de toutes les liaisons propres au domaine
  virtual LinkGroup ownLinks() =0;

  //! Retourne la famille des noeuds duaux
  virtual IItemFamily* dualNodeFamily() =0;
  
  //! Retourne la famille des liaisons
  virtual IItemFamily* linkFamily() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
