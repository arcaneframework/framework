/// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IGraph2.h                                                    (C) 2011-2011 */
/*                                                                           */
/* Interface d'un graphe d'un maillage    .                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IGRAPH2_H
#define ARCANE_IGRAPH2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/ArcaneTypes.h"
#include "arcane/ItemTypes.h"
#include "arcane/IItemConnectivity.h"
#include "arcane/IndexedItemConnectivityView.h"
#include "arcane/mesh/ItemConnectivity.h"
#include "arcane/mesh/IncrementalItemConnectivity.h"
#include "arcane/mesh/IndexedItemConnectivityAccessor.h"
//#include <typeinfo>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Nombre de type d'entit�s duales
static const Integer NB_DUAL_ITEM_TYPE = 5;

extern "C++" ARCANE_CORE_EXPORT eItemKind
dualItemKind(Integer type);

class IGraphModifier2;
class IGraph2 ;
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Outillage de connectivit� d'un graphe
 */


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du gestionnaire de connectivité d'un graphe
 */
class ARCANE_CORE_EXPORT IGraphConnectivity
{
 public :
  virtual ~IGraphConnectivity() {} //<! Lib�re les ressources

  //! accès à l'Item dual d'un DualNode (detype DoF)
  virtual Item dualItem(const DoF& dualNode) const = 0 ;

  //! accès à la vue des DualNodes  constituant un liaison Link de type(DoF)
  virtual DoFVectorView dualNodes(const DoF& link) const = 0 ;
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un graphe du maillage
 */
class ARCANE_CORE_EXPORT IGraph2
{
public:

  virtual ~IGraph2() {} //<! Lib�re les ressources

public:

  virtual IGraphModifier2* modifier() = 0 ;

  virtual const IGraphConnectivity* connectivity() const = 0 ;


public:

  //! Nombre de noeuds duaux du graphe
  virtual Integer nbDualNode() const =0;

  //! Nombre de liaisons du graphe
  virtual Integer nbLink() const =0;

public:

  //! Retourne la famille des noeuds duaux
  virtual const IItemFamily* dualNodeFamily() const = 0;
  virtual IItemFamily* dualNodeFamily() = 0;

  //! Retourne la famille des liaisons
  virtual const IItemFamily* linkFamily() const = 0;
  virtual IItemFamily* linkFamily() = 0;

  virtual void printDualNodes() const = 0;
  virtual void printLinks() const = 0;

};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
