/// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IGraph2.h                                                   (C) 2000-2025 */
/*                                                                           */
/* Interface d'un graphe d'un maillage.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IGRAPH2_H
#define ARCANE_CORE_IGRAPH2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/ItemTypes.h"
#include "arcane/core/IItemConnectivity.h"
#include "arcane/core/IndexedItemConnectivityView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Nombre de type d'entités duales
static const Integer NB_DUAL_ITEM_TYPE = 5;

extern "C++" ARCANE_CORE_EXPORT eItemKind
dualItemKind(Integer type);

class IGraphModifier2;
class IGraph2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Outillage de connectivité d'un graphe
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du gestionnaire de connectivité d'un graphe
 */
class ARCANE_CORE_EXPORT IGraphConnectivity
{
 public:

  virtual ~IGraphConnectivity() = default; //!< Libère les ressources

 public:

  //! accès à l'Item dual d'un DualNode (detype DoF)
  virtual Item dualItem(const DoF& dualNode) const = 0;

  //! accès à la vue des links  constitué du dualNode de type(DoF)
  virtual DoFVectorView links(const DoF& dualNode) const = 0;

  //! accès à la vue des DualNodes  constituant un liaison Link de type(DoF)
  virtual DoFVectorView dualNodes(const DoF& link) const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT IGraphConnectivityObserver
{
 public:

  virtual ~IGraphConnectivityObserver() = default;

 public:

  virtual void notifyUpdateConnectivity() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT IGraphObserver
{
 public:

  virtual ~IGraphObserver() {}

  virtual void notifyUpdate() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un graphe du maillage
 */
class ARCANE_CORE_EXPORT IGraph2
{
 public:

  virtual ~IGraph2() = default; //!< Libère les ressources

 public:

  virtual IGraphModifier2* modifier() = 0;

  virtual const IGraphConnectivity* connectivity() const = 0;

  virtual Integer registerNewGraphConnectivityObserver(IGraphConnectivityObserver* observer) = 0;

  virtual void releaseGraphConnectivityObserver(Integer observer_id) = 0;

  virtual Integer registerNewGraphObserver(IGraphObserver* observer) = 0;

  virtual void releaseGraphObserver(Integer observer_id) = 0;

  virtual bool isUpdated() = 0;

  //! Nombre de noeuds duaux du graphe
  virtual Integer nbDualNode() const = 0;

  //! Nombre de liaisons du graphe
  virtual Integer nbLink() const = 0;

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

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

