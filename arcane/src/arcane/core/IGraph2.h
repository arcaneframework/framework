/// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IGraph2.h                                                   (C) 2000-2025 */
/*                                                                           */
/* Mesh graph interface.                                                     */
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

//! Number of dual entity types
static const Integer NB_DUAL_ITEM_TYPE = 5;

extern "C++" ARCANE_CORE_EXPORT eItemKind
dualItemKind(Integer type);

class IGraphModifier2;
class IGraph2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Graph connectivity tooling
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of the graph connectivity manager
 */
class ARCANE_CORE_EXPORT IGraphConnectivity
{
 public:

  virtual ~IGraphConnectivity() = default; //!< Frees resources

 public:

  //! Access to the dual Item of a DualNode (detype DoF)
  virtual Item dualItem(const DoF& dualNode) const = 0;

  //! Access to the view of links composed of the dualNode of type(DoF)
  virtual DoFVectorView links(const DoF& dualNode) const = 0;

  //! Access to the view of DualNodes constituting a Link of type(DoF)
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
 * \brief Mesh graph interface
 */
class ARCANE_CORE_EXPORT IGraph2
{
 public:

  virtual ~IGraph2() = default; //!< Frees resources

 public:

  virtual IGraphModifier2* modifier() = 0;

  virtual const IGraphConnectivity* connectivity() const = 0;

  virtual Integer registerNewGraphConnectivityObserver(IGraphConnectivityObserver* observer) = 0;

  virtual void releaseGraphConnectivityObserver(Integer observer_id) = 0;

  virtual Integer registerNewGraphObserver(IGraphObserver* observer) = 0;

  virtual void releaseGraphObserver(Integer observer_id) = 0;

  virtual bool isUpdated() = 0;

  //! Number of dual nodes of the graph
  virtual Integer nbDualNode() const = 0;

  //! Number of links of the graph
  virtual Integer nbLink() const = 0;

 public:

  //! Returns the family of dual nodes
  virtual const IItemFamily* dualNodeFamily() const = 0;
  virtual IItemFamily* dualNodeFamily() = 0;

  //! Returns the family of links
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
