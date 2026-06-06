// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IIndexedIncrementalItemConnectivityMng.h                    (C) 2000-2025 */
/*                                                                           */
/* Interface of the 'IIndexedIncrementalItemConnectivity' manager.           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IINDEXEDINCREMENTALITEMCONNECTIVITYMNG_H
#define ARCANE_CORE_IINDEXEDINCREMENTALITEMCONNECTIVITYMNG_H
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
 * \brief Interface of the manager for indexed incremental item connectivities.
 */
class ARCANE_CORE_EXPORT IIndexedIncrementalItemConnectivityMng
{
 public:

  virtual ~IIndexedIncrementalItemConnectivityMng() = default;

 public:

  /*!
   * \brief Searches for or creates a connectivity.
   *
   * Throws an exception if a connectivity with name \a name already exists but
   * not with the same pair (source, target).
   * The instance remains the owner of the returned connectivity.
   */
  virtual Ref<IIndexedIncrementalItemConnectivity>
  findOrCreateConnectivity(IItemFamily* source, IItemFamily* target, const String& name) = 0;

  /*!
   * \brief Searches for or creates a connectivity.
   *
   * Throws an exception if the connectivity with name \a name is not found.
   * The instance remains the owner of the returned connectivity.
   */
  virtual Ref<IIndexedIncrementalItemConnectivity>
  findConnectivity(const String& name) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
