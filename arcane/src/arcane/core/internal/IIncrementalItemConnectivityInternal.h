// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IIncrementalItemConnectivityInternal.h                      (C) 2000-2024 */
/*                                                                           */
/* Internal Arcane API for IncrementalItemConnectivity                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_IINCREMENTALITEMCONNECTIVITYINTERNAL_H
#define ARCANE_CORE_INTERNAL_IINCREMENTALITEMCONNECTIVITYINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

#include "arcane/core/ItemTypes.h"
#include "arcane/core/IItemConnectivityAccessor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Memory usage information for connectivities.
 */
class ItemConnectivityMemoryInfo
{
 public:

  //! Total number of Int32 used (corresponds to the sum of size())
  Int64 m_total_size = 0;
  //! Total number of Int32 allocated (corresponds to the sum of capacity())
  Int64 m_total_capacity = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Internal Arcane API for IIncrementalItemConnectivity.
 */
class ARCANE_CORE_EXPORT IIncrementalItemConnectivityInternal
{
 public:

  virtual ~IIncrementalItemConnectivityInternal() = default;

 public:

  //! Minimally reduces memory usage for connectivities
  virtual void shrinkMemory() = 0;

  //! Adds the instance's memory information to mem_info.
  virtual void addMemoryInfos(ItemConnectivityMemoryInfo& mem_info) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
