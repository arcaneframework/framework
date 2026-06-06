// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemFamilyCompactPolicy.h                                  (C) 2000-2025 */
/*                                                                           */
/* Interface for the compaction policy of family entities.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IITEMFAMILYCOMPACTPOLICY_H
#define ARCANE_CORE_IITEMFAMILYCOMPACTPOLICY_H
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
 * \brief Entity compaction policy.
 *
 * An instance of this class is associated with each family.
 *
 * The call pseudo-code for a compaction is as follows:
 *
 \code
 * IMesh* mesh = ...;
 * IMeshCompacter* compacter = ...;
 * ItemFamilyCollection families = mesh->itemFamilies();
 * UniqueArray<ItemFamilyCompactPolicy> policies;
 * for( IItemFamily* family : mesh->itemFamilies() )
 *   policies.add( createCompactPolicity(family) );
 * for( ItemFamilyCompactPolicity* policy : policies)
 *   policy->beginCompact(...);
 * for( ItemFamilyCompactPolicity* policy : policies)
 *   policy->compactVariablesAndGroups(...);
 * for( ItemFamilyCompactPolicity* policy : policies)
 *   policy->updateInternalReferences(compacter);
 * for( ItemFamilyCompactPolicity* policy : policies)
 *   policy->endCompact(...);
 \endcode
 *
 * Outside of a compaction, it is possible to call
 * compactReferenceData(), which allows compacting the data used
 * to hold connectivity information.
 */
class ARCANE_CORE_EXPORT IItemFamilyCompactPolicy
{
 public:

  virtual ~IItemFamilyCompactPolicy() = default;

 public:

  // TODO: implement a computeCompact method after beginCompact().
  virtual void beginCompact(ItemFamilyCompactInfos& compact_infos) = 0;
  virtual void compactVariablesAndGroups(const ItemFamilyCompactInfos& compact_infos) = 0;
  virtual void updateInternalReferences(IMeshCompacter* compacter) = 0;
  virtual void endCompact(ItemFamilyCompactInfos& compact_infos) = 0;
  virtual void finalizeCompact(IMeshCompacter* compacter) = 0;
  //! Associated family
  virtual IItemFamily* family() const = 0;
  //! Compacts the connectivity data.
  virtual void compactConnectivityData() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
