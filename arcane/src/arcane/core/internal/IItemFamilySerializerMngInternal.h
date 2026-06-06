// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemFamilySerializerMngInternal.h                          (C) 2000-2025 */
/*                                                                           */
/* Family serialization/deserialization tool manager.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IITEMFAMILYSERIALIZERMNGINTERNAL_H
#define ARCANE_CORE_IITEMFAMILYSERIALIZERMNGINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IItemFamily;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Manages the serialization/deserialization of entities in a family.
 */
class ARCANE_CORE_EXPORT IItemFamilySerializerMngInternal
{
 public:

  virtual ~IItemFamilySerializerMngInternal() = default;

 public:

  /*!
   * \brief Finalizes the allocations performed by the serializers registered in the manager.
   *
   * Used for polyhedral meshing where allocations are only performed after
   * all serializations for all families have been completed
   */
  virtual void finalizeItemAllocation() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
