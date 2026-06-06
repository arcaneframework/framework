// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemFamilyPolicyMng.h                                      (C) 2000-2025 */
/*                                                                           */
/* Policies of an entity family.                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IITEMFAMILYPOLICYMNG_H
#define ARCANE_CORE_IITEMFAMILYPOLICYMNG_H
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
 * \brief Interface for entity family policies.
 */
class ARCANE_CORE_EXPORT IItemFamilyPolicyMng
{
 public:

  virtual ~IItemFamilyPolicyMng() = default;

 public:

  //! Compaction policy
  virtual IItemFamilyCompactPolicy* compactPolicy() = 0;
  /*!
   * \brief Creates an instance for exchanging entities between subdomains.
   * The returned instance must be destroyed by the delete operator.
   */
  virtual IItemFamilyExchanger* createExchanger() = 0;

  /*!
   * \brief Creates an instance for entity serialization.
   * The returned instance must be destroyed by the delete operator.
   *
   * \a with_flags indicates whether the value of Item::flags() should be serialized.
   * This is not necessarily supported for all families.
   */
  virtual IItemFamilySerializer* createSerializer(bool with_flags = false) = 0;

  /*!
   * \brief Adds a factory for a serialization step.
   *
   * \a factory remains the property of the caller and must not be destroyed
   * as long as this instance exists.
   */
  virtual void addSerializeStep(IItemFamilySerializeStepFactory* factory) = 0;

  //! Removes a factory for a serialization step.
  virtual void removeSerializeStep(IItemFamilySerializeStepFactory* factory) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
