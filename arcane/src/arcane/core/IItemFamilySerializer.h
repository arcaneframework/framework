// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemFamilySerializer.h                                     (C) 2011-2024 */
/*                                                                           */
/* Manages the serialization/deserialization of entities in a family.        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IITEMFAMILYSERIALIZER_H
#define ARCANE_CORE_IITEMFAMILYSERIALIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

#include "arcane/core/ArcaneTypes.h"

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
class ARCANE_CORE_EXPORT IItemFamilySerializer
{
 public:

  virtual ~IItemFamilySerializer() = default;

 public:

  /*!
   * \brief Serializes the entities of the family \a family() into \a buf.
   *
   * In 'Put' or 'Reserve' mode, \a items contains the local cell numbers.
   * In 'Get' mode, it calls \a deserializeItems() and \a items is unused.
   */
  virtual void serializeItems(ISerializer* buf, Int32ConstArrayView items) = 0;

  /*!
   * \brief Deserializes the entities of the family \a family() from \a buf.
   *
   * If \a items_lid is not null, it contains the local numbers
   * of the deserialized cells in return.
   */
  virtual void deserializeItems(ISerializer* buf, Int32Array* items_lid) = 0;

  /*!
   * \brief Serializes the relations of the entities of the family \a family() into \a buf.
   *
   * In 'Put' or 'Reserve' mode, \a items contains the local cell numbers.
   * In 'Get' mode, it calls \a deserializeItemRelations() and \a items is unused.
   */
  virtual void serializeItemRelations(ISerializer* buf, Int32ConstArrayView items) = 0;

  /*!
   * \brief Deserializes the relations of the entities of the family \a family() from \a buf.
   *
   * If \a items_lid is not null, it contains the local numbers
   * of the cells whose relations have been deserialized in return.
   */
  virtual void deserializeItemRelations(ISerializer* buf, Int32Array* items_lid) = 0;

  //! Associated family
  virtual IItemFamily* family() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
