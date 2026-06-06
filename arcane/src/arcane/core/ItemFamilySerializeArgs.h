// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFamilySerializeArgs.h                                   (C) 2000-2025 */
/*                                                                           */
/* Arguments for the serialization callbacks of entity families.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMFAMILYSERIALIZEARGS_H
#define ARCANE_CORE_ITEMFAMILYSERIALIZEARGS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Mesh
 * \brief Arguments for the serialization callbacks of entity families.
 *
 * The meaning of the arguments depends on the serialization mode.
 *
 * In ISerializer::ModeReserve or ISerializer::ModePut mode:
 * - rank() is the rank of the destination family
 * - localIds() contains the local indices of the entities that will be sent
 *   to the destination family.
 *
 * In ISerializer::ModeGet mode:
 * - rank() is the rank of the source family.
 * - localIds() contains the local indices of the entities that have just
 * been received.
 *
 */
class ARCANE_CORE_EXPORT ItemFamilySerializeArgs
{
 public:

  ItemFamilySerializeArgs(ISerializer* aserializer, Int32 arank,
                          Int32ConstArrayView local_ids, Integer message_index)
  : m_serializer(aserializer)
  , m_rank(arank)
  , m_local_ids(local_ids)
  , m_message_index(message_index)
  {}

 public:

  //! Associated serializer
  ISerializer* serializer() const { return m_serializer; }

  /*!
   * \brief Rank of the source or destination.
   *
   * During serialization, it is the rank of the destination, and during
   * deserialization it is the rank of the source.
   */
  Int32 rank() const { return m_rank; }

  /*!
   * \brief Local indices of the entities.
   * During serialization, these are the local indices of the entities sent to
   * rank \a rank(). During deserialization, these are the local indices
   * received by rank \a rank().
   */
  Int32ConstArrayView localIds() const { return m_local_ids; }

  //! Message index in the list of messages
  Integer messageIndex() const { return m_message_index; }

 public:

  ISerializer* m_serializer = nullptr;
  Int32 m_rank = A_NULL_RANK;
  Int32ConstArrayView m_local_ids;
  Integer m_message_index = -1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
