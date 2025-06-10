// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFamilySerializeArgs.h                                   (C) 2000-2025 */
/*                                                                           */
/* Arguments des callbacks de sérialisation des familles d'entités.          */
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
 * \brief Arguments des callbacks de sérialisation des familles d'entités.
 *
 * La signification des arguments dépend du mode de sérialisation.
 *
 * En mode ISerializer::ModeReserve ou ISerializer::ModePut:
 * - rank() est le rang de la famille de destination
 * - localIds() contient les indices locaux des entités qui seront envoyés
 *   à la famille destination.
 *
 * En mode ISerializer::ModeGet:
 * - rank() est le rang de la famille d'origine.
 * - localIds() contient les indices locaux des entités qu'on vient de recevoir.
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

  //! Sérialiseur associé
  ISerializer* serializer() const { return m_serializer; }

  /*!
   * \brief Rang de la source ou de la destination.
   *
   * En sérialisation, il s'agit du rang de la destination et en
   * désérialisation il s'agit du rang de la source.
   */
  Int32 rank() const { return m_rank; }

  /*!
   * \brief Indices locaux des entités.
   * En sérialisation, il s'agit des indices locaux des entités envoyées au
   * rang \a rank(). En désérialisation, il s'agit des indices locaux
   * recues par le rang \a rank().
   */
  Int32ConstArrayView localIds() const { return m_local_ids; }

  //! Indice du message dans la liste des messages
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
