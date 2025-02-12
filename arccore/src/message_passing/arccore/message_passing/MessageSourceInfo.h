// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MessageSourceInfo.h                                         (C) 2000-2025 */
/*                                                                           */
/* Informations sur la source d'un message.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_MESSAGESOURCEINFO_H
#define ARCCORE_MESSAGEPASSING_MESSAGESOURCEINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessageTag.h"
#include "arccore/message_passing/MessageRank.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations sur la source d'un message.
 *
 * Ces informations sont utilisées pour récupérer les informations d'un
 * message suite à un appel à mpProbe() ou mpLegacyProbe().
 * L'instance retournée peut-être utilisée pour faire une réception via mpReceive().
 */
class ARCCORE_MESSAGEPASSING_EXPORT MessageSourceInfo
{
 public:

  //! Créé une source correspondant à aucun message (isValid()==false)
  MessageSourceInfo() = default;

  /*!
   * \brief Créé une source correspondant au rang \a rank et au tag \a tag.
   *
   * Si \a rank.isNull() ou tag.isNull(), alors isValid() vaudra \a false.
   */
  MessageSourceInfo(MessageRank rank, MessageTag tag, Int64 size)
  : m_rank(rank)
  , m_tag(tag)
  , m_size(size)
  {}

 public:

  //! Rang de la source
  MessageRank rank() const { return m_rank; }

  //! Positionne le rang de la source
  void setRank(MessageRank rank) { m_rank = rank; }

  //! Tag du message
  MessageTag tag() const { return m_tag; }

  //! Positionne le tag du message
  void setTag(MessageTag tag) { m_tag = tag; }

  //! Taille du message
  Int64 size() const { return m_size; }

  //! Positionne la taille du message
  void setSize(Int64 size) { m_size = size; }

  //! Indique si la source est valide
  bool isValid() const { return !m_rank.isNull() && !m_tag.isNull(); }

 private:

  MessageRank m_rank;
  MessageTag m_tag;
  Int64 m_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

