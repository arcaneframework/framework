// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SubRequestCompletionInfo.h                                  (C) 2000-2025 */
/*                                                                           */
/* Informations de complétion pour une sous-requête.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_INTERNAL_SUBREQUESTCOMPLETIONINFO_H
#define ARCCORE_MESSAGEPASSING_INTERNAL_SUBREQUESTCOMPLETIONINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessageRank.h"
#include "arccore/message_passing/MessageTag.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations de complètion d'une sous-requête.
 */
class ARCCORE_MESSAGEPASSING_EXPORT SubRequestCompletionInfo
{
 public:

  SubRequestCompletionInfo(MessageRank rank, MessageTag tag)
  : m_source_rank(rank)
  , m_source_tag(tag)
  {}

 public:

  //! Rang d'origine de la requête
  MessageRank sourceRank() const { return m_source_rank; }
  //! Tag d'origine de la requête
  MessageTag sourceTag() const { return m_source_tag; }

 private:

  MessageRank m_source_rank;
  MessageTag m_source_tag;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

