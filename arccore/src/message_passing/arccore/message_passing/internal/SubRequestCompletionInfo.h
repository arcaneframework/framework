// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SubRequestCompletionInfo.h                                  (C) 2000-2025 */
/*                                                                           */
/* Completion information for a sub-request.                                 */
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
 * \brief Completion information for a sub-request.
 */
class ARCCORE_MESSAGEPASSING_EXPORT SubRequestCompletionInfo
{
 public:

  SubRequestCompletionInfo(MessageRank rank, MessageTag tag)
  : m_source_rank(rank)
  , m_source_tag(tag)
  {}

 public:

  //! Source rank of the request
  MessageRank sourceRank() const { return m_source_rank; }
  //! Source tag of the request
  MessageTag sourceTag() const { return m_source_tag; }

 private:

  MessageRank m_source_rank;
  MessageTag m_source_tag;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
