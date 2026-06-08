// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MessageRank.h                                               (C) 2000-2025 */
/*                                                                           */
/* Rank of a message.                                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_MESSAGERANK_H
#define ARCCORE_MESSAGEPASSING_MESSAGERANK_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Rank of a message.
 *
 * The exact type of the rank depends on the implementation. To be as
 * generic as possible, we use the 'Int32' type, which is also the one
 * used by MPI.
 *
 * There are three special values for the rank:
 * - the default value
 * - the procNullRank() value, which corresponds to MPI_PROC_NULL
 * - the anySourceRank() value, which corresponds to MPI_ANY_SOURCE
 *
 * \sa PointToPointMessageInfo
 */
class ARCCORE_MESSAGEPASSING_EXPORT MessageRank
{
 public:

  /*!
   * \brief Default rank.
   *
   * The meaning of the default rank depends on the message type.
   * \sa PointToPointMessageInfo.
   */
  MessageRank()
  : m_rank(A_NULL_RANK)
  {}

  explicit MessageRank(Int32 rank)
  : m_rank(rank)
  {}

  friend bool operator==(const MessageRank& a, const MessageRank& b)
  {
    return a.m_rank == b.m_rank;
  }
  friend bool operator!=(const MessageRank& a, const MessageRank& b)
  {
    return a.m_rank != b.m_rank;
  }
  friend bool operator<(const MessageRank& a, const MessageRank& b)
  {
    return a.m_rank < b.m_rank;
  }

  //! Rank value
  Int32 value() const { return m_rank; }

  //! Sets the rank value
  void setValue(Int32 rank) { m_rank = rank; }

  //! True if the rank is uninitialized, corresponding to the default rank
  bool isNull() const { return m_rank == A_NULL_RANK; }

  //! True if the rank corresponds to anySourceRank()
  bool isAnySource() const { return m_rank == A_ANY_SOURCE_RANK; }

  //! True if the rank corresponds to procNullRank()
  bool isProcNull() const { return m_rank == A_PROC_NULL_RANK; }

  //! Rank corresponding to MPI_ANY_SOURCE
  static MessageRank anySourceRank() { return MessageRank(A_ANY_SOURCE_RANK); }

  //! Rank corresponding to MPI_PROC_NULL
  static MessageRank procNullRank() { return MessageRank(A_PROC_NULL_RANK); }

  void print(std::ostream& o) const;
  friend inline std::ostream& operator<<(std::ostream& o, const MessageRank& tag)
  {
    tag.print(o);
    return o;
  }

 private:

  Int32 m_rank;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
