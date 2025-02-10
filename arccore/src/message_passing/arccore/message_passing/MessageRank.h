// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MessageRank.h                                               (C) 2000-2025 */
/*                                                                           */
/* Rang d'un message.                                                        */
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
/*!
 * \brief Rang d'un message.
 *
 * Le type exact du rang dépend de l'implémentation. Pour être le plus
 * générique possible, on utilise le type 'Int32' qui est aussi celui
 * utilisé par MPI.
 *
 * Il existe trois valeurs spéciales pour le rang:
 * - la valeur par défaut
 * - la valeur procNullRank() qui correspond à MPI_PROC_NULL
 * - la valeur anySourceRang() qui correspond à MPI_ANY_SOURCE
 *
 * \sa PointToPointMessageInfo
 */
class ARCCORE_MESSAGEPASSING_EXPORT MessageRank
{
 public:

  /*!
   * \brief Rang par défaut.
   *
   * La signification du rang par défaut dépend du type de message.
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

  //! Valeur du rang
  Int32 value() const { return m_rank; }

  //! Positionne la valeur du rang
  void setValue(Int32 rank) { m_rank = rank; }

  //! Vrai si rang non initialisé correspondant au rang par défaut
  bool isNull() const { return m_rank == A_NULL_RANK; }

  //! Vrai si rang correspondant à anySourceRank()
  bool isAnySource() const { return m_rank == A_ANY_SOURCE_RANK; }

  //! Vrai si rang correspondant à procNullRank()
  bool isProcNull() const { return m_rank == A_PROC_NULL_RANK; }

  //! Rang correspondant à MPI_ANY_SOURCE
  static MessageRank anySourceRank() { return MessageRank(A_ANY_SOURCE_RANK); }

  //! Rang correspondant à MPI_PROC_NULL
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

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

