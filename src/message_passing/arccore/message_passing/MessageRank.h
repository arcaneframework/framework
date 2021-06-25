// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MessageRank.h                                               (C) 2000-2020 */
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

namespace Arccore::MessagePassing
{
/*!
 * \brief Rang d'un message.
 *
 * Le type exact du rang dépend de l'implémentation. Pour être le plus
 * générique possible, on utilise le type 'Int32' qui est aussi celui
 * utilisé par MPI.
 */
class ARCCORE_MESSAGEPASSING_EXPORT MessageRank
{
 public:
  MessageRank() : m_rank(A_NULL_RANK){}
  explicit MessageRank(Int32 rank) : m_rank(rank){}
  friend bool operator==(const MessageRank& a,const MessageRank& b)
  {
    return a.m_rank==b.m_rank;
  }
  friend bool operator!=(const MessageRank& a,const MessageRank& b)
  {
    return a.m_rank!=b.m_rank;
  }
  friend bool operator<(const MessageRank& a,const MessageRank& b)
  {
    return a.m_rank<b.m_rank;
  }
  Int32 value() const { return m_rank; }
  void setValue(Int32 rank) { m_rank = rank; }
  bool isNull() const { return m_rank==A_NULL_RANK; }
  void print(std::ostream& o) const;
  friend inline std::ostream& operator<<(std::ostream& o,const MessageRank& tag)
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

