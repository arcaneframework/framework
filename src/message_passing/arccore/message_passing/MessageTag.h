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
/* MessageTag.h                                                (C) 2000-2020 */
/*                                                                           */
/* Tag d'un message.                                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_MESSAGETAG_H
#define ARCCORE_MESSAGEPASSING_MESSAGETAG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
{
/*!
 * \brief Tag d'un message.
 *
 * Le type exact du tag dépend de l'implémentation. Pour être le plus
 * générique possible, on utilise le type 'Int32' qui est aussi le type
 * utilisé couramment avec MPI.
 *
 * Avec l'implémentation MPI, ce type est utilisé pour le tag MPI et
 * les valeurs maximales autorisées dépendent de l'implémentation. La norme
 * MPI indique seulement qu'il faut au moins autoriser 2^30 (32767) valeurs.
 *
 * En mode échange de message hybride (MPI + mémoire partagée), la valeur
 * maximale du tag peut être plus faible. Pour toutes ces raisons, il
 * est conseillé de ne pas dépasser la valeur 4096.
 */
class ARCCORE_MESSAGEPASSING_EXPORT MessageTag
{
 public:
  MessageTag() : m_tag(A_NULL_TAG_VALUE){}
  explicit MessageTag(Int32 tag) : m_tag(tag){}
  friend bool operator==(const MessageTag& a,const MessageTag& b)
  {
    return a.m_tag==b.m_tag;
  }
  friend bool operator!=(const MessageTag& a,const MessageTag& b)
  {
    return a.m_tag!=b.m_tag;
  }
  friend bool operator<(const MessageTag& a,const MessageTag& b)
  {
    return a.m_tag<b.m_tag;
  }
  Int32 value() const { return m_tag; }
  bool isNull() const { return m_tag==A_NULL_TAG_VALUE; }
  void print(std::ostream& o) const;
  friend inline std::ostream&
  operator<<(std::ostream& o,const MessageTag& tag)
  {
    tag.print(o);
    return o;
  }
 public:
  //! Valeur par défaut du tag.
  static constexpr Int32 DEFAULT_TAG_VALUE = 100;
  //! Tag par défaut pour les send/receive sans argument tag.
  static MessageTag defaultTag() { return MessageTag(DEFAULT_TAG_VALUE); }
 private:
  Int32 m_tag;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

