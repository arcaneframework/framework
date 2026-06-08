// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MessageTag.h                                                (C) 2000-2025 */
/*                                                                           */
/* Message tag.                                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_MESSAGETAG_H
#define ARCCORE_MESSAGEPASSING_MESSAGETAG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"

#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*!
 * \brief Message tag.
 *
 * The exact type of the tag depends on the implementation. To be as
 * generic as possible, we use the 'Int32' type, which is also the type
 * commonly used with MPI.
 *
 * With the MPI implementation, this type is used for the MPI tag, and
 * the maximum allowed values depend on the implementation. The MPI standard
 * only indicates that at least 2^30 (32767) values must be allowed.
 *
 * In hybrid message exchange mode (MPI + shared memory), the maximum
 * tag value may be lower. For all these reasons, it
 * is recommended not to exceed the value 4096.
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
  //! Default tag value.
  static constexpr Int32 DEFAULT_TAG_VALUE = 100;
  //! Default tag for send/receive without tag argument.
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
