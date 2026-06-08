// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TraceMessage.h                                              (C) 2000-2025 */
/*                                                                           */
/* Trace message.                                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_TRACE_TRACEMESSAGE_H
#define ARCCORE_TRACE_TRACEMESSAGE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/trace/Trace.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Message handling.
 *
 This class is managed like a standard output stream (ostream&) and
 allows sending a message of the type specified by #eTraceMessageClass.
 
 \warning Instances of this class are normally created by
 an ITraceMng message manager.
*/
class ARCCORE_TRACE_EXPORT TraceMessage
{
 public:
  static const int DEFAULT_LEVEL = Trace::DEFAULT_VERBOSITY_LEVEL;
 public:
  TraceMessage(std::ostream*,ITraceMng*,Trace::eMessageType,int level =DEFAULT_LEVEL);
  TraceMessage(const TraceMessage& from);
  const TraceMessage& operator=(const TraceMessage& from);
  ~TraceMessage() ARCCORE_NOEXCEPT_FALSE;

 public:
  std::ostream& file() const;
  const TraceMessage& width(Integer v) const;
  ITraceMng* parent() const { return m_parent; }
  Trace::eMessageType type() const { return m_type; }
  int level() const { return m_level; }
  int color() const { return m_color; }
 private:
  std::ostream* m_stream; //!< Stream on which the message is sent
  ITraceMng* m_parent; //!< Parent message manager
  Trace::eMessageType m_type; //!< Message type
  int m_level; //!< Message level
 public:
  mutable int m_color; //!< Message color.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Trace
{
ARCCORE_TRACE_EXPORT std::ostream&
operator<< (std::ostream& o,const Width& w);

ARCCORE_TRACE_EXPORT std::ostream&
operator<< (std::ostream& o,const Precision& w);
}

#ifndef ARCCORE_DEBUG
class ARCCORE_TRACE_EXPORT TraceMessageDbg
{
};
template<class T> inline const TraceMessageDbg&
operator<<(const TraceMessageDbg& o,const T&)
{
  return o;
}
#endif

ARCCORE_TRACE_EXPORT const TraceMessage&
operator<<(const TraceMessage& o,const Trace::Color& c);

template<class T> inline const TraceMessage&
operator<<(const TraceMessage& o,const T& v)
{
  o.file() << v;
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
