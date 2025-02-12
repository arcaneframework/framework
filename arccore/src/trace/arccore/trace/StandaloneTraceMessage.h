// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StandaloneTraceMessage.h                                    (C) 2000-2025 */
/*                                                                           */
/* Message de trace indépendant du 'ITraceMng'.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_TRACE_STANDALONETRACEMESSAGE_H
#define ARCCORE_TRACE_STANDALONETRACEMESSAGE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/trace/TraceMessage.h"

#include <sstream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestion d'un message autonome.
 */
class ARCCORE_TRACE_EXPORT StandaloneTraceMessage
{
 public:
  StandaloneTraceMessage() = default;
  StandaloneTraceMessage(Trace::eMessageType,int level = TraceMessage::DEFAULT_LEVEL);
  StandaloneTraceMessage(const TraceMessage& from);
  StandaloneTraceMessage& operator=(const StandaloneTraceMessage& from);

 public:
  std::ostream& file() const { return m_stream; }
  Trace::eMessageType type() const { return m_type; }
  int level() const { return m_level; }
  int color() const { return m_color; }
 public:
  std::string value() const { return m_stream.str(); }
 private:
  //! Flot sur lequel le message est envoyé
  mutable std::ostringstream m_stream;
  //! Type de message
  Trace::eMessageType m_type = Trace::Normal;
  //! Niveau du message
  int m_level = TraceMessage::DEFAULT_LEVEL;
 public:
  //! Couleur du message.
  mutable int m_color = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline const StandaloneTraceMessage&
operator<<(const StandaloneTraceMessage& o,const Trace::Color& c)
{
  o.m_color = c.m_color;
  return o;
}

template<class T> inline const StandaloneTraceMessage&
operator<<(const StandaloneTraceMessage& o,const T& v)
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

