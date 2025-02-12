// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TraceMessage.cc                                             (C) 2000-2025 */
/*                                                                           */
/* Gestion des messages.                                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/trace/ITraceMng.h"
#include "arccore/base/String.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TraceMessage::
TraceMessage(std::ostream* ostr,ITraceMng* m,Trace::eMessageType id,int level)
: m_stream(ostr)
, m_parent(m)
, m_type(id)
, m_level(level)
, m_color(0)
{
  if (m_parent)
    m_parent->beginTrace(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TraceMessage::
TraceMessage(const TraceMessage& from)
: m_stream(from.m_stream)
, m_parent(from.m_parent)
, m_type(from.m_type)
, m_level(from.m_level)
, m_color(from.m_color)
{
  if (m_parent)
    m_parent->beginTrace(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const TraceMessage& TraceMessage::
operator=(const TraceMessage& from)
{
  ITraceMng* from_parent = from.parent();
  if (from_parent)
    from_parent->beginTrace(&from);
  if (m_parent)
    m_parent->endTrace(this);
  m_stream = from.m_stream;
  m_parent = from_parent;
  m_type = from.m_type;
  m_level = from.m_level;
  m_color = from.m_color;
  return *this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// NOTE: ce destructeur peut envoyer une exception si le message est
// de type Fatal ou ParallelFatal. Avec le C++11, il faut le signaler
// explicitement sinon le code se termine via un appel à std::terminate().
// A noter qu'avec gcc 4.7 et avant, même avec l'option std=c++11 il n'est
// pas nécessaire de spécifier le noexcept.
TraceMessage::
~TraceMessage() ARCCORE_NOEXCEPT_FALSE
{
  if (m_parent)
    m_parent->endTrace(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const TraceMessage& TraceMessage::
width(Integer v) const
{
  m_stream->width(v);
  return *this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream& TraceMessage::
file() const
{
  return *m_stream;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//static int zeroi = 0;
std::ostream& Trace::
operator<< (std::ostream& o,const Trace::Width& w)
{
  o.width(w.m_width);
  //if (zeroi==1)
  //o.flags(ios::left|ios::scientific);
  //++zeroi;
  return o;
}

std::ostream& Trace::
operator<< (std::ostream& o,const Trace::Precision& w)
{
  bool is_scientific = w.m_scientific;
  std::ios::fmtflags old_flags = o.flags();
  if (is_scientific){
    o.flags(std::ios::scientific);
  }
  std::streamsize p = o.precision(w.m_precision);
  o << w.m_value;
  o.precision(p);
  if (is_scientific)
    o.flags(old_flags);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const TraceMessage&
operator<<(const TraceMessage& o,const Trace::Color& c)
{
  o.m_color = c.m_color;
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Trace::Setter::
Setter(ITraceMng* msg,const String& name)
: m_msg(msg)
{
  m_msg->pushTraceClass(name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Trace::Setter::
~Setter()
{
  m_msg->popTraceClass();
  m_msg->flush();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

