// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* NotImplementedException.cc                                  (C) 2000-2018 */
/*                                                                           */
/* Exception lorsqu'une fonction n'est pas implémentée.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/String.h"
#include "arccore/base/NotImplementedException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NotImplementedException::
NotImplementedException(const String& where)
: Exception("NotImplemented",where)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NotImplementedException::
NotImplementedException(const String& where,const String& message)
: Exception("NotImplemented",where)
, m_message(message)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NotImplementedException::
NotImplementedException(const TraceInfo& where)
: Exception("NotImplemented",where)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NotImplementedException::
NotImplementedException(const TraceInfo& where,const String& message)
: Exception("NotImplemented",where)
, m_message(message)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NotImplementedException::
NotImplementedException(const NotImplementedException& rhs) ARCCORE_NOEXCEPT
: Exception(rhs)
, m_message(rhs.m_message)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NotImplementedException::
explain(std::ostream& m) const
{
  m << "function not implemented.";

  if (!m_message.null())
    m << "Message: " << m_message << '\n';
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

