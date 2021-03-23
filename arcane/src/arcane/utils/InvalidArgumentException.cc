// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* InvalidArgumentException.cc                                 (C) 2000-2018 */
/*                                                                           */
/* Exception lorsqu'un argument est invalide.                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Iostream.h"
#include "arcane/utils/InvalidArgumentException.h"
#include "arcane/utils/OStringStream.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

InvalidArgumentException::
InvalidArgumentException(const TraceInfo& where,const String& arg_name,
                         int arg_value)
: Exception("InvalidArgument",where)
, m_arg_name(arg_name)
{
  _init(arg_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

InvalidArgumentException::
InvalidArgumentException(const TraceInfo& where,const String& arg_name,
                         double arg_value)
: Exception("InvalidArgument",where)
, m_arg_name(arg_name)
{
  _init(arg_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

InvalidArgumentException::
InvalidArgumentException(const TraceInfo& where,const String& arg_name,
                         const String& arg_value)
: Exception("InvalidArgument",where)
, m_arg_name(arg_name)
{
  _init(arg_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

InvalidArgumentException::
InvalidArgumentException(const TraceInfo& where,const String& arg_name,
                         const void* arg_value)
: Exception("InvalidArgument",where)
, m_arg_name(arg_name)
{
  _init(arg_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

InvalidArgumentException::
InvalidArgumentException(const TraceInfo& where,const String& arg_name,
                         const String& message,
                         int arg_value)
: Exception("InvalidArgument",where)
, m_arg_name(arg_name)
, m_message(message)
{
  _init(arg_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

InvalidArgumentException::
InvalidArgumentException(const TraceInfo& where,const String& arg_name,
                         const String& message,
                         double arg_value)
: Exception("InvalidArgument",where)
, m_arg_name(arg_name)
, m_message(message)
{
  _init(arg_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

InvalidArgumentException::
InvalidArgumentException(const TraceInfo& where,const String& arg_name,
                         const String& message,
                         const String& arg_value)
: Exception("InvalidArgument",where)
, m_arg_name(arg_name)
, m_message(message)
{
  _init(arg_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

InvalidArgumentException::
InvalidArgumentException(const TraceInfo& where,const String& arg_name,
                         const String& message,
                         const void* arg_value)
: Exception("InvalidArgument",where)
, m_arg_name(arg_name)
, m_message(message)
{
  _init(arg_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

InvalidArgumentException::
InvalidArgumentException(const InvalidArgumentException& ex)
: Exception(ex)
, m_arg_name(ex.m_arg_name)
, m_arg_value(ex.m_arg_value)
, m_message(ex.m_message)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename U> void InvalidArgumentException::
_init(const U& arg_value)
{
  OStringStream ostr;
  ostr() << arg_value;
  m_arg_value = ostr.str();
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void InvalidArgumentException::
explain(std::ostream& m) const
{
  m << "Argument invalide: nom='" << m_arg_name
    << "' valeur='" << m_arg_value << "'.";
  if (!m_message.null())
    m << m_message << ".";
  m << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
