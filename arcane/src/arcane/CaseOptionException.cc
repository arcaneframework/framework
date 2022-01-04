// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionException.cc                                      (C) 2000-2018 */
/*                                                                           */
/* Exception en rapport avec le jeu de données.                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Iostream.h"
#include "arcane/utils/StringBuilder.h"

#include "arcane/CaseOptionException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionException::
CaseOptionException(const String& where,const String& node_name,
                    const XmlNode& parent)
: Exception("CaseOptionException",where)
, m_node_name(node_name)
, m_parent(parent)
{
  StringBuilder sb;
  sb = "Configuration item:\n\n"
  "  <" + m_parent.xpathFullName() + String("/") + m_node_name;
  sb += ">\n\n"
  "can not be found.\n\n"
  "Make sure the configuration file is valid and up to date "
  "with the code.\n";
  m_message = sb.toString();
  setCollective(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionException::
CaseOptionException(const String& where,const String& node_name,
                    const XmlNode& parent,const String& value,
                    const String& type)
: Exception("CaseOptionException",where)
, m_node_name(node_name)
, m_parent(parent)
, m_value(value)
, m_type(type)
{
  StringBuilder sb;
  sb = "Configuration item:\n"
  "<" + m_parent.xpathFullName() + String("/") + m_node_name;
  sb += "> is not valid.\n"
  "Unable to cast character chain\n"
  "`" + m_value + "' to type <" + m_type + ">.\n";
  m_message = sb.toString();
  setCollective(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionException::
CaseOptionException(const String& where,const String& message,bool is_collective)
: Exception("CaseOptionException",where)
, m_message(message)
{
  setCollective(is_collective);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionException::
CaseOptionException(const TraceInfo& where,const String& message,bool is_collective)
: Exception("CaseOptionException",where)
, m_message(message)
{
  setCollective(is_collective);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionException::
CaseOptionException(const CaseOptionException& rhs) ARCANE_NOEXCEPT
: Exception(rhs)
, m_node_name(rhs.m_node_name)
, m_parent(rhs.m_parent)
, m_value(rhs.m_value)
, m_type(rhs.m_type)
, m_message(rhs.m_message)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionException::
~CaseOptionException() ARCANE_NOEXCEPT
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionException::
explain(std::ostream& m) const
{
  m << m_message << '\n';
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
