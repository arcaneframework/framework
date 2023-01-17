// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionError.cc                                          (C) 2000-2012 */
/*                                                                           */
/* Erreur dans le jeu de données.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Iostream.h"
#include "arcane/utils/StringBuilder.h"

#include "arcane/CaseOptionError.h"
#include "arcane/ICaseDocument.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionError::
CaseOptionError(const TraceInfo& where,const String& node_name,
                const String& message,bool is_collective)
: m_func_info(where)
, m_node_name(node_name)
, m_message(message)
, m_is_collective(is_collective)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionError::
addOptionNotFoundError(ICaseDocument* document,
                       const TraceInfo& where,
                       const String& node_name,
                       const XmlNode& parent)
{
  String full_node_name = parent.xpathFullName() + "/" + node_name;
  String message = "Element or attribute missing";

  document->addError(CaseOptionError(where,full_node_name,message,true));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionError::
addInvalidTypeError(ICaseDocument* document,
                    const TraceInfo& where,const String& node_name,
                    const XmlNode& parent,const String& value,
                    const String& type_name)
{
  String full_node_name = parent.xpathFullName() + "/" + node_name;

  String message = String::format("Invalid value. Impossible to convert the string"
                                  " '{0}' to the type '{1}'.",value,type_name);

  document->addError(CaseOptionError(where,full_node_name,message,true));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionError::
addInvalidTypeError(ICaseDocument* document,
                    const TraceInfo& where,const String& node_name,
                    const XmlNode& parent,const String& value,
                    const String& type_name,StringConstArrayView valid_values)
{
  String full_node_name = parent.xpathFullName() + " /" + node_name;

  String message = String::format("Invalid value. Impossible to convert the string"
                                  " '{0}' to the type '{1}' (admissible values: {2}). ",value,
                                  type_name,String::join(", ",valid_values));
  message = message + "\nSpaces at the beginning and the end of a string matter.\nPlease check that there is none.";
  document->addError(CaseOptionError(where,full_node_name,message,true));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionError::
addError(ICaseDocument* document,
         const TraceInfo& where,const String& node_name,
         const String& message,bool is_collective)
{
  document->addError(CaseOptionError(where,node_name,message,is_collective));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionError::
addWarning(ICaseDocument* document,
           const TraceInfo& where,const String& node_name,
           const String& message,bool is_collective)
{
  document->addWarning(CaseOptionError(where,node_name,message,is_collective));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
