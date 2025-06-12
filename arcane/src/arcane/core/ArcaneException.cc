// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneException.cc                                          (C) 2000-2025 */
/*                                                                           */
/* Exceptions lancées par l'architecture.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneException.h"

#include "arcane/utils/Iostream.h"
#include "arcane/utils/TraceInfo.h"

#include "arcane/core/IVariable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BadIDException::
BadIDException(const String& where,const String& invalid_name)
: Exception("BadID",where)
, m_invalid_name(invalid_name)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BadIDException::
explain(std::ostream& m) const
{
  m << "Name '" << m_invalid_name << "' is not a valid identifier.\n";
  m << "Identifiers must start with an alphabetical character (a-zA-Z)\n";
  m << "followed by alphabetical characters, figures,\n";
  m << "underscores '_', dots '.' or hyphen '-'.\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BadItemIdException::
BadItemIdException(const String& where,Integer bad_id)
: Exception("BadItemId",where)
, m_bad_id(bad_id)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BadItemIdException::
explain(std::ostream& m) const
{
  m << "Trying to use invalid item identifier: " << m_bad_id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

InternalErrorException::
InternalErrorException(const String& where,const String& why)
: Exception("InternalError",where)
, m_why(why)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

InternalErrorException::
InternalErrorException(const TraceInfo& where,const String& why)
: Exception("InternalError",where)
, m_why(why)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

InternalErrorException::
InternalErrorException(const InternalErrorException& ex) ARCANE_NOEXCEPT
: Exception(ex)
, m_why(ex.m_why)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void InternalErrorException::
explain(std::ostream& m) const
{
  m << "Internal error ocurred:\n";
  m << "'" << m_why << "'\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BadVariableKindTypeException::
BadVariableKindTypeException(const TraceInfo& where,IVariable* prv,
                      eItemKind kind,eDataType data_type,int dimension)
: Exception("BadVariableKindType",where)
, m_valid_var(prv)
, m_item_kind(kind)
, m_data_type(data_type)
, m_dimension(dimension)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BadVariableKindTypeException::
explain(std::ostream& m) const
{
  m << "Wrong variable type:\n"
    << "Variable '" << m_valid_var->name() << "'\n";
  m << "declared as '" << m_item_kind << '.' << m_data_type << "." << m_dimension
    << "' has already been declared\n"
    << "as '" << m_valid_var->itemKind() << '.' << m_valid_var->dataType()
    << "." << m_valid_var->dimension()
    << "'";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BadPartialVariableItemGroupNameException::
BadPartialVariableItemGroupNameException(const TraceInfo& where,IVariable* prv,
                                         const String& item_group_name)
: Exception("BadVariableKindType",where)
, m_valid_var(prv)
, m_item_group_name(item_group_name)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BadPartialVariableItemGroupNameException::
explain(std::ostream& m) const
{
  m << "Wrong partial variable type:\n"
    << "Partial Variable '" << m_valid_var->name() << "'\n";
  m << "declared on item group '" << m_item_group_name
    << "' has already been declared\n"
    << "on item group '" << m_valid_var->itemGroupName()
    << "'";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

UnknownItemTypeException::
UnknownItemTypeException(const String& where,Integer nb_node,Integer item_id)
: Exception("UnknownItemType",where)
, m_nb_node(nb_node)
, m_item_id(item_id)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

UnknownItemTypeException::
UnknownItemTypeException(const UnknownItemTypeException& ex) ARCANE_NOEXCEPT
: Exception(ex)
, m_nb_node(ex.m_nb_node)
, m_item_id(ex.m_item_id)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UnknownItemTypeException::
explain(std::ostream& m) const
{
  m << "Item number <" << m_item_id << "> with " << m_nb_node << " nodes\n"
    << "is not a known type.\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BadReferenceException::
BadReferenceException(const String& where)
: Exception("BadReference",where)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BadReferenceException::
explain(std::ostream& m) const
{
  m << "Trying to dereference a null pointer.\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ReaderWriterException::
ReaderWriterException(const String& where,const String& message)
: Exception("ReaderWriterException",where, message)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ReaderWriterException::
ReaderWriterException(const TraceInfo& where,const String& message)
: Exception("ReaderWriterException",where, message)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ReaderWriterException::
ReaderWriterException(const ReaderWriterException& ex) ARCANE_NOEXCEPT
: Exception(ex)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ReaderWriterException::
explain(std::ostream& m) const
{
  m << "Exception reading/writing.\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AssertionException::
AssertionException(const TraceInfo& where)
: Exception("AssertionException", where)
, m_file(where.file())
, m_line(where.line())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AssertionException::
AssertionException(const TraceInfo& where, const String& expected, const String& actual)
: Exception("ReaderWriterException", where, "Actual : " + actual + ". Expected : " + expected + ".")
, m_file(where.file())
, m_line(where.line())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AssertionException::
explain(std::ostream& m) const
{
  m << "Assertion failed.\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
