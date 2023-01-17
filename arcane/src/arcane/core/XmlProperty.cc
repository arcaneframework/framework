// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* XmlProperty.cc                                              (C) 2000-2005 */
/*                                                                           */
/* Propriétés liée à un noeud XML.                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/XmlProperty.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
  
XmlPropertyValue::
XmlPropertyValue(const XmlNode& node,IPropertyType* type,
		 IPropertyTypeInstance* type_instance)
: m_node(node)
, m_type(type)
, m_type_instance(type_instance)
{
}

XmlPropertyValue::
XmlPropertyValue()
: m_node(0)
, m_type(0)
, m_type_instance(0)
{
}

void XmlPropertyValue::
valueToString(String& str) const
{
  str = m_node.value();
}

void XmlPropertyValue::
nameToString(String& str) const
{
  str = m_node.name();
}

void XmlPropertyValue::
setValueFromString(const String& str)
{
  m_node.setValue(str);
}

bool XmlPropertyValue::
isDefaultValue() const
{
  return false;
}

bool XmlPropertyValue::
isOriginalValue() const
{
  return false;
}

void XmlPropertyValue::
originalValueToString(String& str) const
{
  str = String();
}

bool XmlPropertyValue::
canBeEdited() const
{
  return true;
}

IPropertyType* XmlPropertyValue::
type()
{
  return m_type;
}

IPropertyTypeInstance* XmlPropertyValue::
typeInstance()
{
  return m_type_instance;
}

void XmlPropertyValue::
setType(IPropertyType* type)
{
  m_type = type;
}

void XmlPropertyValue::
setTypeInstance(IPropertyTypeInstance* type_instance)
{
  m_type_instance = type_instance;
}

XmlNode& XmlPropertyValue::
node()
{
  return m_node;
}

void XmlPropertyValue::
setNode(const XmlNode& node)
{
  m_node = node;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

