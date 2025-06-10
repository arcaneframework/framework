// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* XmlProperty.h                                               (C) 2000-2025 */
/*                                                                           */
/* Propriétés liée à un noeud XML.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_XMLPROPERTY_H
#define ARCANE_CORE_XMLPROPERTY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IProperty.h"
#include "arcane/core/XmlNode.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une valeur propriété.
 */
class XmlPropertyValue
: public IPropertyValue
{
 public:

  XmlPropertyValue(const XmlNode& node,IPropertyType* type,
		   IPropertyTypeInstance* type_instance);
  XmlPropertyValue();

 public:

  void valueToString(String& str) const override;
  void nameToString(String& str) const override;
  void setValueFromString(const String& str) override;
  bool isDefaultValue() const override;
  bool isOriginalValue() const override;
  void originalValueToString(String& str) const override;
  bool canBeEdited() const override;
  IPropertyType* type() override;
  IPropertyTypeInstance* typeInstance() override;

 public:

  XmlNode& node();
  void setNode(const XmlNode& node);
  void setType(IPropertyType* type);
  void setTypeInstance(IPropertyTypeInstance* type_instance);

 private:

  XmlNode m_node;
  IPropertyType* m_type = nullptr;
  IPropertyTypeInstance* m_type_instance = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
