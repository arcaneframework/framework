// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* XmlProperty.h                                               (C) 2000-2002 */
/*                                                                           */
/* Propriétés liée à un noeud XML.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_XMLPROPERTY_H
#define ARCANE_XMLPROPERTY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/IProperty.h"

#include "arcane/XmlNode.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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

  virtual ~XmlPropertyValue() {} //!< Libère les ressources

 public:

  virtual void valueToString(String& str) const;
  virtual void nameToString(String& str) const;
  virtual void setValueFromString(const String& str);
  virtual bool isDefaultValue() const;
  virtual bool isOriginalValue() const;
  virtual void originalValueToString(String& str) const;
  virtual bool canBeEdited() const;
  virtual IPropertyType* type();
  virtual IPropertyTypeInstance* typeInstance();

 public:

  XmlNode& node();
  void setNode(const XmlNode& node);
  void setType(IPropertyType* type);
  void setTypeInstance(IPropertyTypeInstance* type_instance);

 private:

  XmlNode   m_node;
  IPropertyType* m_type;
  IPropertyTypeInstance* m_type_instance;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

