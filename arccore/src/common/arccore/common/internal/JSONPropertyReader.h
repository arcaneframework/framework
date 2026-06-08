// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* JSONPropertyReader.h                                        (C) 2000-2025 */
/*                                                                           */
/* Reading properties in JSON format.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_INTERNAL_JSONPROPERTYREADER_H
#define ARCANE_UTILS_INTERNAL_JSONPROPERTYREADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * NOTE: The classes in this file are under development.
 * NOTE: The API may change at any time. Do not use outside of Arcane.
 */

#include "arccore/common/JSONReader.h"
#include "arccore/common/internal/Property.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::properties
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T>
class JSONPropertyReader
: public PropertyVisitor<T>
{
 public:
  JSONPropertyReader(JSONValue jv,T& instance)
  : m_jv(jv), m_instance(instance){}
 private:
  JSONValue m_jv;
  T& m_instance;
 public:
  void visit(const PropertySettingBase<T>& s) override
  {
    JSONValue child_value = m_jv.child(s.setting()->name());
    if (child_value.null())
      return;
    s.setFromJSON(child_value,m_instance);
    s.print(std::cout,m_instance);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Fills the values of \a instance from the JSON element \a jv.
 *
 * The property values must be in a child element of \a jv
 * whose name is that of the class \a T.
 */
template<typename T, typename PropertyType = T> inline void
readFromJSON(JSONValue jv,T& instance)
{
  const char* instance_property_name = PropertyType :: propertyClassName();
  JSONValue child_value = jv.child(instance_property_name);
  if (child_value.null())
    return;
  JSONPropertyReader reader(child_value,instance);
  PropertyType :: applyPropertyVisitor(reader);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::properties

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
