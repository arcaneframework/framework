// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConfigurationPropertyReader.h                               (C) 2000-2025 */
/*                                                                           */
/* Reading properties from an 'IConfiguration'.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_CONFIGURATIONPROPERTYREADER_H
#define ARCANE_CORE_INTERNAL_CONFIGURATIONPROPERTYREADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * NOTE: The classes in this file are currently under development.
 * NOTE: The API may change at any time. Do not use outside of Arcane.
 */

#include "arcane/core/IConfiguration.h"
#include "arccore/common/internal/Property.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::properties
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T>
class ConfigurationPropertyReader
: public PropertyVisitor<T>
{
 public:
  ConfigurationPropertyReader(IConfigurationSection* cs,T& instance)
  : m_configuration_section(cs), m_instance(instance){}
 private:
  IConfigurationSection* m_configuration_section;
  T& m_instance;
 public:
  void visit(const PropertySettingBase<T>& s) override
  {
    const String& pname = s.setting()->name();
    String value = m_configuration_section->value(pname,String());
    if (value.null())
      return;
    s.setFromString(value,m_instance);
    s.print(std::cout,m_instance);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Fills the values of \a instance from a configuration.
 *
 * The property values must be in a subsection
 * of \a c whose name is that of the class \a T.
 */
template<typename T> inline void
readFromConfiguration(IConfiguration* c,T& instance)
{
  if (!c)
    return;
  const char* instance_property_name = T :: propertyClassName();
  ScopedPtrT<IConfigurationSection> cs(c->createSection(instance_property_name));
  ConfigurationPropertyReader reader(cs.get(),instance);
  T :: applyPropertyVisitor(reader);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::properties

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
