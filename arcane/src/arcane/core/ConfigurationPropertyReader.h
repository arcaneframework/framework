// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConfigurationPropertyReader.h                               (C) 2000-2020 */
/*                                                                           */
/* Lecture de propriétés à partir d'un 'IConfiguration'.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_CONFIGURATIONPROPERTYREADER_H
#define ARCANE_CORE_CONFIGURATIONPROPERTYREADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * NOTE: Les classes de ce fichier sont en cours de mise au point.
 * NOTE: L'API peut changer à tout moment. Ne pas utiliser en dehors de Arcane.
 */

#include "arcane/core/IConfiguration.h"
#include "arcane/utils/Property.h"

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
 * \brief Remplit les valeurs de \a instance à partir d'une configuration.
 *
 * Les valeurs de la propriété doivent être dans une sous-section
 * de \a c dont le nom est celui de la classe \a T.
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
