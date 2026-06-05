// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConfigurationReader.cc                                      (C) 2000-2023 */
/*                                                                           */
/* Configuration file readers.                                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/ConfigurationReader.h"
#include "arcane/utils/JSONReader.h"
#include "arcane/core/IConfiguration.h"
#include "arcane/core/XmlNode.h"
#include "arcane/core/XmlNodeList.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConfigurationReader::
addValuesFromXmlNode(const XmlNode& root_elem, Integer priority)
{
  UniqueArray<String> all_names;

  XmlNodeList sections = root_elem.children("add");
  for (Integer i = 0, n = sections.size(); i < n; ++i) {
    XmlNode sec_node = sections[i];

    String sec_name = sec_node.attrValue("name");
    String sec_value = sec_node.attrValue("value");

    if (sec_name.null()) {
      pwarning() << "Missing 'name' attribute in <section> in configuration file";
      continue;
    }

    if (sec_value.null()) {
      pwarning() << "Missing 'value' attribute in <section> in configuration file";
      continue;
    }
    //info() << "GET CONFIG name=" << sec_name << " value=" << sec_value;

    all_names.clear();
    sec_name.split(all_names, '.');
    m_configuration->addValue(sec_name, sec_value, priority);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConfigurationReader::
_addValuesFromJSON(const JSONValue& jv, Integer priority, const String& base_name)
{
  for (JSONKeyValue v : jv.keyValueChildren()) {
    String name = v.name();
    JSONValue value = v.value();
    if (value.isObject()) {
      _addValuesFromJSON(value, priority, base_name + name + ".");
    }
    else if (value.isArray()) {
      // Does not process arrays for now because they are
      // not supported in the configuration.
    }
    else {
      String v_value = value.value();
      //info() << "B=" << base_name << " N=" << name << " V=" << v_value;
      m_configuration->addValue(base_name + name, v_value, priority);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConfigurationReader::
addValuesFromJSON(const JSONValue& jv, Integer priority)
{
  _addValuesFromJSON(jv, priority, String());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
