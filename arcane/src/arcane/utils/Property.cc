// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Property.cc                                                 (C) 2000-2025 */
/*                                                                           */
/* Gestion des propriétés.                                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/internal/Property.h"

#include "arcane/utils/Ref.h"
#include "arcane/utils/JSONReader.h"
#include "arcane/utils/List.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/CheckedConvert.h"

#include <cstdlib>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::properties
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

auto PropertySettingTraits<String>::
fromJSON(const JSONValue& jv) -> InputType
{
  return jv.value();
}

auto PropertySettingTraits<String>::
fromString(const String& v) -> InputType
{
  return v;
}

void PropertySettingTraits<String>::
print(std::ostream& o,InputType v)
{
  o << v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

auto PropertySettingTraits<StringList>::
fromJSON(const JSONValue& jv) -> InputType
{
  StringList string_list;
  for(JSONValue jv2 : jv.valueAsArray())
    string_list.add(jv2.value());
  return string_list;
}

auto PropertySettingTraits<StringList>::
fromString(const String& v) -> InputType
{
  StringList string_list;
  string_list.add(v);
  return string_list;
}

void PropertySettingTraits<StringList>::
print(std::ostream& o,StringCollection v)
{
  bool is_not_first = false;
  for( String x : v ){
    if (is_not_first)
      o << ',';
    o << x;
    is_not_first = true;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

auto PropertySettingTraits<bool>::
fromJSON(const JSONValue& jv) -> InputType
{
  return jv.valueAsInt64()!=0;
}

auto PropertySettingTraits<bool>::
fromString(const String& v) -> InputType
{
  if (v=="0" || v=="false")
    return false;
  if (v=="1" || v=="true")
    return true;
  ARCANE_FATAL("Can not convert '{0}' to type bool "
               "(valid values are '0', '1', 'true', 'false')",v);
}

void PropertySettingTraits<bool>::
print(std::ostream& o,InputType v)
{
  o << v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

auto PropertySettingTraits<Int64>::
fromJSON(const JSONValue& jv) -> InputType
{
  return jv.valueAsInt64();
}

auto PropertySettingTraits<Int64>::
fromString(const String& v) -> InputType
{
  Int64 read_value = 0;
  bool is_bad = builtInGetValue(read_value,v);
  if (is_bad)
    ARCANE_FATAL("Can not convert '{0}' to type 'Int64' ",v);
  return read_value;
}

void PropertySettingTraits<Int64>::
print(std::ostream& o,InputType v)
{
  o << v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

auto PropertySettingTraits<Int32>::
fromJSON(const JSONValue& jv) -> InputType
{
  return CheckedConvert::toInt32(jv.valueAsInt64());
}

auto PropertySettingTraits<Int32>::
fromString(const String& v) -> InputType
{
  Int32 read_value = 0;
  bool is_bad = builtInGetValue(read_value,v);
  if (is_bad)
    ARCANE_FATAL("Can not convert '{0}' to type 'Int32' ",v);
  return read_value;
}

void PropertySettingTraits<Int32>::
print(std::ostream& o,InputType v)
{
  o << v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
PropertySettingsRegisterer* global_arcane_first_registerer = nullptr;
Integer global_arcane_nb_registerer = 0;
}

PropertySettingsRegisterer::
PropertySettingsRegisterer(CreateFunc func,CreateBuildInfoFunc, const char* name) ARCANE_NOEXCEPT
: m_name(name)
, m_create_func(func)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PropertySettingsRegisterer::
_init()
{
  if (global_arcane_first_registerer==nullptr){
    global_arcane_first_registerer = this;
    _setPreviousRegisterer(nullptr);
    _setNextRegisterer(nullptr);
  }
  else{
    auto* next = global_arcane_first_registerer->nextRegisterer();
    _setNextRegisterer(global_arcane_first_registerer); 
    global_arcane_first_registerer = this;
    if (next)
      next->_setPreviousRegisterer(this);
  }
  ++global_arcane_nb_registerer;

  { // Check integrity
    auto* p = global_arcane_first_registerer;
    Integer count = global_arcane_nb_registerer;
    while (p && count > 0) {
      p = p->nextRegisterer();
      --count;
    }
    if (p) {
      std::cerr << "Arcane Fatal Error: Registerer '" << m_name
                << "' conflict in registerer registration\n";
      std::abort();
    }
    else if (count > 0) {
      cout << "Arcane Fatal Error: Registerer '" << m_name
           << "' breaks registerer registration (inconsistent shortcut)\n";
      std::abort();      
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PropertySettingsRegisterer* PropertySettingsRegisterer::
firstRegisterer()
{
  return global_arcane_first_registerer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer PropertySettingsRegisterer::
nbRegisterer()
{
  return global_arcane_nb_registerer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IPropertySettingsInfo> PropertySettingsRegisterer::
createSettingsInfoRef() const
{
  IPropertySettingsInfo* s = nullptr;
  if (m_create_func){
    PropertySettingsBuildInfo sbi;
    s = (m_create_func)(sbi);
  }
  return makeRef(s);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
visitAllRegisteredProperties(IPropertyVisitor* visitor)
{
  auto* rs = PropertySettingsRegisterer::firstRegisterer();
  while (rs){
    Ref<IPropertySettingsInfo> si = rs->createSettingsInfoRef();
    si->applyVisitor(visitor);
    rs = rs->nextRegisterer();
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::properties

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
