// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParameterListWithCaseOption.cc                              (C) 2000-2025 */
/*                                                                           */
/* Parameter list with support for dataset options.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ParameterList.h"
#include "arcane/utils/StringDictionary.h"
#include "arcane/utils/String.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Ref.h"

#include "arcane/utils/internal/ParameterOption.h"
#include "arcane/utils/internal/ParameterListWithCaseOption.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * This class manages command-line parameters that allow overriding
 * dataset options.
 *
 * It is a copy of the ParameterList class.
 *
 * TODO: This class should only retain options that start with '//' and are
 * related to the dataset.
 */
namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ParameterListWithCaseOption::Impl
{
 public:

  struct NameValuePair
  {
    String name;
    String value;
    friend bool operator==(const NameValuePair& v1, const NameValuePair& v2)
    {
      return (v1.name == v2.name && v1.value == v2.value);
    }
  };

 public:

  Impl()
  : m_parameter_option(makeRef(new ParameterOptionElementsCollection()))
  {}

 public:

  String getParameter(const String& key)
  {
    if (key.startsWith("//")) {
      if (const auto value = m_parameter_option->value(ParameterOptionAddr(key.view().subView(2))))
        return value.value();
      return {};
    }
    String x = m_parameters_dictionary.find(key);
    return x;
  }

  void addParameter(const String& name, const String& value)
  {
    //std::cout << "__ADD_PARAMETER name='" << name << "' v='" << value << "'\n";
    if (name.empty())
      return;

    if (name.startsWith("//")) {
      m_parameters_option_list.add({ name, value });
      m_parameter_option->addParameter(m_parameters_option_list[m_parameters_option_list.size() - 1].name, m_parameters_option_list[m_parameters_option_list.size() - 1].value);
      return;
    }

    m_parameters_dictionary.add(name, value);
    m_parameters_list.add({ name, value });
    m_parameter_option->addParameter(m_parameters_list[m_parameters_list.size() - 1].name, m_parameters_list[m_parameters_list.size() - 1].value);
  }

  void setParameter(const String& name, const String& value)
  {
    //std::cout << "__SET_PARAMETER name='" << name << "' v='" << value << "'\n";
    if (name.empty())
      return;

    if (name.startsWith("//")) {
      ARCANE_FATAL("Set parameter not supported for ParameterOptions.");
    }

    m_parameters_dictionary.add(name, value);
    // Remove all occurrences from the list having \a name as the parameter
    auto comparer = [=](const NameValuePair& nv) { return nv.name == name; };
    auto new_end = std::remove_if(m_parameters_list.begin(), m_parameters_list.end(), comparer);
    m_parameters_list.resize(new_end - m_parameters_list.begin());
  }

  void removeParameter(const String& name, const String& value)
  {
    //std::cout << "__REMOVE_PARAMETER name='" << name << "' v='" << value << "'\n";
    if (name.empty())
      return;
    if (name.startsWith("//")) {
      ARCANE_FATAL("Remove parameter not supported for ParameterOptions.");
    }
    // If the parameter \a name with the value \a value is found, it is removed.
    // In this case, we must check if there is still a parameter \a name in
    // \a m_parameters_list, and if so, we will take its value.
    String x = m_parameters_dictionary.find(name);
    bool need_fill = false;
    if (x == value) {
      m_parameters_dictionary.remove(name);
      need_fill = true;
    }
    // Remove all occurrences
    // of the parameter with the desired value
    NameValuePair ref_value{ name, value };
    auto new_end = std::remove(m_parameters_list.begin(), m_parameters_list.end(), ref_value);
    m_parameters_list.resize(new_end - m_parameters_list.begin());
    if (need_fill)
      _fillDictionaryWithValueInList(name);
  }
  void fillParameters(StringList& param_names, StringList& values) const
  {
    m_parameters_dictionary.fill(param_names, values);
    for (const auto& [name, value] : m_parameters_option_list) {
      param_names.add(name);
      values.add(value);
      std::cout << "FILL name='" << name << "' value='" << value << "'\n";
    }
  }

  ParameterOptionElementsCollection* getParameterOption() const
  {
    return m_parameter_option.get();
  }

 private:

  void _fillDictionaryWithValueInList(const String& name)
  {
    for (auto& nv : m_parameters_list)
      if (nv.name == name)
        m_parameters_dictionary.add(nv.name, nv.value);
  }

 private:

  StringDictionary m_parameters_dictionary;
  UniqueArray<NameValuePair> m_parameters_list;
  UniqueArray<NameValuePair> m_parameters_option_list;
  Ref<ParameterOptionElementsCollection> m_parameter_option;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParameterListWithCaseOption::
ParameterListWithCaseOption()
: m_p(new Impl())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParameterListWithCaseOption::
ParameterListWithCaseOption(const ParameterListWithCaseOption& rhs)
: m_p(new Impl(*rhs.m_p))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParameterListWithCaseOption::
~ParameterListWithCaseOption()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String ParameterListWithCaseOption::
getParameterOrNull(const String& param_name) const
{
  return m_p->getParameter(param_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParameterListWithCaseOption::
addParameterLine(const String& line)
{
  Span<const Byte> bytes = line.bytes();
  Int64 len = bytes.length();
  for (Int64 i = 0; i < len; ++i) {
    Byte c = bytes[i];
    Byte cnext = ((i + 1) < len) ? bytes[i + 1] : '\0';
    if (c == '=') {
      m_p->addParameter(line.substring(0, i), line.substring(i + 1));
      return false;
    }
    if (c == '+' && cnext == '=') {
      m_p->addParameter(line.substring(0, i), line.substring(i + 2));
      return false;
    }
    if (c == ':' && cnext == '=') {
      m_p->setParameter(line.substring(0, i), line.substring(i + 2));
      return false;
    }
    if (c == '-' && cnext == '=') {
      m_p->removeParameter(line.substring(0, i), line.substring(i + 2));
      return false;
    }
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParameterCaseOption ParameterListWithCaseOption::
getParameterCaseOption(const String& language) const
{
  return { m_p->getParameterOption(), language };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParameterListWithCaseOption::
addParameters(const ParameterList& parameters)
{
  // TODO: Only consider options that start with '//'
  StringList names;
  StringList values;
  parameters.fillParameters(names, values);
  Int32 size = names.count();
  for (Int32 i = 0; i < size; ++i)
    m_p->addParameter(names[i], values[i]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
