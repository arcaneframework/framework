// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParameterList.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Liste de paramêtres.                                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/ParameterList.h"
#include "arccore/common/StringDictionary.h"
#include "arccore/base/String.h"
#include "arccore/common/Array.h"
#include "arccore/base/FatalErrorException.h"
//#include "arccore/common/Ref.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ParameterList::Impl
{
 public:
  struct NameValuePair
  {
    String name;
    String value;
    friend bool operator==(const NameValuePair& v1,const NameValuePair& v2)
    {
      return (v1.name==v2.name && v1.value==v2.value);
    }
  };
 public:

  Impl()
  {}

 public:

  String getParameter(const String& key)
  {
    String x = m_parameters_dictionary.find(key);
    return x;
  }

  void addParameter(const String& name,const String& value)
  {
    //std::cout << "__ADD_PARAMETER name='" << name << "' v='" << value << "'\n";
    if (name.empty())
      return;

    m_parameters_dictionary.add(name, value);
    m_parameters_list.add({ name, value });
  }
  void setParameter(const String& name,const String& value)
  {
    //std::cout << "__SET_PARAMETER name='" << name << "' v='" << value << "'\n";
    if (name.empty())
      return;

    if (name.startsWith("//")) {
      ARCCORE_FATAL("Set parameter not supported for ParameterOptions.");
    }

    m_parameters_dictionary.add(name,value);
    // Supprime de la liste toutes les occurences ayant
    // pour paramètre \a name
    auto comparer = [=](const NameValuePair& nv){ return nv.name==name; };
    auto new_end = std::remove_if(m_parameters_list.begin(),m_parameters_list.end(),comparer);
    m_parameters_list.resize(new_end-m_parameters_list.begin());
  }
  void removeParameter(const String& name,const String& value)
  {
    //std::cout << "__REMOVE_PARAMETER name='" << name << "' v='" << value << "'\n";
    if (name.empty())
      return;
    if (name.startsWith("//")) {
      ARCCORE_FATAL("Remove parameter not supported for ParameterOptions.");
    }
    // Si le paramètre \a name avec la valeur \a value est trouvé, le supprime.
    // Dans ce cas, il faudra regarder s'il y a toujours
    // dans \a m_parameters_list un paramètre \a name et si c'est le
    // cas c'est la valeur de celui-là qu'on prendra
    String x = m_parameters_dictionary.find(name);
    bool need_fill = false;
    if (x==value){
      m_parameters_dictionary.remove(name);
      need_fill = true;
    }
    // Supprime de la liste toutes les occurences 
    // du paramètre avec la valeur souhaitée
    NameValuePair ref_value{name,value};
    auto new_end = std::remove(m_parameters_list.begin(),m_parameters_list.end(),ref_value);
    m_parameters_list.resize(new_end-m_parameters_list.begin());
    if (need_fill)
      _fillDictionaryWithValueInList(name);
  }
  void fillParameters(StringList& param_names,StringList& values) const
  {
    m_parameters_dictionary.fill(param_names, values);
  }

 private:
  void _fillDictionaryWithValueInList(const String& name)
  {
    for( auto& nv : m_parameters_list )
      if (nv.name==name)
        m_parameters_dictionary.add(nv.name,nv.value);
  }
 private:
  StringDictionary m_parameters_dictionary;
  UniqueArray<NameValuePair> m_parameters_list;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParameterList::
ParameterList()
: m_p(new Impl())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParameterList::
ParameterList(const ParameterList& rhs)
: m_p(new Impl(*rhs.m_p))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParameterList::
~ParameterList()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String ParameterList::
getParameterOrNull(const String& param_name) const
{
  return m_p->getParameter(param_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParameterList::
addParameterLine(const String& line)
{
  Span<const Byte> bytes = line.bytes();
  Int64 len = bytes.length();
  for( Int64 i=0; i<len; ++i ){
    Byte c = bytes[i];
    Byte cnext = ((i+1)<len) ? bytes[i+1] : '\0';
    if (c=='='){
      m_p->addParameter(line.substring(0,i),line.substring(i+1));
      return false;
    }
    if (c=='+' && cnext=='='){
      m_p->addParameter(line.substring(0,i),line.substring(i+2));
      return false;
    }
    if (c==':' && cnext=='='){
      m_p->setParameter(line.substring(0,i),line.substring(i+2));
      return false;
    }
    if (c=='-' && cnext=='='){
      m_p->removeParameter(line.substring(0,i),line.substring(i+2));
      return false;
    }
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParameterList::
fillParameters(StringList& param_names,StringList& values) const
{
  m_p->fillParameters(param_names,values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
