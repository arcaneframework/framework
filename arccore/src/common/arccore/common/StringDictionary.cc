// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringDictionary.cc                                         (C) 2000-2025 */
/*                                                                           */
/* Dictionnaire de chaînes de caractères.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/StringDictionary.h"

#include "arccore/base/String.h"
#include "arccore/common/List.h"

#include <map>
#include <exception>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class BadIndexException
: public std::exception
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 \brief Implémentation du dictionnaire de chaîne unicode.
  
  L'implémentation utilise la classe map de la STL.
*/
class StringDictionary::Impl
{
 public:

  typedef std::map<String, String> StringDictType;

 public:

  Impl() {}

 public:

  StringDictType m_dictionary;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringDictionary::
StringDictionary()
: m_p(new Impl())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringDictionary::
StringDictionary(const StringDictionary& rhs)
: m_p(new Impl(*rhs.m_p))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringDictionary::
~StringDictionary()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringDictionary::
add(const String& key, const String& value)
{
  m_p->m_dictionary.insert(std::make_pair(key, value));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String StringDictionary::
remove(const String& key)
{
  auto i = m_p->m_dictionary.find(key);
  String value;
  if (i != m_p->m_dictionary.end()) {
    value = i->second;
    m_p->m_dictionary.erase(i);
  }
  return value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String StringDictionary::
find(const String& key, bool throw_exception) const
{
  auto i = m_p->m_dictionary.find(key);
  String value;
  if (i != m_p->m_dictionary.end())
    value = i->second;
  else if (throw_exception)
    throw BadIndexException();
  return value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StringDictionary::
fill(StringList& keys, StringList& values) const
{
  keys.clear();
  values.clear();
  for (const auto& x : m_p->m_dictionary) {
    keys.add(x.first);
    values.add(x.second);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
