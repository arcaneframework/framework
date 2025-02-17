// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParameterOption.cc                                          (C) 2000-2025 */
/*                                                                           */
/* Classe représentant l'ensemble des paramètres pouvant modifier les        */
/* options du jeu de données.                                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/internal/ParameterOption.h"

#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Ref.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParameterOptionAddrPart::
ParameterOptionAddrPart()
: m_tag(ANY_TAG)
, m_index(ANY_INDEX)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParameterOptionAddrPart::
ParameterOptionAddrPart(const StringView tag)
: m_tag(tag)
, m_index(1)
{
  ARCANE_ASSERT(tag != ANY_TAG, ("ANY_TAG without ANY_INDEX is forbidden"));
  ARCANE_ASSERT(!tag.empty(), ("tag is empty"));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParameterOptionAddrPart::
ParameterOptionAddrPart(const StringView tag, const Integer index)
: m_tag(tag)
, m_index(index)
{
  ARCANE_ASSERT(index == ANY_INDEX || tag != ANY_TAG, ("ANY_TAG without ANY_INDEX is forbidden"));
  ARCANE_ASSERT(!tag.empty(), ("tag is empty"));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringView ParameterOptionAddrPart::
tag() const
{
  return m_tag;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ParameterOptionAddrPart::
index() const
{
  return m_index;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParameterOptionAddrPart::
setTag(const StringView tag)
{
  ARCANE_ASSERT(m_index == ANY_INDEX || tag != ANY_TAG, ("ANY_TAG without ANY_INDEX is forbidden"));
  ARCANE_ASSERT(!tag.empty(), ("tag is empty"));

  m_tag = tag;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParameterOptionAddrPart::
setIndex(const Integer index)
{
  m_index = index;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParameterOptionAddrPart::
isAny() const
{
  return (m_tag == ANY_TAG && m_index == ANY_INDEX);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParameterOptionAddrPart::
operator==(const ParameterOptionAddrPart& other) const
{
  return (m_tag == other.m_tag || m_tag == ANY_TAG || other.m_tag == ANY_TAG) &&
  (m_index == other.m_index || m_index == ANY_INDEX || other.m_index == ANY_INDEX || m_index == GET_INDEX || other.m_index == GET_INDEX);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO AH : À supprimer lors du passage en C++20.
bool ParameterOptionAddrPart::
operator!=(const ParameterOptionAddrPart& other) const
{
  return !operator==(other);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream& operator<<(std::ostream& o, const ParameterOptionAddrPart& h)
{
  o << (h.tag() == ParameterOptionAddrPart::ANY_TAG ? "ANY" : h.tag())
    << "[" << (h.index() == ParameterOptionAddrPart::ANY_INDEX ? "ANY" : (h.index() == ParameterOptionAddrPart::GET_INDEX ? "GET" : std::to_string(h.index())))
    << "]";
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParameterOptionAddr::
ParameterOptionAddr(const StringView addr_str_view)
{
  Span span_line(addr_str_view.bytes());
  Integer begin = 0;
  Integer size = 0;
  Integer index_begin = -1;
  // On interdit les options qui s'appliquent à toutes les caseoptions.
  bool have_a_no_any = false;

  // aaa[0]
  for (Integer i = 0; i < span_line.size(); ++i) {
    if (span_line[i] == '[') {
      index_begin = i + 1;
      size = i - begin;
      if (size == 0) {
        const StringView current = addr_str_view.subView(0, i + 1);
        ARCANE_FATAL("Invalid parameter option (empty tag) -- Current read : {0}", current);
      }
      if (index_begin >= span_line.size()) {
        const StringView current = addr_str_view.subView(0, i + 1);
        ARCANE_FATAL("Invalid parameter option (']' not found) -- Current read : {0}", current);
      }
    }
    else if (span_line[i] == ']') {
      if (index_begin == -1) {
        const StringView current = addr_str_view.subView(0, i + 1);
        ARCANE_FATAL("Invalid parameter option (']' found without '[' before) -- Current read : {0}", current);
      }

      // Motif spécial "[]" (= ANY_INDEX)
      if (index_begin == i) {
        m_parts.add(makeRef(new ParameterOptionAddrPart(addr_str_view.subView(begin, size), ParameterOptionAddrPart::ANY_INDEX)));
        have_a_no_any = true;
      }
      else {
        StringView index_str = addr_str_view.subView(index_begin, i - index_begin);
        Integer index;
        bool is_bad = builtInGetValue(index, index_str);
        if (is_bad) {
          const StringView current = addr_str_view.subView(0, i + 1);
          ARCANE_FATAL("Invalid index in parameter option -- Current read : {0}", current);
        }
        m_parts.add(makeRef(new ParameterOptionAddrPart(addr_str_view.subView(begin, size), index)));
        have_a_no_any = true;
      }
    }

    else if (span_line[i] == '/') {
      if (i + 1 == span_line.size()) {
        const StringView current = addr_str_view.subView(0, i + 1);
        ARCANE_FATAL("Invalid parameter option ('/' found at the end of the param option) -- Current read : {0}", current);
      }

      if (index_begin == -1) {
        size = i - begin;
        // Cas ou on a un any_tag any_index ("truc1//truc2").
        if (size == 0) {
          m_parts.add(makeRef(new ParameterOptionAddrPart()));
        }
        else {
          m_parts.add(makeRef(new ParameterOptionAddrPart(addr_str_view.subView(begin, size))));
          have_a_no_any = true;
        }
      }

      begin = i + 1;
      size = 0;
      index_begin = -1;
    }
  }
  if (index_begin == -1) {
    size = static_cast<Integer>(span_line.size()) - begin;
    if (size == 0) {
      const StringView current = addr_str_view.subView(0, size);
      ARCANE_FATAL("Invalid parameter option (empty tag) -- Current read : {0}", current);
    }

    m_parts.add(makeRef(new ParameterOptionAddrPart(addr_str_view.subView(begin, size))));
    have_a_no_any = true;
  }
  if (!have_a_no_any) {
    ARCANE_FATAL("Invalid option");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// On ne doit pas bloquer les multiples ParameterOptionAddrPart(ANY) :
// Construction par iteration : aaaa/bb/ANY/ANY/cc
void ParameterOptionAddr::
addAddrPart(ParameterOptionAddrPart* part)
{
  m_parts.add(makeRef(part));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParameterOptionAddrPart* ParameterOptionAddr::
addrPart(const Integer index_of_part) const
{
  if (index_of_part >= m_parts.size()) {
    if (m_parts[m_parts.size() - 1]->isAny()) {
      return lastAddrPart();
    }
    ARCANE_FATAL("Invalid index");
  }
  return m_parts[index_of_part].get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParameterOptionAddrPart* ParameterOptionAddr::
lastAddrPart() const
{
  return m_parts[m_parts.size() - 1].get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ParameterOptionAddr::
nbAddrPart() const
{
  return m_parts.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParameterOptionAddr::
getIndexInAddr(const ParameterOptionAddr& addr_with_get_index, ArrayView<Integer> indexes) const
{
  if (!operator==(addr_with_get_index))
    return false;

  ARCANE_ASSERT(indexes.size() == addr_with_get_index.nbIndexToGetInAddr(), ("ArrayView too small"));

  Integer index = 0;
  for (Integer i = 0; i < addr_with_get_index.nbAddrPart(); ++i) {
    if (addr_with_get_index.addrPart(i)->index() == ParameterOptionAddrPart::GET_INDEX) {
      Integer index_tag = addrPart(i)->index();
      if (index_tag == ParameterOptionAddrPart::ANY_INDEX)
        return false;
      indexes[index++] = index_tag;
    }
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ParameterOptionAddr::
nbIndexToGetInAddr() const
{
  Integer count = 0;
  for (const auto& elem : m_parts) {
    if (elem->index() == ParameterOptionAddrPart::GET_INDEX) {
      count++;
    }
  }
  return count;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParameterOptionAddr::
operator==(const ParameterOptionAddr& other) const
{
  Integer nb_iter = 0;
  if (lastAddrPart()->isAny()) {
    nb_iter = nbAddrPart() - 1;
  }
  else if (other.lastAddrPart()->isAny()) {
    nb_iter = other.nbAddrPart() - 1;
  }
  else if (nbAddrPart() != other.nbAddrPart()) {
    return false;
  }
  else {
    nb_iter = nbAddrPart();
  }

  for (Integer i = 0; i < nb_iter; ++i) {
    if (*addrPart(i) != *other.addrPart(i)) {
      return false;
    }
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream& operator<<(std::ostream& o, const ParameterOptionAddr& h)
{
  Integer nb_part = h.nbAddrPart();
  if (nb_part != 0)
    o << *(h.addrPart(0));
  for (Integer i = 1; i < nb_part; ++i) {
    o << "/" << *(h.addrPart(i));
  }
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParameterOptionElement::
ParameterOptionElement(const StringView addr, const StringView value)
: m_addr(addr)
, m_value(value)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParameterOptionAddr ParameterOptionElement::
addr() const
{
  return m_addr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringView ParameterOptionElement::
value() const
{
  return m_value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParameterOptionElement::
operator==(const ParameterOptionAddr& addr) const
{
  return m_addr == addr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParameterOptionElementsCollection::
addParameter(const String& parameter, const String& value)
{
  if (parameter.startsWith("//")) {
    addElement(parameter.view().subView(2), value.view());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParameterOptionElementsCollection::
addElement(StringView addr, StringView value)
{
  m_elements.add({ addr, value });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Un StringView "vide" est éqal à un StringView "nul".
// Comme on travaille avec des String et que la distinction
// vide/nul est importante, on passe par un std::optional.
std::optional<StringView> ParameterOptionElementsCollection::
value(const ParameterOptionAddr& addr)
{
  for (const auto& elem : m_elements) {
    if (elem == addr)
      return elem.value();
  }
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParameterOptionElementsCollection::
isExistAddr(const ParameterOptionAddr& addr)
{
  for (const auto& elem : m_elements) {
    if (elem == addr)
      return true;
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ParameterOptionElementsCollection::
countAddr(const ParameterOptionAddr& addr)
{
  Integer count = 0;
  for (const auto& elem : m_elements) {
    if (elem == addr)
      count++;
  }
  return count;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParameterOptionElementsCollection::
getIndexInAddr(const ParameterOptionAddr& addr_with_get_index, UniqueArray<Integer>& indexes)
{
  UniqueArray<Integer> new_indexes(addr_with_get_index.nbIndexToGetInAddr());
  for (const auto& elem : m_elements) {
    if (elem.addr().getIndexInAddr(addr_with_get_index, new_indexes)) {
      indexes.addRange(new_indexes);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
