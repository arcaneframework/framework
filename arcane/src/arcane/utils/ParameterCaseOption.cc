// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParameterCaseOption.cc                                      (C) 2000-2025 */
/*                                                                           */
/* TODO.                                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ParameterCaseOption.h"

#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/CommandLineArguments.h"
#include "arcane/utils/ParameterList.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Ref.h"

#include "arcane/core/IApplication.h"
#include "arcane/core/ICaseDocument.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
const Arcane::String ANY_TAG_STR = "/";
const Arcane::StringView ANY_TAG = ANY_TAG_STR.view();
constexpr Arcane::Integer ANY_INDEX = -1;
constexpr Arcane::Integer GET_INDEX = -2;
} // namespace

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ParameterOptionAddrPart
{
 public:

  ParameterOptionAddrPart()
  : m_tag(ANY_TAG)
  , m_index(ANY_INDEX)
  {}

  explicit ParameterOptionAddrPart(const StringView tag)
  : m_tag(tag)
  , m_index(1)
  {
    ARCANE_ASSERT(tag != ANY_TAG, ("ANY_TAG without ANY_INDEX is forbiden"));
  }

  ParameterOptionAddrPart(const StringView tag, const Integer index)
  : m_tag(tag)
  , m_index(index)
  {
    ARCANE_ASSERT(index == ANY_INDEX || tag != ANY_TAG, ("ANY_TAG without ANY_INDEX is forbiden"));
  }

 public:

  StringView tag() const
  {
    return m_tag;
  }
  Integer index() const
  {
    return m_index;
  }
  void setTag(const StringView tag)
  {
    ARCANE_ASSERT(m_index == ANY_INDEX || tag != ANY_TAG, ("ANY_TAG without ANY_INDEX is forbiden"));
    m_tag = tag;
  }
  void setIndex(const Integer index)
  {
    m_index = index;
  }
  bool isAny() const
  {
    return (m_tag == ANY_TAG && m_index == ANY_INDEX);
  }
  bool operator==(const ParameterOptionAddrPart& other) const
  {
    return (m_tag == other.m_tag || m_tag == ANY_TAG || other.m_tag == ANY_TAG) &&
    (m_index == other.m_index || m_index == ANY_INDEX || other.m_index == ANY_INDEX || m_index == GET_INDEX || other.m_index == GET_INDEX);
  }
  // TODO AH : À supprimer lors du passage en C++20.
  bool operator!=(const ParameterOptionAddrPart& other) const
  {
    return !operator==(other);
  }

 private:

  StringView m_tag;
  Integer m_index;
};

std::ostream& operator<<(std::ostream& o, const ParameterOptionAddrPart& h)
{
  o << (h.tag() == ANY_TAG ? "ANY" : h.tag())
    << "[" << (h.index() == ANY_INDEX ? "ANY" : (h.index() == GET_INDEX ? "GET" : std::to_string(h.index())))
    << "]";
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ParameterOptionAddr
{
 public:

  explicit ParameterOptionAddr(const StringView line)
  {
    Span span_line(line.bytes());
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
        ARCANE_ASSERT(size != 0, ("Invalid option (empty name)"));
        ARCANE_ASSERT(index_begin < span_line.size(), ("Invalid option (']' not found)"));
      }
      else if (span_line[i] == ']') {
        ARCANE_ASSERT(index_begin != i, ("Invalid option ('[]' found without integer)"));
        ARCANE_ASSERT(index_begin != -1, ("Invalid option (']' found without '[')"));

        StringView index_str = line.subView(index_begin, i - index_begin);
        Integer index;
        bool is_bad = builtInGetValue(index, index_str);
        if (is_bad) {
          ARCANE_FATAL("Invalid index");
        }
        m_parts.add(makeRef(new ParameterOptionAddrPart(line.subView(begin, size), index)));
        have_a_no_any = true;
      }

      else if (span_line[i] == '/') {
        ARCANE_ASSERT(i + 1 != span_line.size(), ("Invalid option ('/' found at the end of the param option)"));

        if (index_begin == -1) {
          size = i - begin;
          // Cas ou on a un any_tag any_index ("truc1//truc2").
          if (size == 0) {
            m_parts.add(makeRef(new ParameterOptionAddrPart()));
          }
          else {
            m_parts.add(makeRef(new ParameterOptionAddrPart(line.subView(begin, size))));
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
      ARCANE_ASSERT(size != 0, ("Invalid option (empty name)"));

      m_parts.add(makeRef(new ParameterOptionAddrPart(line.subView(begin, size))));
      have_a_no_any = true;
    }
    if (!have_a_no_any) {
      ARCANE_FATAL("Invalid option");
    }
  }

 public:

  // On ne doit pas bloquer les multiples ParameterOptionAddrPart(ANY) :
  // Construction par iteration : aaaa/bb/ANY/ANY/cc
  void addAddrPart(ParameterOptionAddrPart* part)
  {
    m_parts.add(makeRef(part));
  }

  ParameterOptionAddrPart* addrPart(const Integer index) const
  {
    if (index >= m_parts.size()) {
      if (m_parts[m_parts.size() - 1]->isAny()) {
        return lastAddrPart();
      }
      ARCANE_FATAL("Invalid index");
    }
    return m_parts[index].get();
  }

  ParameterOptionAddrPart* lastAddrPart() const
  {
    return m_parts[m_parts.size() - 1].get();
  }

  Integer nbAddrPart() const
  {
    return m_parts.size();
  }

  bool getIndexes(const ParameterOptionAddr& addr_with_get_index, ArrayView<Integer> indexes) const
  {
    if (!operator==(addr_with_get_index))
      return false;

    ARCANE_ASSERT(indexes.size() == addr_with_get_index.nbIndexToGet(), ("ArrayView too small"));

    Integer index = 0;
    for (Integer i = 0; i < addr_with_get_index.nbAddrPart(); ++i) {
      if (addr_with_get_index.addrPart(i)->index() == GET_INDEX) {
        Integer index_tag = addrPart(i)->index();
        if (index_tag == ANY_INDEX)
          return false;
        indexes[index++] = index_tag;
      }
    }
    return true;
  }

  Integer nbIndexToGet() const
  {
    Integer count = 0;
    for (const auto& elem : m_parts) {
      if (elem->index() == GET_INDEX) {
        count++;
      }
    }
    return count;
  }

 public:

  bool operator==(const ParameterOptionAddr& other) const
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

 private:

  UniqueArray<Ref<ParameterOptionAddrPart>> m_parts;
};

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

class ParameterOptionElement
{
 public:

  ParameterOptionElement(StringView addr, StringView value)
  : m_addr(addr)
  , m_value(value)
  {}

  ParameterOptionAddr addr() const
  {
    return m_addr;
  }

  StringView value() const
  {
    return m_value;
  }

  bool operator==(const ParameterOptionAddr& addr) const
  {
    return m_addr == addr;
  }

 private:

  ParameterOptionAddr m_addr;
  StringView m_value;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ParameterOptionElementsCollection
{
 public:

  void addOption(StringView addr, StringView value)
  {
    m_elements.add({ addr, value });
  }

  ParameterOptionElement element(Integer index)
  {
    return m_elements[index];
  }

  std::optional<StringView> value(const ParameterOptionAddr& addr)
  {
    for (const auto& elem : m_elements) {
      if (elem == addr)
        return elem.value();
    }
    return {};
  }

  bool isExistAddr(const ParameterOptionAddr& addr)
  {
    for (const auto& elem : m_elements) {
      if (elem == addr)
        return true;
    }
    return false;
  }

  Integer countAddr(const ParameterOptionAddr& addr)
  {
    Integer count = 0;
    for (const auto& elem : m_elements) {
      if (elem == addr)
        count++;
    }
    return count;
  }

  void getIndexesOfLine(const ParameterOptionAddr& line, UniqueArray<Integer>& indexes)
  {
    UniqueArray<Integer> new_indexes(line.nbIndexToGet());
    for (const auto& elem : m_elements) {
      if (elem.addr().getIndexes(line, new_indexes)) {
        indexes.addRange(new_indexes);
      }
    }
  }

 private:

  UniqueArray<ParameterOptionElement> m_elements;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParameterCaseOption::
ParameterCaseOption(ICaseMng* case_mng)
: m_case_mng(case_mng)
, m_lines(new ParameterOptionElementsCollection)
{
  m_lang = m_case_mng->caseDocumentFragment()->language();

  m_case_mng->application()->applicationInfo().commandLineArguments().parameters().fillParameters(m_param_names, m_values);

  //Integer d_count = 0;
  for (Integer i = 0; i < m_param_names.count(); ++i) {
    const String& param = m_param_names[i];
    if (param.startsWith("//")) {
      m_lines->addOption(param.view().subView(2), m_values[i].view());
      //m_case_mng->traceMng()->info() << "AddOption : " << m_lines->getOption(d_count).getLine() << " = " << m_lines->getOption(d_count).getValue();
      //d_count++;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParameterCaseOption::
~ParameterCaseOption()
{
  delete m_lines;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// xpath_before_index[index]/xpath_after_index
String ParameterCaseOption::
getParameterOrNull(const String& xpath_before_index, const String& xpath_after_index, Integer index)
{
  // m_case_mng->traceMng()->info() << "getParameterOrNull(SSI)";
  // m_case_mng->traceMng()->info() << "xpath_before_index : " << xpath_before_index
  //                                << " -- xpath_after_index : " << xpath_after_index
  //                                << " -- index : " << index;
  if (index <= 0) {
    ARCANE_FATAL("Index in XML start at 1");
  }

  ParameterOptionAddr addr{ _removeUselessPartInXpath(xpath_before_index.view()) };
  addr.lastAddrPart()->setIndex(index);
  addr.addAddrPart(new ParameterOptionAddrPart(xpath_after_index.view()));

  // m_case_mng->traceMng()->info() << "Line : " << line;

  std::optional<StringView> value = m_lines->value(addr);
  if (value.has_value()) {
    // m_case_mng->traceMng()->info() << "Ret : " << value.value();
    return value.value();
  }
  // m_case_mng->traceMng()->info() << "Ret : {}";
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String ParameterCaseOption::
getParameterOrNull(const String& xpath_before_index, Integer index, bool allow_elems_after_index)
{
  // m_case_mng->traceMng()->info() << "getParameterOrNull(SIB)";
  // m_case_mng->traceMng()->info() << "xpath_before_index : " << xpath_before_index
  //                                << " -- index : " << index
  //                                << " -- allow_elems_after_index : " << allow_elems_after_index;
  if (index <= 0) {
    ARCANE_FATAL("Index in XML start at 1");
  }
  ParameterOptionAddr addr{ _removeUselessPartInXpath(xpath_before_index.view()) };
  addr.lastAddrPart()->setIndex(index);
  if (allow_elems_after_index) {
    addr.addAddrPart(new ParameterOptionAddrPart());
  }

  // m_case_mng->traceMng()->info() << "Line : " << line;
  std::optional<StringView> value = m_lines->value(addr);
  if (value.has_value()) {
    // m_case_mng->traceMng()->info() << "Ret : " << value.value();
    return value.value();
  }
  // m_case_mng->traceMng()->info() << "Ret : {}";
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String ParameterCaseOption::
getParameterOrNull(const String& full_xpath)
{
  // m_case_mng->traceMng()->info() << "getParameterOrNull(S)";
  // m_case_mng->traceMng()->info() << "full_xpath : " << full_xpath;
  const ParameterOptionAddr addr{ _removeUselessPartInXpath(full_xpath.view()) };

  // m_case_mng->traceMng()->info() << "Line : " << line;

  std::optional<StringView> value = m_lines->value(addr);
  if (value.has_value()) {
    // m_case_mng->traceMng()->info() << "Ret : " << value.value();
    return value.value();
  }
  // m_case_mng->traceMng()->info() << "Ret : {}";
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParameterCaseOption::
exist(const String& full_xpath)
{
  // m_case_mng->traceMng()->info() << "exist(S)";
  // m_case_mng->traceMng()->info() << "full_xpath : " << full_xpath;
  const ParameterOptionAddr addr{ _removeUselessPartInXpath(full_xpath.view()) };

  // m_case_mng->traceMng()->info() << "Line : " << line;
  // m_case_mng->traceMng()->info() << "Ret : " << m_lines->exist(line);

  return m_lines->isExistAddr(addr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParameterCaseOption::
existAnyIndex(const String& xpath_before_index, const String& xpath_after_index)
{
  // m_case_mng->traceMng()->info() << "existAnyIndex(SS)";
  // m_case_mng->traceMng()->info() << "xpath_before_index : " << xpath_before_index
  //                                << " -- xpath_after_index : " << xpath_after_index;

  ParameterOptionAddr addr{ _removeUselessPartInXpath(xpath_before_index.view()) };
  addr.lastAddrPart()->setIndex(ANY_INDEX);

  addr.addAddrPart(new ParameterOptionAddrPart(xpath_after_index.view()));

  // m_case_mng->traceMng()->info() << "Line : " << line;
  // m_case_mng->traceMng()->info() << "Ret : " << m_lines->exist(line);

  return m_lines->isExistAddr(addr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParameterCaseOption::
existAnyIndex(const String& full_xpath)
{
  // m_case_mng->traceMng()->info() << "existAnyIndex(S)";
  // m_case_mng->traceMng()->info() << "full_xpath : " << full_xpath;

  ParameterOptionAddr addr{ _removeUselessPartInXpath(full_xpath.view()) };
  addr.lastAddrPart()->setIndex(ANY_INDEX);

  // m_case_mng->traceMng()->info() << "Line : " << line;
  // m_case_mng->traceMng()->info() << "Ret : " << m_lines->exist(line);

  return m_lines->isExistAddr(addr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParameterCaseOption::
indexesInParam(const String& xpath_before_index, const String& xpath_after_index, UniqueArray<Integer>& indexes)
{
  // m_case_mng->traceMng()->info() << "indexesInParam(SSU)";
  // m_case_mng->traceMng()->info() << "xpath_before_index : " << xpath_before_index
  //                                << " -- xpath_after_index : " << xpath_after_index
  //                                << " -- indexes : " << indexes;

  ParameterOptionAddr addr{ _removeUselessPartInXpath(xpath_before_index.view()) };
  addr.lastAddrPart()->setIndex(GET_INDEX);
  addr.addAddrPart(new ParameterOptionAddrPart(xpath_after_index.view()));

  // m_case_mng->traceMng()->info() << "Line : " << line;

  m_lines->getIndexesOfLine(addr, indexes);
  // m_case_mng->traceMng()->info() << "indexes : " << indexes;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParameterCaseOption::
indexesInParam(const String& xpath_before_index, UniqueArray<Integer>& indexes, bool allow_elems_after_index)
{
  // m_case_mng->traceMng()->info() << "indexesInParam(SUB)";
  // m_case_mng->traceMng()->info() << "xpath_before_index : " << xpath_before_index
  //                                << " -- indexes : " << indexes
  //                                << " -- allow_elems_after_index : " << allow_elems_after_index;

  ParameterOptionAddr addr{ _removeUselessPartInXpath(xpath_before_index.view()) };
  addr.lastAddrPart()->setIndex(GET_INDEX);
  if (allow_elems_after_index) {
    addr.addAddrPart(new ParameterOptionAddrPart());
  }

  // m_case_mng->traceMng()->info() << "Line : " << line;
  m_lines->getIndexesOfLine(addr, indexes);
  // m_case_mng->traceMng()->info() << "indexes : " << indexes;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ParameterCaseOption::
count(const String& xpath_before_index, const String& xpath_after_index)
{
  // m_case_mng->traceMng()->info() << "count(SS)";
  // m_case_mng->traceMng()->info() << "xpath_before_index : " << xpath_before_index
  //                                << " -- xpath_after_index : " << xpath_after_index;

  ParameterOptionAddr addr{ _removeUselessPartInXpath(xpath_before_index.view()) };
  addr.lastAddrPart()->setIndex(ANY_INDEX);
  addr.addAddrPart(new ParameterOptionAddrPart(xpath_after_index.view()));
  // m_case_mng->traceMng()->info() << "Line : " << line;
  // m_case_mng->traceMng()->info() << "Ret : " << m_lines->existAndCount(line);

  return m_lines->countAddr(addr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ParameterCaseOption::
count(const String& xpath_before_index)
{
  // m_case_mng->traceMng()->info() << "count(S)";
  // m_case_mng->traceMng()->info() << "xpath_before_index : " << xpath_before_index;

  ParameterOptionAddr addr{ _removeUselessPartInXpath(xpath_before_index.view()) };
  addr.lastAddrPart()->setIndex(ANY_INDEX);
  // m_case_mng->traceMng()->info() << "Line : " << line;
  // m_case_mng->traceMng()->info() << "Ret : " << m_lines->existAndCount(line);

  return m_lines->countAddr(addr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline StringView ParameterCaseOption::
_removeUselessPartInXpath(StringView xpath)
{
  if (m_lang == "fr")
    return xpath.subView(6);
  return xpath.subView(7);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
