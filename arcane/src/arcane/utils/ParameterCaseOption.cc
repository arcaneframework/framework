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
#include "arcane/core/IApplication.h"
#include "arcane/core/ICaseDocument.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/String.h"
#include "arcane/utils/StringDictionary.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/FatalErrorException.h"
#include "arccore/trace/ITraceMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

struct ParameterOptionPart
{
  StringView m_part;
  Integer m_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ParameterOptionLine
{
 public:

  explicit ParameterOptionLine(const StringView line)
  {
    Span span_line(line.bytes());
    Integer begin = 0;
    Integer size = 0;
    Integer index_begin = -1;

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
        m_parts.add({ line.subView(begin, size), index });
      }
      else if (span_line[i] == '/') {
        ARCANE_ASSERT(i + 1 != span_line.size(), ("Invalid option ('/' found at the end of the param option)"));

        if (index_begin == -1) {
          size = i - begin;
          ARCANE_ASSERT(size != 0, ("Invalid option (empty name)"));

          m_parts.add({ line.subView(begin, size), 1 });
        }

        begin = i + 1;
        size = 0;
        index_begin = -1;
      }
    }
    if (index_begin == -1) {
      size = span_line.size() - begin;
      ARCANE_ASSERT(size != 0, ("Invalid option (empty name)"));

      m_parts.add({ line.subView(begin, size), 1 });
    }
    m_parts.shrink();
  }

 public:

  ParameterOptionPart getPart(Integer index)
  {
    return m_parts[index];
  }

  Integer nbPart()
  {
    return m_parts.size();
  }

 private:

  UniqueArray<ParameterOptionPart> m_parts;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ParameterCaseOptionLine
{
 public:

  ParameterCaseOptionLine(StringView line, StringView value)
  : m_line(line)
  , m_value(value)
  {}

  ParameterOptionLine getLine()
  {
    return m_line;
  }

  StringView getValue() const
  {
    return m_value;
  }

 private:

  ParameterOptionLine m_line;
  StringView m_value;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ParameterCaseOptionMultiLine
{
 public:

  void addOption(StringView line, StringView value)
  {
    m_lines.add({ line, value });
  }

  ParameterCaseOptionLine getOption(Integer index)
  {
    return m_lines[index];
  }

 private:

  UniqueArray<ParameterCaseOptionLine> m_lines;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParameterCaseOption::
ParameterCaseOption(ICaseMng* case_mng)
: m_case_mng(case_mng)
, m_lines(new ParameterCaseOptionMultiLine)
{
  m_lang = m_case_mng->caseDocumentFragment()->language();

  m_case_mng->application()->applicationInfo().commandLineArguments().parameters().fillParameters(m_param_names, m_values);

  Integer size = m_param_names.count();

  m_params_view.reserve(size);
  m_values_view.reserve(size);

  Integer true_size = 0;
  for (Integer i = 0; i < m_param_names.count(); ++i) {
    const String& param = m_param_names[i];
    if (param.startsWith("//")) {
      m_params_view.add(param.view().subView(2));
      m_values_view.add(m_values[i].view());

      m_lines->addOption(param.view().subView(2), m_values[i].view());

      true_size++;
    }
  }
  m_params_view.resize(true_size);
  m_values_view.resize(true_size);

  ITraceMng* tm = case_mng->traceMng();
  tm->info() << "Try with : " << m_params_view[2];
  ParameterCaseOptionLine pa(m_lines->getOption(2));
  ParameterOptionLine line = pa.getLine();
  for (Integer i = 0; i < line.nbPart(); ++i) {
    tm->info() << "i : " << i << " -- elem : " << line.getPart(i).m_part << " -- index : " << line.getPart(i).m_index;
  }
  tm->info() << "Value : " << pa.getValue();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParameterCaseOption::~ParameterCaseOption()
{
  delete m_lines;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// xpath_before_index[index]/xpath_after_index
String ParameterCaseOption::
getParameterOrNull(const String& xpath_before_index, const String& xpath_after_index, Integer index)
{
  if (index <= 0) {
    ARCANE_FATAL("Index in XML start at 1");
  }

  StringView xpath_view = _removeIndexAtEnd(_removeUselessPartInXpath(xpath_before_index));

  String xpath_with_index_attr = String::format("{0}[{1}]/{2}", xpath_view, index, xpath_after_index);

  if (index > 1) {
    Integer index_value = _getValueIndex(xpath_with_index_attr);
    if (index_value == -1)
      return {};
    return m_values_view[index_value];
  }

  String xpath_with_attr = String::format("{0}/{1}", xpath_view, xpath_after_index);
  for (Integer i = 0; i < m_params_view.size(); ++i) {
    if (m_params_view[i] == xpath_with_index_attr.view() || m_params_view[i] == xpath_with_attr.view()) {
      return m_values_view[i];
    }
  }
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String ParameterCaseOption::
getParameterOrNull(const String& xpath_before_index, Integer index, bool allow_elems_after_index)
{
  if (index <= 0) {
    ARCANE_FATAL("Index in XML start at 1");
  }

  StringView xpath_view = _removeIndexAtEnd(_removeUselessPartInXpath(xpath_before_index));

  String xpath_with_index_attr = String::format("{0}[{1}]", xpath_view, index);


  if (allow_elems_after_index) {
    StringView before_view_with_attr = xpath_with_index_attr.view();
    if (index > 1) {
      for (Integer i = 0; i < m_params_view.size(); ++i) {
        StringView param = m_params_view[i];
        if (param.size()-before_view_with_attr.size() >= 0) {
          StringView begin = param.subView(0, before_view_with_attr.size());
          if (begin == before_view_with_attr) {
            return m_values_view[i];
          }
        }
      }
      return {};
    }

    for (Integer i = 0; i < m_params_view.size(); ++i) {
      StringView param = m_params_view[i];
      if (param.size()-xpath_view.size() >= 0) {
        StringView begin = param.subView(0, xpath_view.size());
        if (begin == xpath_view) {
          return m_values_view[i];
        }
      }
      if (param.size()-before_view_with_attr.size() >= 0) {
        StringView begin = param.subView(0, before_view_with_attr.size());
        if (begin == before_view_with_attr) {
          return m_values_view[i];
        }
      }
    }
    return {};

  }
  else {
    if (index > 1) {
      for (Integer i = 0; i < m_params_view.size(); ++i) {
        if (m_params_view[i] == xpath_with_index_attr.view()) {
          return m_values_view[i];
        }
      }
      return {};
    }

    for (Integer i = 0; i < m_params_view.size(); ++i) {
      if (m_params_view[i] == xpath_view || m_params_view[i] == xpath_with_index_attr.view()) {
        return m_values_view[i];
      }
    }
  }
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String ParameterCaseOption::
getParameterOrNull(const String& full_xpath)
{
  Integer index_value = _getValueIndex(_removeUselessPartInXpath(full_xpath));
  if (index_value == -1) return {};
  return m_values_view[index_value];
}



/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParameterCaseOption::
exist(const String& full_xpath)
{
  StringView xpath = _removeUselessPartInXpath(full_xpath);

  for (auto param : m_params_view) {
    if (param == xpath) {
      return true;
    }
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParameterCaseOption::
existAnyIndex(const String& xpath_before_index, const String& xpath_after_index)
{
  StringView before_view = _removeIndexAtEnd(_removeUselessPartInXpath(xpath_before_index));
  StringView after_view = xpath_after_index.view();

  for (auto param : m_params_view) {
    // > 0 car il doit y avoir au moins un "/" entre les deux.
    if (param.size()-after_view.size()-before_view.size() > 0) {
      StringView begin = param.subView(0, before_view.size());
      if (begin == before_view) {
        StringView end = param.subView(param.size()-after_view.size(), after_view.size());
        if (end == after_view) {
          // Le "-1" : On retire le "/" à la fin du between.
          StringView between = param.subView(before_view.size(), param.size()-after_view.size()-before_view.size()-1);
          if (_hasOnlyIndex(between)){
            return true;
          }
        }
      }
    }
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParameterCaseOption::
existAnyIndex(const String& full_xpath)
{
  StringView xpath_view = _removeIndexAtEnd(_removeUselessPartInXpath(full_xpath));

  for (auto param : m_params_view) {
    if (_removeIndexAtEnd(param) == xpath_view) {
      return true;
    }
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParameterCaseOption::
indexesInParam(const String& xpath_before_index, const String& xpath_after_index, UniqueArray<Integer>& indexes)
{
  StringView before_view = _removeIndexAtEnd(_removeUselessPartInXpath(xpath_before_index));
  StringView after_view = xpath_after_index.view();

  // m_case_mng->traceMng()->info() << "full_xpath : " << xpath_view
  //                                << " -- name_view : " << name_view
  //                                << " -- m_params_view.size() : " << m_params_view.size()
  // ;

  for (auto param : m_params_view) {
    // > 0 car il doit y avoir au moins un "/" entre les deux.
    if (param.size()-after_view.size()-before_view.size() > 0) {
      StringView begin = param.subView(0, before_view.size());
      if (begin == before_view) {
        StringView end = param.subView(param.size()-after_view.size(), after_view.size());
        if (end == after_view) {
          // Le "-1" : On retire le "/" à la fin du between.
          StringView between = param.subView(before_view.size(), param.size()-after_view.size()-before_view.size()-1);
          if (_hasOnlyIndex(between)){
            Integer index = _getIndexAtBegin(between);
            if (indexes.contains(index)) {
              // TODO Warning : Doublon
            }
            else {
              indexes.add(index);
            }
          }
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParameterCaseOption::
indexesInParam(const String& xpath_before_index, UniqueArray<Integer>& indexes, bool allow_elems_after_index)
{
  StringView before_view = _removeIndexAtEnd(_removeUselessPartInXpath(xpath_before_index));

  for (auto param : m_params_view) {
    // > 0 car il doit y avoir au moins un "/" entre les deux.
    if (param.size()-before_view.size() > 0) {
      StringView begin = param.subView(0, before_view.size());
      if (begin == before_view) {
        StringView end = param.subView(before_view.size());
        if (allow_elems_after_index || _hasOnlyIndex(end)){
          Integer index = _getIndexAtBegin(end);
          if (indexes.contains(index)) {
            // TODO Warning : Doublon
          }
          else {
            indexes.add(index);
          }
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ParameterCaseOption::
count(const String& xpath_before_index, const String& xpath_after_index)
{
  UniqueArray<Integer> indexes;
  indexesInParam(xpath_before_index, xpath_after_index, indexes);
  return indexes.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ParameterCaseOption::
count(const String& xpath_before_index)
{
  UniqueArray<Integer> indexes;
  indexesInParam(xpath_before_index, indexes, false);
  return indexes.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringView ParameterCaseOption::
_removeUselessPartInXpath(StringView xpath)
{
  if (m_lang == "fr")
    return xpath.subView(6);
  return xpath.subView(7);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringView ParameterCaseOption::
_removeIndexAtEnd(StringView xpath)
{
  Span<const Byte> bytes_span = xpath.bytes();
  if (bytes_span[bytes_span.size()-1] == ']') {
    // bytes_span.size()-3 car on ne peut pas avoir de crochets vides : "[]".
    // i >= 2 car il y a forcément quelque chose avant les crochets.
    for (Integer i = bytes_span.size()-3; i >= 2; --i) {
      if (bytes_span[i] == '[') {
        return StringView{bytes_span.subSpan(0, i)};
      }
    }
    ARCANE_FATAL("Bad xpath");
  }
  return xpath;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ParameterCaseOption::
_getIndexAtEnd(StringView xpath)
{
  if (xpath.size() == 0) return 1;
  Span<const Byte> bytes_span = xpath.bytes();
  if (bytes_span[bytes_span.size() - 1] == ']') {
    // bytes_span.size()-3 car on ne peut pas avoir de crochets vides : "[]".
    // i >= 2 car il y a forcément quelque chose avant les crochets.
    for (Integer i = bytes_span.size() - 3; i >= 2; --i) {
      if (bytes_span[i] == '[') {
        StringView index_str = bytes_span.subSpan(i + 1, bytes_span.size() - 1 - i - 1);
        Integer index = 0;
        bool is_bad = builtInGetValue(index, index_str);
        if (is_bad) {
          ARCANE_FATAL("Invalid index");
        }
        if (index < 1) {
          ARCANE_FATAL("Index in XML start at 1");
        }
        return index;
      }
    }
    ARCANE_FATAL("Bad xpath");
  }
  return 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ParameterCaseOption::
_getIndexAtBegin(StringView xpath)
{
  if (xpath.size() == 0) return 1;
  Span<const Byte> bytes_span = xpath.bytes();
  if (bytes_span[0] == '[') {
    for (Integer i = 2; i < bytes_span.size(); ++i) {
      if (bytes_span[i] == ']') {
        StringView index_str = bytes_span.subSpan(1, i-1);
        Integer index = 0;
        bool is_bad = builtInGetValue(index, index_str);
        if (is_bad) {
          ARCANE_FATAL("Invalid index");
        }
        if (index < 1) {
          ARCANE_FATAL("Index in XML start at 1");
        }
        return index;
      }
    }
    ARCANE_FATAL("Bad xpath");
  }
  return 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ParameterCaseOption::
_getValueIndex(StringView xpath)
{
  for (Integer i = 0; i < m_params_view.size(); ++i) {
    if (m_params_view[i] == xpath) {
      return i;
    }
  }
  return -1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParameterCaseOption::
_hasOnlyIndex(StringView xpath_part)
{
  // xpath_before_index=0  (xpath_part="")
  if (xpath_part.size() == 0) return true;
  Span<const Byte> bytes_span = xpath_part.bytes();

  // xpath_before_index[0]=0  (xpath_part="[0]")
  if (bytes_span.size() < 3) return false;
  // xpath_before_indextruc[0]=0  (xpath_part="truc[0]")
  if (bytes_span[0] != '[') return false;

  // xpath_before_index[0]/xpath_after_index=0  (xpath_part="[0]")
  for (Integer i = 2; i < bytes_span.size(); ++i) {
    if (bytes_span[i] == ']') {
      return (i == bytes_span.size()-1);
    }
  }

  // xpath_before_index[0]/truc/xpath_after_index=0  (xpath_part="[0]/truc/")
  // xpath_before_index[0]/truc[0]/xpath_after_index=0  (xpath_part="[0]/truc[0]/")
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
