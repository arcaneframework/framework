// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParameterCaseOption.h                                       (C) 2000-2025 */
/*                                                                           */
/* TODO.                                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_PARAMETERCASEOPTION_H
#define ARCANE_UTILS_PARAMETERCASEOPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/String.h"
#include "arcane/utils/List.h"

#include "arcane/core/ICaseMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ParameterCaseOptionMultiLine;

class ARCANE_UTILS_EXPORT
ParameterCaseOption
{

 public:

  ParameterCaseOption(ICaseMng* case_mng);
  ~ParameterCaseOption();

 public:

  String getParameterOrNull(const String& xpath_before_index, const String& xpath_after_index, Integer index);
  String getParameterOrNull(const String& xpath_before_index, Integer index, bool allow_elems_after_index);
  String getParameterOrNull(const String& full_xpath);

  bool exist(const String& full_xpath);

  bool existAnyIndex(const String& xpath_before_index, const String& xpath_after_index);
  bool existAnyIndex(const String& full_xpath);

  void indexesInParam(const String& xpath_before_index, const String& xpath_after_index, UniqueArray<Integer>& indexes);
  void indexesInParam(const String& xpath_before_index, UniqueArray<Integer>& indexes, bool allow_elems_after_index);

  Integer count(const String& xpath_before_index, const String& xpath_after_index);
  Integer count(const String& xpath_before_index);

 private:

  inline StringView _removeUselessPartInXpath(StringView xpath);

 private:

  StringList m_param_names;
  StringList m_values;
  String m_lang;
  ICaseMng* m_case_mng;
  ParameterCaseOptionMultiLine* m_lines;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
