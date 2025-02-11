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

#include "arcane/core/ICaseMng.h"
#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/String.h"
#include "arcane/utils/List.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_UTILS_EXPORT
ParameterCaseOption
{

 public:

  ParameterCaseOption(ICaseMng* case_mng);
  ~ParameterCaseOption()=default;

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

  StringView _removeUselessPartInXpath(StringView xpath);
  StringView _removeIndexAtEnd(StringView xpath);
  Integer _getIndexAtEnd(StringView xpath);
  Integer _getIndexAtBegin(StringView xpath);


  Integer _getValueIndex(StringView xpath);

  bool _hasOnlyIndex(StringView xpath_part);


 private:

  StringList m_param_names;
  StringList m_values;
  UniqueArray<StringView> m_params_view;
  UniqueArray<StringView> m_values_view;
  String m_lang;
  ICaseMng* m_case_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
