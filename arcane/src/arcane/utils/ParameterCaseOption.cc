// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParameterCaseOption.cc                                      (C) 2000-2025 */
/*                                                                           */
/* Classe permettant d'interroger les paramètres pour savoir si des options  */
/* du jeu de données doivent être modifiées par ceux-ci.                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ParameterCaseOption.h"

#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Ref.h"

#include "arcane/utils/internal/ParameterOption.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParameterCaseOption::
ParameterCaseOption(ParameterOptionElementsCollection* parameter_options, const String& lang)
: m_is_fr(lang == "fr")
, m_lines(parameter_options) // On ne récupère pas la propriété.
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String ParameterCaseOption::
getParameterOrNull(const String& xpath_before_index, const String& xpath_after_index, Integer index) const
{
  if (index <= 0) {
    ARCANE_FATAL("Index in XML start at 1");
  }

  ParameterOptionAddr addr{ _removeUselessPartInXpath(xpath_before_index.view()) };
  addr.lastAddrPart()->setIndex(index);
  addr.addAddrPart(new ParameterOptionAddrPart(xpath_after_index.view()));

  std::optional<StringView> value = m_lines->value(addr);
  if (value.has_value())
    return value.value();
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String ParameterCaseOption::
getParameterOrNull(const String& xpath_before_index, Integer index, bool allow_elems_after_index) const
{
  if (index <= 0) {
    ARCANE_FATAL("Index in XML start at 1");
  }
  ParameterOptionAddr addr{ _removeUselessPartInXpath(xpath_before_index.view()) };
  addr.lastAddrPart()->setIndex(index);
  if (allow_elems_after_index) {
    addr.addAddrPart(new ParameterOptionAddrPart());
  }

  std::optional<StringView> value = m_lines->value(addr);
  if (value.has_value())
    return value.value();
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String ParameterCaseOption::
getParameterOrNull(const String& full_xpath) const
{
  const ParameterOptionAddr addr{ _removeUselessPartInXpath(full_xpath.view()) };

  std::optional<StringView> value = m_lines->value(addr);
  if (value.has_value())
    return value.value();
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParameterCaseOption::
exist(const String& full_xpath) const
{
  const ParameterOptionAddr addr{ _removeUselessPartInXpath(full_xpath.view()) };
  return m_lines->isExistAddr(addr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParameterCaseOption::
existAnyIndex(const String& xpath_before_index, const String& xpath_after_index) const
{
  ParameterOptionAddr addr{ _removeUselessPartInXpath(xpath_before_index.view()) };
  addr.lastAddrPart()->setIndex(ParameterOptionAddrPart::ANY_INDEX);

  addr.addAddrPart(new ParameterOptionAddrPart(xpath_after_index.view()));

  return m_lines->isExistAddr(addr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParameterCaseOption::
existAnyIndex(const String& full_xpath) const
{
  const ParameterOptionAddr addr{ _removeUselessPartInXpath(full_xpath.view()) };
  addr.lastAddrPart()->setIndex(ParameterOptionAddrPart::ANY_INDEX);

  return m_lines->isExistAddr(addr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParameterCaseOption::
indexesInParam(const String& xpath_before_index, const String& xpath_after_index, UniqueArray<Integer>& indexes) const
{
  ParameterOptionAddr addr{ _removeUselessPartInXpath(xpath_before_index.view()) };
  addr.lastAddrPart()->setIndex(ParameterOptionAddrPart::GET_INDEX);
  addr.addAddrPart(new ParameterOptionAddrPart(xpath_after_index.view()));

  m_lines->getIndexInAddr(addr, indexes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParameterCaseOption::
indexesInParam(const String& xpath_before_index, UniqueArray<Integer>& indexes, bool allow_elems_after_index) const
{
  ParameterOptionAddr addr{ _removeUselessPartInXpath(xpath_before_index.view()) };
  addr.lastAddrPart()->setIndex(ParameterOptionAddrPart::GET_INDEX);
  if (allow_elems_after_index) {
    addr.addAddrPart(new ParameterOptionAddrPart());
  }

  m_lines->getIndexInAddr(addr, indexes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ParameterCaseOption::
count(const String& xpath_before_index, const String& xpath_after_index) const
{
  ParameterOptionAddr addr{ _removeUselessPartInXpath(xpath_before_index.view()) };
  addr.lastAddrPart()->setIndex(ParameterOptionAddrPart::ANY_INDEX);
  addr.addAddrPart(new ParameterOptionAddrPart(xpath_after_index.view()));

  return m_lines->countAddr(addr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ParameterCaseOption::
count(const String& xpath_before_index) const
{
  const ParameterOptionAddr addr{ _removeUselessPartInXpath(xpath_before_index.view()) };
  addr.lastAddrPart()->setIndex(ParameterOptionAddrPart::ANY_INDEX);

  return m_lines->countAddr(addr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline StringView ParameterCaseOption::
_removeUselessPartInXpath(StringView xpath) const
{
  if (m_is_fr)
    return xpath.subView(6);
  return xpath.subView(7);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
