// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionBuildInfo.h                                       (C) 2000-2025 */
/*                                                                           */
/* Information for building a dataset option                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_CASEOPTIONBUILDINFO_H
#define ARCANE_CORE_CASEOPTIONBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/core/XmlNode.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ICaseMng;
class ICaseOptionList;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \ingroup CaseOption
 * \brief Information for building a dataset option.
 */
class ARCANE_CORE_EXPORT CaseOptionBuildInfo
{
 public:

  CaseOptionBuildInfo(ICaseOptionList* icl, const String& s,
                      const XmlNode& element, const String& def_val,
                      Integer min_occurs, Integer max_occurs);
  CaseOptionBuildInfo(ICaseOptionList* icl, const String& s,
                      const XmlNode& element, const String& def_val,
                      Integer min_occurs, Integer max_occurs, bool is_optional);

 public:

  ICaseMng* caseMng() const { return m_case_mng; }
  ICaseOptionList* caseOptionList() const { return m_case_option_list; }
  String name() const { return m_name; }
  String defaultValue() const { return m_default_value; }
  XmlNode element() const { return m_element; }
  Integer minOccurs() const { return m_min_occurs; }
  Integer maxOccurs() const { return m_max_occurs; }
  bool isOptional() const { return m_is_optional; }

 private:

  ICaseMng* m_case_mng;
  ICaseOptionList* m_case_option_list;
  String m_name; //!< Name of the option
  String m_default_value; //!< Default value (null if none)
  XmlNode m_element; //!< Element of the option
  Integer m_min_occurs; //!< Minimum number of occurrences
  Integer m_max_occurs; //!< Maximum number of occurrences (-1 == unbounded)
  bool m_is_optional;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
