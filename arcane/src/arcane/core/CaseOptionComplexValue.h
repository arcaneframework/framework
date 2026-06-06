// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionComplexValue.h                                    (C) 2000-2023 */
/*                                                                           */
/* Option for a 'complex' data set.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CASEOPTIONCOMPLEXVALUE_H
#define ARCANE_CASEOPTIONCOMPLEXVALUE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounter.h"

#include "arcane/core/ICaseOptionList.h"
#include "arcane/core/XmlNode.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class ICaseOptionsMulti;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Base class for a complex option value.
 *
 * A complex option is composed of multiple instances of this class.
 */
class ARCANE_CORE_EXPORT CaseOptionComplexValue
{
 public:

  CaseOptionComplexValue(ICaseOptionsMulti* opt,ICaseOptionList* clist,const XmlNode& parent_elem);
  virtual ~CaseOptionComplexValue();

 public:

  ARCANE_DEPRECATED_REASON("Y2022: Do not access XML item from option")
  XmlNode element() const { return m_element; }

  ARCANE_DEPRECATED_REASON("Y2022: This method is internal to Arcane. Do not use it")
  ICaseOptionList* configList() const { return m_config_list.get(); }

  //! Full name in the format specified by the XPath standard.
  String xpathFullName() const { return m_element.xpathFullName(); }

 protected:

  // The following two methods are used by the 'axl2cc' generator and
  // must not be modified.
  ICaseOptionList* _configList() { return m_config_list.get(); }
  XmlNode _element() { return m_element; }

 private:

  ReferenceCounter<ICaseOptionList> m_config_list;
  XmlNode m_element;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
