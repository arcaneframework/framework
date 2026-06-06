// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICaseOptionListInternal.h                                   (C) 2000-2023 */
/*                                                                           */
/* Internal part of Arcane's 'ICaseOptionList'.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ICASEOPTIONLISTINTERNAL_H
#define ARCANE_CORE_ICASEOPTIONLISTINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/CaseOptionTypes.h"
#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Internal API of the data set options list interface.
 */
class ARCANE_CORE_EXPORT ICaseOptionListInternal
{
 protected:

  virtual ~ICaseOptionListInternal() = default;

 public:

  virtual void addConfig(CaseOptionBase* o, const XmlNode& parent) = 0;

  //! Positions the root element of the list, with \a parent_element as parent. If already positioned, does nothing
  virtual void setRootElementWithParent(const XmlNode& parent_element) = 0;

  //! Positions the root element of the list. If already positioned, throws an exception
  virtual void setRootElement(const XmlNode& root_element) = 0;

  //! Adds child elements that do not correspond to options in \a nlist
  virtual void addInvalidChildren(XmlNodeList& nlist) = 0;

 public:

  static ICaseOptionList* create(ICaseMng* m, ICaseOptions* ref_opt,
                                 const XmlNode& parent_element);
  static ICaseOptionList* create(ICaseOptionList* parent, ICaseOptions* ref_opt,
                                 const XmlNode& parent_element);
  static ICaseOptionList* create(ICaseOptionList* parent, ICaseOptions* ref_opt,
                                 const XmlNode& parent_element,
                                 bool is_optional, bool is_multi);
  static ICaseOptionList* create(ICaseOptionsMulti* com, ICaseOptions* co,
                                 ICaseMng* m, const XmlNode& element,
                                 Integer min_occurs, Integer max_occurs);
  static ICaseOptionList* create(ICaseOptionsMulti* com, ICaseOptions* co,
                                 ICaseOptionList* parent, const XmlNode& element,
                                 Integer min_occurs, Integer max_occurs);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
