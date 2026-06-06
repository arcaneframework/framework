// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICaseOptionList.h                                           (C) 2000-2025 */
/*                                                                           */
/* Data set options.                                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ICASEOPTIONLIST_H
#define ARCANE_CORE_ICASEOPTIONLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/CaseOptionTypes.h"
#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class XmlNode;
class ICaseMng;
class XmlNodeList;
class ICaseOptionListInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface for a data set options list.
 */
class ARCANE_CORE_EXPORT ICaseOptionList
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 protected:

  virtual ~ICaseOptionList() = default;

 public:

  /*!
   * \brief Returns the element associated with this options list.
   *
   * If there are none, returns a null XmlNode. If there are multiple, returns the
   * first one.
   */
  virtual XmlNode rootElement() const = 0;
  //! Returns the parent element.
  virtual XmlNode parentElement() const = 0;
  //! Adds the list \a co to the list of children.
  virtual void addChild(ICaseOptions* co) = 0;
  //! Removes \a co from the list of children.
  virtual void removeChild(ICaseOptions* co) = 0;
  //! Returns the case manager
  virtual ICaseMng* caseMng() const = 0;

  //! Reads the option values from the DOM elements.
  virtual void readChildren(bool is_phase1) = 0;
  //! Displays the list of child options in language \a lang and their value
  virtual void printChildren(const String& lang, int indent) = 0;
  //! Returns the name of the element of this list
  virtual String rootTagName() const = 0;
  //! Adds all child options to the list \a col.
  virtual void deepGetChildren(Array<CaseOptionBase*>& col) = 0;
  //! Indicates if the option is present in the data set.
  virtual bool isPresent() const = 0;
  //! Indicates if the option is optional
  virtual bool isOptional() const = 0;
  //! Minimum number of occurrences
  virtual Integer minOccurs() const = 0;
  //! Maximum number of occurrences
  virtual Integer maxOccurs() const = 0;
  //! Applies the visitor \a visitor
  virtual void visit(ICaseDocumentVisitor* visitor) = 0;
  //! Full name in XPath format corresponding to rootElement()
  virtual String xpathFullName() const = 0;
  //! Handle of the associated mesh
  virtual MeshHandle meshHandle() const = 0;

  //! Associated document.
  virtual ICaseDocumentFragment* caseDocumentFragment() const = 0;

  /*!
   * \brief Disables the option as if it were absent.
   *
   * This is used, for example, if the option is associated with a mesh
   * that is not defined.
   */
  virtual void disable() = 0;

 public:

  //! Adds option \a o with parent \a parent
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual void addConfig(CaseOptionBase* o, XmlNode parent) = 0;

  //! Positions the root element of the list, with \a parent_element as parent. If already positioned, does nothing
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual void setRootElementWithParent(XmlNode parent_element) = 0;

  //! Positions the root element of the list. If already positioned, throws an exception
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual void setRootElement(XmlNode root_element) = 0;

  //! Adds child elements that do not correspond to options in \a nlist
  ARCANE_DEPRECATED_REASON("Y2023: This method is internal to Arcane")
  virtual void addInvalidChildren(XmlNodeList& nlist) = 0;

 public:

  //! Internal Arcane API
  virtual ICaseOptionListInternal* _internalApi() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
