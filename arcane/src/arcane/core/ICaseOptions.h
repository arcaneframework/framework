// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICaseOptions.h                                              (C) 2000-2025 */
/*                                                                           */
/* Data set options.                                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ICASEOPTIONS_H
#define ARCANE_CORE_ICASEOPTIONS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/CaseOptionTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface for a list of data set options.
 *
 * This interface is managed by a reference counter and should not be
 * explicitly destroyed.
 */
class ARCANE_CORE_EXPORT ICaseOptions
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 protected:

  virtual ~ICaseOptions() = default;

 public:

  //! True name (untranslated) of the element.
  virtual String rootTagTrueName() const = 0;

  //! Name of the element in the data set language.
  virtual String rootTagName() const = 0;

  //! Name in the language \a lang of the option. Returns \a rootTagTrueName() if no translation exists.
  virtual String translatedName(const String& lang) const = 0;

  ARCCORE_DEPRECATED_2019("Use read(eCaseOptionReadPhase) instead")
  virtual void read(bool is_phase1) = 0;

  /*!
   * \brief Performs the reading of the \a read_phase phase of the options.
   */
  virtual void read(eCaseOptionReadPhase read_phase) = 0;

  virtual void addInvalidChildren(XmlNodeList&) = 0;

  virtual void printChildren(const String& lang, int indent) = 0;

  //! Returns the associated service or `nullptr` if none exists.
  virtual IServiceInfo* caseServiceInfo() const = 0;

  //! Returns the associated module or `nullptr` if none exists.
  virtual IModule* caseModule() const = 0;

  /*!
   * \internal
   * \brief Associates the service \a m with this data set.
   */
  virtual void setCaseServiceInfo(IServiceInfo* m) = 0;

  /*!
   * \internal
   * \brief  Associates the module \a m with this data set.
   */
  virtual void setCaseModule(IModule* m) = 0;

  /*!
   * \internal
   * \brief Adds all child options to the list \a col.
   */
  virtual void deepGetChildren(Array<CaseOptionBase*>& col) = 0;

  virtual ICaseOptionList* configList() = 0;

  virtual const ICaseOptionList* configList() const = 0;

  //! Function indicating the activation status of the option
  virtual ICaseFunction* activateFunction() = 0;

  /*!
   * \brief Indicates whether the option is present in the data set.
   *
   * An option may not appear if it only contains options with a default value.
   */
  virtual bool isPresent() const = 0;

  /*!
   * \internal
   * \brief Adds a translation for the option name.
   *
   * Adds the option name \a name corresponding to the language \a lang.
   * If a translation already exists for this language, it is replaced by
   * this one.
   */
  virtual void addAlternativeNodeName(const String& lang, const String& name) = 0;

  virtual ICaseMng* caseMng() const = 0;
  virtual ITraceMng* traceMng() const = 0;
  /*!
   * \brief Associated sub-domain.
   *
   * \deprecated Do not use this method because eventually an option
   * may exist without a sub-domain.
   */
  ARCCORE_DEPRECATED_2019("Do not use subDomain(). Try to get subDomain from an other way.")
  virtual ISubDomain* subDomain() const = 0;
  ARCCORE_DEPRECATED_2019("Use meshHandle().mesh() instead")
  virtual IMesh* mesh() const = 0;
  virtual MeshHandle meshHandle() const = 0;
  ARCANE_DEPRECATED_REASON("Y2023: use caseMng()->caseDocument() instead.")
  virtual ICaseDocument* caseDocument() const = 0;
  virtual ICaseDocumentFragment* caseDocumentFragment() const = 0;

  /*!
   * \internal
   * Detaches the option from its parent.
   */
  virtual void detach() = 0;

  //! Applies the visitor to this option
  virtual void visit(ICaseDocumentVisitor* visitor) const = 0;

  //! Full name in XPath format corresponding to rootElement()
  virtual String xpathFullName() const = 0;

 public:

  virtual Ref<ICaseOptions> toReference() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface for a list of options that appears multiple times.
 */
class ARCANE_CORE_EXPORT ICaseOptionsMulti
{
 public:

  virtual ~ICaseOptionsMulti() {}

 public:

  virtual void multiAllocate(const XmlNodeList&) = 0;
  virtual ICaseOptions* toCaseOptions() = 0;
  virtual void addChild(ICaseOptionList* v) = 0;
  virtual Integer nbChildren() const = 0;
  virtual ICaseOptionList* child(Integer index) const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT ISubDomain*
_arcaneDeprecatedGetSubDomain(ICaseOptions* opt);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
