// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionBase.h                                            (C) 2000-2025 */
/*                                                                           */
/* Base class for a data set option.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CASEOPTIONBASE_H
#define ARCANE_CASEOPTIONBASE_H
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

class CaseOptionBuildInfo;
class ICaseDocument;
class ICaseDocumentFragment;
class ICaseMng;
class ICaseOptionList;
class ISubDomain;
class ICaseFunction;
class CaseOptionBasePrivate;
class ICaseDocumentVisitor;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Base class for a data set option.
 *
 * \ingroup CaseOption
 *
 * Links the option with the name \a m_name to the corresponding DOM node.
 */
class ARCANE_CORE_EXPORT CaseOptionBase
{
 protected:

  CaseOptionBase(const CaseOptionBuildInfo& cob);

 public:

  virtual ~CaseOptionBase();

 public:

  //! Returns the true name (untranslated) of the option.
  String trueName() const;

  //! Returns the option name corresponding to the data set language.
  String name() const;

  //! Name of the option in the language \a lang. Returns \a name() if no translation exists.
  String translatedName(const String& lang) const;

  //! Retrieves the value from the configuration file for the variable
  void search(bool is_phase1);

  //! Prints the option value in the language \a lang, to the stream \a o
  virtual void print(const String& lang, std::ostream& o) const = 0;

  //! Case manager
  ICaseMng* caseMng() const;

  //! Parent OptionList
  ICaseOptionList* parentOptionList() const;

  //! Trace manager
  ITraceMng* traceMng() const;

  //! Sub-domain manager
  ARCCORE_DEPRECATED_2019("Do not use subDomain(). Try to get subDomain from an other way.")
  ISubDomain* subDomain() const;

  //! Returns the document manager
  ARCANE_DEPRECATED_REASON("Y2023: use caseMng()->caseDocument() instead.")
  ICaseDocument* caseDocument() const;

  //! Returns the document associated with this option
  ICaseDocumentFragment* caseDocumentFragment() const;

  //! Positions the root element at \a root_element
  void setRootElement(const XmlNode& root_element);

  //! Returns the root element of the DOM
  XmlNode rootElement() const;

  //! Returns the function linked to this option or `nullptr` if none exists
  virtual ICaseFunction* function() const = 0;

  //! Minimum number of occurrences (for a multiple option)
  Integer minOccurs() const;

  //! Maximum number of occurrences (for a multiple option) (-1 == unbounded)
  Integer maxOccurs() const;

  //! Allows knowing if an option is optional.
  bool isOptional() const;

  /*! \brief Updates the option value from a function.
   *
   * If the option is not linked to a workflow table, it does nothing.
   * Otherwise, it uses \a current_time or \a current_iteration depending on the function parameter type to calculate the new
   * option value. This value will then be normally accessible via
   * the operator() method.
   */
  virtual void updateFromFunction(Real current_time, Integer current_iteration) = 0;

  /*!
    \brief Adds a translation for the option name.
    *
    Adds the option name \a name corresponding to the language \a lang.
    If a translation already exists for this language, it is replaced by
    this one.
  */
  void addAlternativeNodeName(const String& lang, const String& name);

  //! Adds the default value \a value to the category \a category
  void addDefaultValue(const String& category, const String& value);

  //! Applies the visitor to this option
  virtual void visit(ICaseDocumentVisitor* visitor) const = 0;

  //! Throws an exception if the option has not been initialized.
  void checkIsInitialized() const { _checkIsInitialized(); }

 protected:

  //! Returns the default value of the option or 0 if none exists
  String _defaultValue() const;

  void _setDefaultValue(const String& def_value);

 protected:

  virtual void _search(bool is_phase1) = 0;
  void _setIsInitialized();
  bool _isInitialized() const;
  void _checkIsInitialized() const;
  void _checkMinMaxOccurs(Integer nb_occur);
  String _xpathFullName() const;

 private:

  CaseOptionBasePrivate* m_p; //!< Implementation.

 private:

  void _setTranslatedName();
  void _setCategoryDefaultValue();
  /*! \brief Copy constructor.
   *
   * The copy constructor is private because the option should not be
   * copied, notably due to the ICaseFunction which is unique.
   */
  CaseOptionBase(const CaseOptionBase& from) = delete;
  //! Copy assignment operator
  CaseOptionBase& operator=(const CaseOptionBase& from) = delete;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
