// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICaseDocument.h                                             (C) 2000-2025 */
/*                                                                           */
/* Interface of a class managing an XML document of the dataset.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ICASEDOCUMENT_H
#define ARCANE_CORE_ICASEDOCUMENT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/XmlNodeList.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CaseNodeNames;
class XmlNode;
class IXmlDocumentHolder;
class CaseOptionError;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface of a part of a dataset.
 */
class ICaseDocumentFragment
{
 public:

  virtual ~ICaseDocumentFragment() = default;

 public:

  /*!
   * \brief Returns the XML document of the dataset.
   * This pointer remains the property of this class and is destroyed when this
   * instance is destroyed.
   */
  virtual IXmlDocumentHolder* documentHolder() =0;

  //! Returns the document node
  virtual XmlNode documentNode() =0;

  //! Returns the root element.
  virtual XmlNode rootElement() =0;

  //! Language used in the dataset
  virtual String language() const =0;

  //! Category used for default values.
  virtual String defaultCategory() const =0;

  //! Returns the instance containing the names of XML nodes by language.
  virtual CaseNodeNames* caseNodeNames() =0;
  
 public:

  //! Adds an error to the dataset
  virtual void addError(const CaseOptionError& case_error) =0;

  //! Adds a warning to the dataset
  virtual void addWarning(const CaseOptionError& case_error) =0;

  // Indicates if the dataset contains errors.
  virtual bool hasError() const =0;

  // Indicates if the dataset contains warnings.
  virtual bool hasWarnings() const =0;

  //! Writes the errors to the stream o
  virtual void printErrors(std::ostream& o) =0;

  //! Writes the warnings to the stream o
  virtual void printWarnings(std::ostream& o) =0;

  //! Clears the recorded error and warning messages
  virtual void clearErrorsAndWarnings() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface of a class managing an XML document of the dataset.
 * \todo Stop inheriting from ICaseDocumentFragment (use the fragment() method instead)
 */
class ICaseDocument
: public ICaseDocumentFragment
{
 public:

  //! Constructs the instance
  virtual void build() =0;

  //! Clones the document
  virtual ICaseDocument* clone() =0;

 public:

  //! Returns the instance containing the names of XML nodes by language.
  virtual CaseNodeNames* caseNodeNames() =0;

  //! Returns the information element for Arcane
  virtual XmlNode arcaneElement() =0;

  //! Returns the configuration information element
  virtual XmlNode configurationElement() =0;

  //! Returns the element containing the time loop choice
  virtual XmlNode timeloopElement() =0;
  //! Returns the element containing the case title
  virtual XmlNode titleElement() =0;
  //! Returns the element containing the case description
  virtual XmlNode descriptionElement() =0;
  //! Returns the element containing the module descriptions
  virtual XmlNode modulesElement() =0;
  //! Returns the element containing the service descriptions
  virtual XmlNode servicesElement() =0;

  //! Returns the root element of the functions
  virtual XmlNode functionsElement() =0;

  //! Returns the root element of the mesh information
  virtual const XmlNodeList& meshElements() =0;

  //! Element containing the list of meshes (new mechanism) (can be null)
  virtual XmlNode meshesElement() =0;

  //! Name of the case usage class
  virtual String userClass() const =0;
  //! Sets the name of the case usage class
  virtual void setUserClass(const String& value) =0;

  //! Name of the case code
  virtual String codeName() const =0;
  //! Sets the name of the case code
  virtual void setCodeName(const String& value) =0;

  //! Version number of the code corresponding to the case
  virtual String codeVersion() const =0;
  //! Sets the version number of the code
  virtual void setCodeVersion(const String& value) =0;

  //! Name of the document's unit system.
  virtual String codeUnitSystem() const =0;
  //! Sets the name of the document's unit system.
  virtual void setCodeUnitSystem(const String& value) =0;

  //! Sets the category used for default values.
  virtual void setDefaultCategory(const String& v) =0;

  //! Fragment corresponding to this document
  virtual ICaseDocumentFragment* fragment() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
