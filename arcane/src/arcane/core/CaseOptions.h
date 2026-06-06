// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptions.h                                               (C) 2000-2025 */
/*                                                                           */
/* Data set options.                                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CASEOPTIONS_H
#define ARCANE_CASEOPTIONS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounterImpl.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/String.h"
#include "arcane/utils/Ref.h"

#include "arcane/core/XmlNode.h"
#include "arcane/core/ICaseOptions.h"
#include "arcane/core/ICaseOptionList.h"

// These files are not necessary for this '.h' but are included
// because axlstar only includes 'CaseOptions.h'
#include "arcane/core/CaseOptionSimple.h"
#include "arcane/core/CaseOptionEnum.h"
#include "arcane/core/CaseOptionExtended.h"
#include "arcane/core/CaseOptionComplexValue.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
namespace AxlOptionsBuilder
{
class Document;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Base class for a data set options list.
 *
 * Instances of this class must all be allocated by
 * the new() operator and must not be destroyed; the case manager
 * (ICaseMng) handles this.
 */
class ARCANE_CORE_EXPORT CaseOptions
: private ReferenceCounterImpl
, public ICaseOptions
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();

 private:

  struct XmlContent
  {
    String m_xml_content;
    IXmlDocumentHolder* m_document = nullptr;
  };

 public:

  //! Constructs an options set.
  CaseOptions(ICaseMng* cm, const String& name);
  //! Constructs an options set.
  CaseOptions(ICaseOptionList*, const String& name);
  //! Constructs an options set.
  CaseOptions(ICaseMng* cm, const String& name, const XmlNode& parent);
  //! Constructs an options set.
  CaseOptions(ICaseOptionList*, const String& name, const XmlNode& parent, bool is_optional = false, bool is_multi = false);
  //! Constructs an options set.

 protected:

  CaseOptions(ICaseMng*, const String& name, ICaseOptionList* parent);
  //! Constructs an options set.
  CaseOptions(ICaseOptionList*, const String& name, ICaseOptionList* parent);

 private:

  friend class ServiceBuilderWithOptionsBase;

  // Only for ServiceBuilderWithOptionsBase
  static ReferenceCounter<ICaseOptions> createDynamic(ICaseMng* cm, const AxlOptionsBuilder::Document& options_doc);

  //! \internal
  CaseOptions(ICaseMng*, const XmlContent& xm_content);

 public:

  //! Frees resources
  ~CaseOptions() override;

 private:

  CaseOptions(const CaseOptions& rhs) = delete;
  CaseOptions& operator=(const CaseOptions& rhs) = delete;

 public:

  //! Returns the true name (non-translated) of the element.
  String rootTagTrueName() const override;

  //! Returns the name of the element in the data set language.
  String rootTagName() const override;

  //! Name in language \a lang of the option. Returns \a rootTagTrueName() if no translation exists.
  String translatedName(const String& lang) const override;

  //! Returns the true name (non-translated) of the element.
  virtual String trueName() const { return rootTagTrueName(); }

  //! Returns the name of the element in the data set language.
  virtual String name() const { return rootTagName(); }

  void read(bool is_phase1) override
  {
    auto p = (is_phase1) ? eCaseOptionReadPhase::Phase1 : eCaseOptionReadPhase::Phase2;
    read(p);
  }

  void read(eCaseOptionReadPhase phase) override;

  void addInvalidChildren(XmlNodeList&) override;

  void printChildren(const String& lang, int indent) override;

  //! Returns the associated service or 0 if none exists.
  IServiceInfo* caseServiceInfo() const override;

  //! Returns the associated module or 0 if none exists.
  IModule* caseModule() const override;

  //! Associates service \a m with this data set.
  void setCaseServiceInfo(IServiceInfo* m) override;

  //! Associates module \a m with this data set.
  void setCaseModule(IModule* m) override;

  //! Adds all child options to the list \a col.
  void deepGetChildren(Array<CaseOptionBase*>& col) override;

  ICaseOptionList* configList() override;

  const ICaseOptionList* configList() const override;

  //! Function indicating the activation status of the option
  ICaseFunction* activateFunction() override;

  /*!
    \brief True if the option is present in the file,
    false if it is the default value.
  */
  bool isPresent() const override;

  /*!
    \brief Adds a translation for the option name.
    Adds the option name \a name corresponding to language \a lang.
    If a translation already exists for this language, it is replaced by
    this one.
  */
  void addAlternativeNodeName(const String& lang, const String& name) override;

  ICaseMng* caseMng() const override;
  ITraceMng* traceMng() const override;
  ISubDomain* subDomain() const override;
  IMesh* mesh() const override;
  MeshHandle meshHandle() const override;
  ICaseDocument* caseDocument() const override;
  ICaseDocumentFragment* caseDocumentFragment() const override;

  void detach() override;

  void visit(ICaseDocumentVisitor* visitor) const override;

  String xpathFullName() const override;

  Ref<ICaseOptions> toReference() override;

 protected:

  friend class CaseOptionMultiServiceImpl;

  void _setTranslatedName();
  bool _setMeshHandleAndCheckDisabled(const String& mesh_name);

 protected:

  CaseOptionsPrivate* m_p; //!< Implementation

 private:

  void _setMeshHandle(const MeshHandle& handle);
  void _setParent(ICaseOptionList* parent);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
