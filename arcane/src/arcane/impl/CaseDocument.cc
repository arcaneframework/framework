// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseDocument.cc                                             (C) 2000-2023 */
/*                                                                           */
/* Classe gérant un document XML du jeu de données.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Array.h"

#include "arcane/XmlNode.h"
#include "arcane/IXmlDocumentHolder.h"
#include "arcane/ICaseDocument.h"
#include "arcane/CaseNodeNames.h"
#include "arcane/CaseOptionError.h"
#include "arcane/DomUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CaseDocumentFragment
: public TraceAccessor
, public ICaseDocumentFragment
{
 public:

  CaseDocumentFragment(ITraceMng* tm,IXmlDocumentHolder* document);

  ~CaseDocumentFragment()
  {
    delete m_case_node_names;
  }

 public:

  void init();

 public:

  IXmlDocumentHolder* documentHolder() override { return m_doc_holder.get(); }
  XmlNode documentNode() override { return m_document_node; }
  XmlNode rootElement() override { return m_root_elem; }
  String language() const override { return m_language; }
  String defaultCategory() const override { return m_default_category; }
  CaseNodeNames* caseNodeNames() override { return m_case_node_names; }

 public:

  void addError(const CaseOptionError& case_error) override;
  void addWarning(const CaseOptionError& case_error) override;
  bool hasError() const override;
  bool hasWarnings() const override;
  void printErrors(std::ostream& o) override;
  void printWarnings(std::ostream& o) override;
  void clearErrorsAndWarnings() override;

 public:

  ICaseDocumentFragment* fragment() { return this; }

 public:

  CaseNodeNames* m_case_node_names = nullptr;
  ScopedPtrT<IXmlDocumentHolder> m_doc_holder;
  XmlNode m_document_node;
  XmlNode m_root_elem;
  String m_language;
  String m_default_category;
  UniqueArray<CaseOptionError> m_errors;
  UniqueArray<CaseOptionError> m_warnings;

 private:

  void _assignLanguage(const String& langname);
  void _printErrorsOrWarnings(std::ostream& o,ConstArrayView<CaseOptionError> errors);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe gérant un document XML du jeu de données.
 */
class CaseDocument
: public TraceAccessor
, public ICaseDocument
{
 public:

  CaseDocument(ITraceMng* sm,IXmlDocumentHolder* document);
  ~CaseDocument() override;

  void build() override;
  ICaseDocument* clone() override;

 public:

  //@{
  IXmlDocumentHolder* documentHolder() override { return m_fragment.m_doc_holder.get(); }
  XmlNode documentNode() override { return m_fragment.m_document_node; }
  XmlNode rootElement() override { return m_fragment.m_root_elem; }
  String language() const override { return m_fragment.m_language; }
  String defaultCategory() const override { return m_fragment.m_default_category; }
  CaseNodeNames* caseNodeNames() override { return m_fragment.m_case_node_names; }
  void addError(const CaseOptionError& case_error) override { m_fragment.addError(case_error); }
  void addWarning(const CaseOptionError& case_error) override { m_fragment.addWarning(case_error); }
  bool hasError() const override { return m_fragment.hasError(); }
  bool hasWarnings() const override  { return m_fragment.hasWarnings(); }
  void printErrors(std::ostream& o) override { m_fragment.printErrors(o); }
  void printWarnings(std::ostream& o) override { m_fragment.printWarnings(o); }
  void clearErrorsAndWarnings() override { m_fragment.clearErrorsAndWarnings(); }
  //@}

 public:

  XmlNode arcaneElement() override { return m_arcane_elem; }
  XmlNode configurationElement() override { return m_configuration_elem; }

  XmlNode timeloopElement() override { return m_timeloop_elem; }
  XmlNode titleElement() override { return m_title_elem; }
  XmlNode descriptionElement() override { return m_description_elem; }
  XmlNode modulesElement() override { return m_modules_elem; }
  XmlNode servicesElement() override { return m_services_elem; }

  const XmlNodeList& meshElements() override { return m_mesh_elems; }

  XmlNode meshesElement() override { return m_meshes_elem; }

  XmlNode functionsElement() override { return m_functions_elem; }

  String userClass() const override { return m_user_class; }
  void setUserClass(const String& value) override;

  String codeName() const override { return m_code_name; }
  void setCodeName(const String& value) override;

  String codeVersion() const override { return m_code_version; }
  void setCodeVersion(const String& value) override;

  String codeUnitSystem() const override { return m_code_unit_system; }
  void setCodeUnitSystem(const String& value) override;

  void setDefaultCategory(const String& v) override { m_fragment.m_default_category = v; }

  ICaseDocumentFragment* fragment() override { return m_fragment.fragment(); }

 public:

  // Positionne la langue. Doit être fait avant l'appel à build.
  void setLanguage(const String& language)
  {
    if (!m_fragment.m_language.null())
      ARCANE_FATAL("Language already set");
    m_fragment.m_language = language;
  }

 private:

  CaseDocumentFragment m_fragment;

  XmlNode m_arcane_elem;
  XmlNode m_configuration_elem;
  XmlNode m_timeloop_elem;
  XmlNode m_title_elem;
  XmlNode m_description_elem;
  XmlNode m_modules_elem;
  XmlNode m_services_elem;
  XmlNodeList m_mesh_elems;
  XmlNode m_functions_elem;
  XmlNode m_meshes_elem;

  String m_user_class;
  String m_code_name;
  String m_code_version;
  String m_code_unit_system;

 private:
  
  XmlNode _forceCreateChild(XmlNode& parent,const String& us);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ICaseDocumentFragment*
arcaneCreateCaseDocumentFragment(ITraceMng* tm,IXmlDocumentHolder* document)
{
  auto* doc = new CaseDocumentFragment(tm,document);
  doc->init();
  return doc;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ICaseDocument*
arcaneCreateCaseDocument(ITraceMng* tm,IXmlDocumentHolder* document)
{
  ICaseDocument* doc = new CaseDocument(tm,document);
  doc->build();
  return doc;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ICaseDocument*
arcaneCreateCaseDocument(ITraceMng* tm,const String& lang)
{
  IXmlDocumentHolder* xml_doc = domutils::createXmlDocument();
  CaseDocument* doc = new CaseDocument(tm,xml_doc);
  if (!lang.null())
    doc->setLanguage(lang);
  doc->build();
  return doc;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseDocumentFragment::
CaseDocumentFragment(ITraceMng* tm,IXmlDocumentHolder* document)
: TraceAccessor(tm)
, m_case_node_names(new CaseNodeNames(String()))
, m_doc_holder(document)
, m_document_node(m_doc_holder->documentNode())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseDocument::
CaseDocument(ITraceMng* tm,IXmlDocumentHolder* document)
: TraceAccessor(tm)
, m_fragment(tm,document)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseDocument::
~CaseDocument()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseDocumentFragment::
init()
{
  CaseNodeNames* cnn = caseNodeNames();

  m_root_elem = m_document_node.documentElement();
  if (m_root_elem.null()){
    // Nouveau cas, pour l'instant langue francaise par défaut.
    if (m_language.null())
      m_language = String("fr");
    _assignLanguage(m_language);
    cnn = caseNodeNames();
    m_root_elem = m_document_node.createAndAppendElement(cnn->root,String());
    m_root_elem.setAttrValue(cnn->lang_attribute,m_language);
  }

  m_language = m_root_elem.attrValue(cnn->lang_attribute);

  if (m_language.null()){
    ARCANE_FATAL("Attribute '{0}' not specified in the element <{1}>",
                 cnn->lang_attribute,m_root_elem.name());
  }
  else
    _assignLanguage(m_language);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseDocument::
build()
{
  m_fragment.init();

  // Ces noeuds ont un nom indépendant du langage.
  m_arcane_elem = _forceCreateChild(m_fragment.m_root_elem,"arcane");
  m_configuration_elem = _forceCreateChild(m_arcane_elem,"configuration");

  // Ne pas faire avant 'm_fragment.init()'
  CaseNodeNames* cnn = caseNodeNames();

  // NOTE: Si on ajoute ou change des éléments, il faut mettre
  // à jour la conversion correspondante dans CaseDocumentLangTranslator
  m_timeloop_elem = _forceCreateChild(m_arcane_elem,cnn->timeloop);
  m_title_elem = _forceCreateChild(m_arcane_elem,cnn->title);
  m_description_elem = _forceCreateChild(m_arcane_elem,cnn->description);
  m_modules_elem = _forceCreateChild(m_arcane_elem, cnn->modules);
  m_services_elem = _forceCreateChild(m_arcane_elem, cnn->services);

  XmlNode& root_elem = m_fragment.m_root_elem;

  _forceCreateChild(root_elem,cnn->mesh);
  m_mesh_elems = root_elem.children(cnn->mesh);

  m_functions_elem = _forceCreateChild(root_elem,cnn->functions);
  m_meshes_elem = root_elem.child(cnn->meshes);

  m_user_class = root_elem.attrValue(cnn->user_class);
  m_code_name = root_elem.attrValue(cnn->code_name);
  m_code_version = root_elem.attrValue(cnn->code_version);
  m_code_unit_system = root_elem.attrValue(cnn->code_unit);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ICaseDocument* CaseDocument::
clone()
{
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseDocumentFragment::
_assignLanguage(const String& langname)
{
  delete m_case_node_names;
  m_case_node_names = new CaseNodeNames(langname);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNode CaseDocument::
_forceCreateChild(XmlNode& parent,const String& name)
{
  XmlNode node(parent.child(name));
  if (node.null())
    node = parent.createAndAppendElement(name,String());
  return node;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseDocument::
setUserClass(const String& value)
{
  m_user_class = value;
  m_fragment.m_root_elem.setAttrValue(caseNodeNames()->user_class,value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseDocument::
setCodeName(const String& value)
{
  m_code_name = value;
  m_fragment.m_root_elem.setAttrValue(caseNodeNames()->code_name,value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseDocument::
setCodeVersion(const String& value)
{
  m_code_version = value;
  m_fragment.m_root_elem.setAttrValue(caseNodeNames()->code_version,value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseDocument::
setCodeUnitSystem(const String& value)
{
  m_code_unit_system = value;
  m_fragment.m_root_elem.setAttrValue(caseNodeNames()->code_unit,value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseDocumentFragment::
addError(const CaseOptionError& case_error)
{
  m_errors.add(case_error);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseDocumentFragment::
addWarning(const CaseOptionError& case_error)
{
  m_warnings.add(case_error);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool CaseDocumentFragment::
hasError() const
{
  return m_errors.size()!=0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool CaseDocumentFragment::
hasWarnings() const
{
  return m_warnings.size()!=0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseDocumentFragment::
printErrors(std::ostream& o)
{
  _printErrorsOrWarnings(o,m_errors);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseDocumentFragment::
printWarnings(std::ostream& o)
{
  _printErrorsOrWarnings(o,m_warnings);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseDocumentFragment::
clearErrorsAndWarnings()
{
  m_errors.clear();
  m_warnings.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseDocumentFragment::
_printErrorsOrWarnings(std::ostream& o,ConstArrayView<CaseOptionError> errors)
{
  for( const CaseOptionError& error : errors ){
    if (arcaneIsCheck()){
      o << "TraceFile: " << error.trace().file() << ":" << error.trace().line() << '\n';
      o << "TraceFunc: " << error.trace().name() << '\n';
    }
    o << '<' << error.nodeName() << "> : " << error.message() << '\n';
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
