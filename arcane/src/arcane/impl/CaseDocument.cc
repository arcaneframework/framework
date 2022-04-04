// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseDocument.cc                                             (C) 2000-2018 */
/*                                                                           */
/* Classe gérant un document XML du jeu de données.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Array.h"

#include "arcane/XmlNode.h"
#include "arcane/IApplication.h"
#include "arcane/IXmlDocumentHolder.h"
#include "arcane/IRessourceMng.h"
#include "arcane/ICaseDocument.h"
#include "arcane/CaseNodeNames.h"
#include "arcane/CaseOptionError.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

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

  CaseDocument(IApplication* sm,IRessourceMng* rm,IXmlDocumentHolder* document);
  ~CaseDocument() override;

  void build() override;
  ICaseDocument* clone() override;

 public:

  IXmlDocumentHolder* documentHolder() override { return m_doc_holder.get(); }

  CaseNodeNames* caseNodeNames() override { return m_case_node_names; }

  XmlNode documentNode() override { return m_document_node; }
  XmlNode rootElement() override { return m_root_elem; }
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

  const String& userClass() const override { return m_user_class; }
  void setUserClass(const String& value) override;

  const String& codeName() const override { return m_code_name; }
  void setCodeName(const String& value) override;

  const String& codeVersion() const override { return m_code_version; }
  void setCodeVersion(const String& value) override;

  const String& codeUnitSystem() const override { return m_code_unit_system; }
  void setCodeUnitSystem(const String& value) override;

  const String& defaultCategory() const override { return m_default_category; }
  void setDefaultCategory(const String& v) override { m_default_category = v; }

  const String& language() const override { return m_language; }

  void addError(const CaseOptionError& case_error) override;
  void addWarning(const CaseOptionError& case_error) override;
  bool hasError() const override;
  bool hasWarnings() const override;
  void printErrors(std::ostream& o) override;
  void printWarnings(std::ostream& o) override;
  void clearErrorsAndWarnings() override;

 public:
  // Positionne la langue. Doit être fait avant l'appel à build.
  void setLanguage(const String& language)
  {
    if (!m_language.null())
      ARCANE_FATAL("Language already set");
    m_language = language;
  }
 private:

  CaseNodeNames* m_case_node_names;
  ScopedPtrT<IXmlDocumentHolder> m_doc_holder;
  XmlNode m_document_node;
  XmlNode m_root_elem; 
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
  String m_language;
  String m_default_category;

  UniqueArray<CaseOptionError> m_errors;
  UniqueArray<CaseOptionError> m_warnings;

 private:
  
  XmlNode _forceCreateChild(XmlNode& parent,const String& us);
  void _assignLanguage(const String& langname);
  void _printErrorsOrWarnings(std::ostream& o,ConstArrayView<CaseOptionError> errors);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ICaseDocument*
arcaneCreateCaseDocument(IApplication* sm,IXmlDocumentHolder* document)
{
  IRessourceMng* rm = sm->ressourceMng();
  ICaseDocument* doc = new CaseDocument(sm,rm,document);
  doc->build();
  return doc;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ICaseDocument*
arcaneCreateCaseDocument(IApplication* sm,const String& lang)
{
  IRessourceMng* rm = sm->ressourceMng();
  IXmlDocumentHolder* xml_doc = rm->createXmlDocument();
  CaseDocument* doc = new CaseDocument(sm,rm,xml_doc);
  if (!lang.null())
    doc->setLanguage(lang);
  doc->build();
  return doc;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseDocument::
CaseDocument(IApplication* sm,IRessourceMng* rm,IXmlDocumentHolder* document)
: TraceAccessor(sm->traceMng())
, m_case_node_names(new CaseNodeNames(String()))
, m_doc_holder(document)
, m_document_node(m_doc_holder->documentNode())
{
  ARCANE_UNUSED(rm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseDocument::
~CaseDocument()
{
  delete m_case_node_names;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseDocument::
build()
{
  CaseNodeNames* cnn = caseNodeNames();

  m_root_elem = m_document_node.documentElement();
  if (m_root_elem.null()){
    if (m_language.null())
      m_language = String("fr");
    _assignLanguage(m_language);
    cnn = caseNodeNames();
    // Nouveau cas, pour l'instant langue francaise par défaut.
    m_root_elem = _forceCreateChild(m_document_node,cnn->root);
    m_root_elem.setAttrValue(cnn->lang_attribute,m_language);
  }

  // Ces noeuds ont un nom indépendant du langage.
  m_arcane_elem = _forceCreateChild(m_root_elem,"arcane");
  m_configuration_elem = _forceCreateChild(m_arcane_elem,"configuration");
  m_language = m_root_elem.attrValue(cnn->lang_attribute);

  if (m_language.null()){
    fatal() << "Attribute '" << cnn->lang_attribute
            << "' not specified in the element <"
            << m_root_elem.name() << ">";
    //m_root_elem.setAttrValue(cnn->lang_attribute,us("fr"));
    //m_language = m_root_elem.attrValue(cnn->lang_attribute);
  }
  else
    _assignLanguage(m_language);

  // Nécessaire car _assignLanguage() détruit l'ancien
  cnn = caseNodeNames();

  // NOTE: Si on ajoute ou change des éléments, il faut mettre
  // à jour la conversion correspondante dans CaseDocumentLangTranslator
  m_timeloop_elem = _forceCreateChild(m_arcane_elem,cnn->timeloop);
  m_title_elem = _forceCreateChild(m_arcane_elem,cnn->title);
  m_description_elem = _forceCreateChild(m_arcane_elem,cnn->description);
  m_modules_elem = _forceCreateChild(m_arcane_elem, cnn->modules);
  m_services_elem = _forceCreateChild(m_arcane_elem, cnn->services);

  _forceCreateChild(m_root_elem,cnn->mesh);
  m_mesh_elems = m_root_elem.children(cnn->mesh);

  m_functions_elem = _forceCreateChild(m_root_elem,cnn->functions);
  m_meshes_elem = m_root_elem.child(cnn->meshes);

  m_user_class = m_root_elem.attrValue(cnn->user_class);
  m_code_name = m_root_elem.attrValue(cnn->code_name);
  m_code_version = m_root_elem.attrValue(cnn->code_version);
  m_code_unit_system = m_root_elem.attrValue(cnn->code_unit);
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

void CaseDocument::
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
  m_root_elem.setAttrValue(caseNodeNames()->user_class,value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseDocument::
setCodeName(const String& value)
{
  m_code_name = value;
  m_root_elem.setAttrValue(caseNodeNames()->code_name,value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseDocument::
setCodeVersion(const String& value)
{
  m_code_version = value;
  m_root_elem.setAttrValue(caseNodeNames()->code_version,value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseDocument::
setCodeUnitSystem(const String& value)
{
  m_code_unit_system = value;
  m_root_elem.setAttrValue(caseNodeNames()->code_unit,value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseDocument::
addError(const CaseOptionError& case_error)
{
  m_errors.add(case_error);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseDocument::
addWarning(const CaseOptionError& case_error)
{
  m_warnings.add(case_error);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool CaseDocument::
hasError() const
{
  return m_errors.size()!=0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool CaseDocument::
hasWarnings() const
{
  return m_warnings.size()!=0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseDocument::
printErrors(std::ostream& o)
{
  _printErrorsOrWarnings(o,m_errors);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseDocument::
printWarnings(std::ostream& o)
{
  _printErrorsOrWarnings(o,m_warnings);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseDocument::
clearErrorsAndWarnings()
{
  m_errors.clear();
  m_warnings.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseDocument::
_printErrorsOrWarnings(std::ostream& o,ConstArrayView<CaseOptionError> errors)
{
  for( const CaseOptionError& error : errors.range() ){
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
