// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseDocumentLangTranslator.cc                               (C) 2000-2022 */
/*                                                                           */
/* Classe gérant la traduction d'un jeu de données dans une autre langue.    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Collection.h"
#include "arcane/utils/Enumerator.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/impl/CaseDocumentLangTranslator.h"

#include "arcane/AbstractCaseDocumentVisitor.h"
#include "arcane/CaseOptions.h"
#include "arcane/CaseOptionService.h"
#include "arcane/ICaseMng.h"
#include "arcane/ICaseDocument.h"
#include "arcane/CaseNodeNames.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CaseDocumentLangTranslatorVisitor
: public AbstractCaseDocumentVisitor
, public TraceAccessor
{
 public:
  CaseDocumentLangTranslatorVisitor(ITraceMng* tm,const String& new_lang)
  : TraceAccessor(tm)
  {
    m_new_lang = new_lang;
  }
 public:
  void beginVisit(const ICaseOptions* opt) override
  {
    info() << "BeginOpt " << _getName(opt) << " {";
  }
  void endVisit(const ICaseOptions* opt) override
  {
    info() << "EndOptList " << opt->rootTagName() << " }";
  }
  void applyVisitor(const CaseOptionSimple* opt) override
  {
    info() << "SimpleOpt " << _getName(opt);
  }
  void applyVisitor(const CaseOptionMultiSimple* opt) override
  {
    info() << "MultiSimple " << _getName(opt);
  }
  void applyVisitor(const CaseOptionMultiExtended* opt) override
  {
    info() << "MultiExtended " << _getName(opt);
  }
  void applyVisitor(const CaseOptionExtended* opt) override
  {
    info() << "Extended " << _getName(opt);
  }
  void applyVisitor(const CaseOptionMultiEnum* opt) override
  {
    info() << "MultiEnum " << _getName(opt);
    info() << "WARNING: MultiEnum not handled in translator";
  }
  void applyVisitor(const CaseOptionEnum* opt) override
  {
    info() << "Enum " << _getName(opt);
    _manageEnum(opt);
  }
  void beginVisit(const CaseOptionServiceImpl* opt) override
  {
    info() << "BeginService " << _getName(opt);
  }
  void endVisit(const CaseOptionServiceImpl* opt) override
  {
    info() << "EndService " << _getName(opt);
  }
  void beginVisit(const CaseOptionMultiServiceImpl* opt,Integer index) override
  {
    info() << "BeginMultiService " << _getName(opt) << " index=" << index;
  }
  void endVisit(const CaseOptionMultiServiceImpl* opt,Integer index) override
  {
    info() << "EndMultiService " << _getName(opt) << " index=" << index;
  }
  String _getName(const CaseOptionBase* opt)
  {
    String full_xpath = opt->rootElement().xpathFullName();
    String name = opt->name();
    String new_name = opt->translatedName(m_new_lang);

    if (name!=new_name)
      m_stream() << full_xpath << "/" << name << ":" << new_name << '\n';
    return name;
  }
  String _getName(const ICaseOptions* opt)
  {
    String full_xpath = opt->configList()->rootElement().xpathFullName();
    String name = opt->rootTagName();
    String new_name = opt->translatedName(m_new_lang);

    if (name!=new_name)
      m_stream() << full_xpath << ":" << new_name << '\n';
    return name;
  }
  void _manageEnum(const CaseOptionEnum* opt)
  {
    // Rien à convertir si aucune élément associé à l'option
    // ou si valeur invalide.
    if (!opt->isPresent())
      return;
    if (!opt->hasValidValue())
      return;

    int v = opt->enumValueAsInt();
    String new_name = opt->enumValues()->nameOfValue(v,m_new_lang);
    m_stream() << opt->xpathFullName() << ":text#" << new_name << '\n';
  }
 public:
  void printAll()
  {
    info() << "ALL: " << m_stream.str();
  }
 public:
  String convertString() { return m_stream.str(); }
 private:
  OStringStream m_stream;
  String m_new_lang;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseDocumentLangTranslator::
CaseDocumentLangTranslator(ITraceMng* tm)
: TraceAccessor(tm)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseDocumentLangTranslator::
~CaseDocumentLangTranslator()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseDocumentLangTranslator::
build()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String CaseDocumentLangTranslator::
translate(ICaseMng* cm,const String& new_lang)
{
  if (new_lang.null())
    throw ArgumentException(A_FUNCINFO,"Invalid value for langage");
  CaseDocumentLangTranslatorVisitor my_visitor(traceMng(),new_lang);
  CaseOptionsCollection opts = cm->blocks();
  for( CaseOptionsCollection::Enumerator i(opts); ++i; ){
    ICaseOptions* o = *i;
    info() << " OptName=" << o->rootTagName();
    o->visit(&my_visitor);
  }
  my_visitor.printAll();

  ICaseDocument* cd = cm->caseDocument();
  CaseNodeNames* current_cnn = cd->caseNodeNames();
  ScopedPtrT<CaseNodeNames> cnn { new CaseNodeNames(new_lang) };

  // NOTE: Ces conversions dépendent de CaseDocument et doivent être
  // mise à jour si ce dernier change (ainsi que CaseNodeNames)

  _addConvert(cd->fragment()->rootElement(),cnn->root);
  _addConvert(cd->timeloopElement(),cnn->timeloop);
  _addConvert(cd->titleElement(),cnn->title);
  _addConvert(cd->descriptionElement(),cnn->description);
  _addConvert(cd->modulesElement(),cnn->modules);

  String slash = "/";
  const XmlNodeList& mesh_elems = cd->meshElements();
  for( Integer i=0, n=mesh_elems.size(); i<n; ++i ){
    XmlNode xnode(mesh_elems.node(i));
    _addConvert(xnode,cnn->mesh);
    _addConvert(xnode.child(current_cnn->mesh_file),cnn->mesh_file);
  }

  _addConvert(cd->functionsElement(),cnn->functions);

  // TODO: Gerer TiedInterface + CaseFunctions + Attributs suivants:
  // TODO: Utiliser le format JSON pour sortir les informations
  // de conversion.

  /*m_user_class = m_root_elem.attrValue(cnn->user_class);
  m_code_name = m_root_elem.attrValue(cnn->code_name);
  m_code_version = m_root_elem.attrValue(cnn->code_version);
  m_code_unit_system = m_root_elem.attrValue(cnn->code_unit);*/

  return my_visitor.convertString() + m_global_convert_string;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseDocumentLangTranslator::
_addConvert(XmlNode node,const String& new_name)
{
  if (!node.null())
    m_global_convert_string = m_global_convert_string + node.xpathFullName() + ":" + new_name + "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
