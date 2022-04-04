// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PrintCaseDocumentVisitor.h                                  (C) 2000-2019 */
/*                                                                           */
/* Visiteur pour afficher les valeurs du jeu de données.                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotImplementedException.h"

#include "arcane/AbstractCaseDocumentVisitor.h"

#include "arcane/CaseOptions.h"
#include "arcane/CaseOptionService.h"
#include "arcane/ICaseFunction.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Visiteur pour afficher les valeurs du jeu de données.
 */
class PrintCaseDocumentVisitor
: public AbstractCaseDocumentVisitor
{
 public:
  struct Indent
  {
    Indent(int n): m_n(n) {}
    int m_n;
  };
 public:
  PrintCaseDocumentVisitor(ITraceMng* tm,const String& lang)
  : m_trace_mng(tm), m_lang(lang)
  {
  }
  void beginVisit(const ICaseOptions* opt) override;
  void endVisit(const ICaseOptions* opt) override;
  void applyVisitor(const CaseOptionSimple* opt) override
  {
    _printOption(opt);
  }
  void applyVisitor(const CaseOptionMultiSimple* opt) override
  {
    _printOption(opt);
  }
  void applyVisitor(const CaseOptionExtended* opt) override
  {
    _printOption(opt);
  }
  void applyVisitor(const CaseOptionMultiExtended* opt) override
  {
    _printOption(opt);
  }
  void applyVisitor(const CaseOptionEnum* opt) override
  {
    _printOption(opt);
  }
  void applyVisitor(const CaseOptionMultiEnum* opt) override
  {
    _printOption(opt);
  }
  void beginVisit(const CaseOptionServiceImpl* opt) override
  {
    //std::cout << "BEGIN_VISIT SERVICE name=" << opt->name() << "\n";
    // Le visiteur appelle d'abord le service puis le ICaseOptions associé
    // à ce service
    m_current_service_name = opt->serviceName();
  }
  void endVisit(const CaseOptionServiceImpl* opt) override
  {
    ARCANE_UNUSED(opt);
    //std::cout << "END_VISIT SERVICE name=" << opt->name() << "\n";  
  }
  void beginVisit(const CaseOptionMultiServiceImpl* opt,Integer index) override
  {
    //std::cout << "WARNING: BEGIN MULTI_SERVICE index=" << index << "\n";
    m_current_service_name = opt->serviceName(index);
    //opt->print(m_lang,m_stream);
  }
  void endVisit(const CaseOptionMultiServiceImpl* opt,Integer index) override
  {
    ARCANE_UNUSED(opt);
    ARCANE_UNUSED(index);
    //std::cout << "WARNING: END MULTI_SERVICE\n";
    //opt->print(m_lang,m_stream);
  }
 protected:
  void _printOption(const CaseOptionBase* co)
  {
    m_stream = std::ostringstream();
    std::ostream& o = m_stream;
    _printOption(co,o);
    m_trace_mng->info() << m_stream.str();
  }
  void _printOption(const CaseOptionBase* co,std::ostream& o);
 private:
  ITraceMng* m_trace_mng;
  String m_lang;
  std::ostringstream m_stream;
  int m_indent = 0;
  String m_current_service_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline std::ostream&
operator<< (std::ostream& o, const PrintCaseDocumentVisitor::Indent& indent)
{
  for( int i=0; i<indent.m_n; ++i )
    o << ' ';
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PrintCaseDocumentVisitor::
beginVisit(const ICaseOptions* opt)
{
  String service_name;
  if (!m_current_service_name.null()){
    service_name = " name=\""+ m_current_service_name + "\"";
  }
  else {
    IServiceInfo* service = opt->caseServiceInfo();
    if (service)
      m_trace_mng->info() << "WARNING: service_name not handled name=\""+ service->localName() + "\"";
  }
  m_current_service_name = String();
  m_trace_mng->info() << Indent(m_indent) << "<" << opt->translatedName(m_lang) << service_name << ">";
  ++m_indent;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PrintCaseDocumentVisitor::
endVisit(const ICaseOptions* opt)
{
  --m_indent;
  m_trace_mng->info() << Indent(m_indent) << "</" << opt->translatedName(m_lang) << ">";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PrintCaseDocumentVisitor::
_printOption(const CaseOptionBase* co,std::ostream& o)
{
  std::ios_base::fmtflags f = o.flags(std::ios::left);
  o << " ";
  o << Indent(m_indent);
  o.width(40-m_indent);
  o << co->translatedName(m_lang);
  co->print(m_lang,o);
  ICaseFunction* func = co->function();
  if (func){
    o << " (fonction: " << func->name() << ")";
  }
  o.flags(f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" std::unique_ptr<ICaseDocumentVisitor>
createPrintCaseDocumentVisitor(ITraceMng* tm,const String& lang)
{
  return std::make_unique<PrintCaseDocumentVisitor>(tm,lang);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
