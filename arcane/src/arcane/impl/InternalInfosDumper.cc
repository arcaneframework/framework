// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* InternalInfosDumper.cc                                      (C) 2000-2025 */
/*                                                                           */
/* Sorties des informations internes de Arcane.                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/InternalInfosDumper.h"

#include "arcane/utils/String.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/JSONWriter.h"
#include "arcane/utils/FileContent.h"

#include "arcane/Parallel.h"
#include "arcane/IParallelMng.h"
#include "arcane/Directory.h"
#include "arcane/ServiceFinder2.h"
#include "arcane/IApplication.h"
#include "arcane/ICodeService.h"
#include "arcane/SubDomainBuildInfo.h"
#include "arcane/IServiceLoader.h"
#include "arcane/IParallelSuperMng.h"
#include "arcane/IMainFactory.h"
#include "arcane/ISession.h"
#include "arcane/ISubDomain.h"
#include "arcane/ITimeLoopMng.h"
#include "arcane/IVariableMng.h"
#include "arcane/IXmlDocumentHolder.h"
#include "arcane/XmlNodeList.h"
#include "arcane/IModuleMng.h"
#include "arcane/IModule.h"
#include "arcane/VariableRef.h"
#include "arcane/VariableCollection.h"
#include "arcane/IIOMng.h"
#include "arcane/ITimeLoop.h"
#include "arcane/ICaseMng.h"
#include "arcane/ICaseOptions.h"
#include "arcane/IRessourceMng.h"

#include "arcane/impl/TimeLoopReader.h"

#include <set>
#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

extern "C++" ARCANE_IMPL_EXPORT Ref<ICodeService>
createArcaneCodeService(IApplication* app);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

InternalInfosDumper::
InternalInfosDumper(IApplication* application)
: m_application(application)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ICodeService> InternalInfosDumper::
_getDefaultService()
{
  String code_name = platform::getEnvironmentVariable("STDENV_CODE_NAME");
  if (code_name.null())
    ARCANE_FATAL("environment variable 'STDENV_CODE_NAME' is not defined");

  IApplication* app = m_application;
  ServiceFinder2T<ICodeService,IApplication> code_services_utils(app,app);
  String full_code_name = code_name + "Code";
  Ref<ICodeService> code = code_services_utils.createReference(full_code_name);
  if (!code){
    // Le service de code de nom Arcane est spécial et sert pour générer
    // la documentation interne. S'il n'est pas enregistré, on créé directement
    // l'instance.
    if (code_name=="Arcane")
      code = createArcaneCodeService(app);
  }
  if (!code)
    ARCANE_FATAL("No code service named '{0}' found",full_code_name);
  return code;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void InternalInfosDumper::
dumpInternalInfos()
{
  ITraceMng* tr = m_application->traceMng();
  IMainFactory* main_factory = m_application->mainFactory();
  tr->info() << "Sortie des infos internes à Arcane";
  Ref<ICodeService> code_service = _getDefaultService();
  ISession* session(code_service->createSession());
  IParallelSuperMng* psm = m_application->parallelSuperMng();
  Ref<IParallelMng> world_pm = psm->internalCreateWorldParallelMng(0);
  SubDomainBuildInfo sdbi(world_pm,0);
  sdbi.setCaseFileName(String());
  sdbi.setCaseBytes(ByteConstArrayView());
  ISubDomain* sub_domain(session->createSubDomain(sdbi));
  ScopedPtrT<IServiceLoader> service_loader(main_factory->createServiceLoader());
  // Les services sont déjà enregistrées lors de la création du sous-domaine
  {
    TimeLoopReader stl(m_application);
    stl.readTimeLoops();
    stl.registerTimeLoops(sub_domain);
  }
  // Charge tous les modules disponibles
  service_loader->loadModules(sub_domain,true);

  // Conserve dans le tableau tous les IServiceInfo.
  UniqueArray<IServiceInfo*> service_infos;
  {
    std::set<IServiceInfo*> done_set;
    for( ServiceFactory2Collection::Enumerator j(m_application->serviceFactories2()); ++j; ){
      Internal::IServiceFactory2* sf2 = *j;
      IServiceInfo* s = sf2->serviceInfo();
      if (done_set.find(s)==done_set.end()){
        service_infos.add(s);
        done_set.insert(s);
      }
    }
  }

  IModuleMng* module_mng = sub_domain->moduleMng();
  IVariableMng* variable_mng = sub_domain->variableMng();

  VariableRefList var_ref_list;

  String us_name("name");
  String us_ref("ref");
  String us_datatype("datatype");
  String us_dimension("dimension");
  String us_kind("kind");
  String us_root("root");
  String us_modules("modules");
  String us_module("module");
  String us_services("services");
  String us_service("service");

  ScopedPtrT<IXmlDocumentHolder> doc_holder(m_application->ressourceMng()->createXmlDocument());
  XmlNode doc_element = doc_holder->documentNode();
  XmlNode root_element = doc_element.createAndAppendElement(us_root);

  XmlNode modules = root_element.createAndAppendElement(us_modules);
  // Liste des modules avec les variables qu'ils utilisent.
  for( ModuleCollection::Enumerator i(module_mng->modules()); ++i; ){
    XmlNode module_element = modules.createAndAppendElement(us_module);
    module_element.setAttrValue(us_name,String((*i)->name()));
    var_ref_list.clear();
    variable_mng->variables(var_ref_list,*i);
    for( VariableRefList::Enumerator j(var_ref_list); ++j; ){
      XmlNode variable_element = module_element.createAndAppendElement("variable-ref");
      variable_element.setAttrValue(us_ref,String((*j)->name()));
    }
  }

  XmlNode variables = root_element.createAndAppendElement("variables");
  // Liste des variables.
  VariableCollection var_prv_list(variable_mng->variables());
  for( VariableCollection::Enumerator j(var_prv_list); ++j; ){
    IVariable* var = *j;
    String dim(String::fromNumber(var->dimension()));
    XmlNode variable_element = variables.createAndAppendElement("variable");
    variable_element.setAttrValue(us_name,var->name());
    variable_element.setAttrValue(us_datatype,dataTypeName(var->dataType()));
    variable_element.setAttrValue(us_dimension,dim);
    variable_element.setAttrValue(us_kind,itemKindName(var->itemKind()));
  }

  // Liste des boucles en temps
  ITimeLoopMng* tm = sub_domain->timeLoopMng();
  StringList timeloop_name_list;
  tm->timeLoopsName(timeloop_name_list);

  XmlNode timeloops = root_element.createAndAppendElement("timeloops");
  for( StringCollection::Enumerator i(timeloop_name_list); ++i; ){
    XmlNode timeloop_elem = timeloops.createAndAppendElement("timeloop");
    timeloop_elem.setAttrValue(us_name,*i);
  }

  // Liste des services
  {
    // Liste des services qui implementent une interface donnée
    std::map<String,List<IServiceInfo*> > interfaces_to_service;
    XmlNode services_elem = root_element.createAndAppendElement(us_services);
    
    for( int i=0, n=service_infos.size(); i<n; ++i ){
      IServiceInfo* service_info = service_infos[i];
      XmlNode service_elem = services_elem.createAndAppendElement(us_service);
      service_elem.setAttrValue(us_name,service_info->localName());
      {
        auto xml_file_base_name = service_info->caseOptionsFileName();
        if (!xml_file_base_name.null()){
          service_elem.setAttrValue("file-base-name",xml_file_base_name);
        }
      }
      for( StringCollection::Enumerator j(service_info->implementedInterfaces()); ++j; ){
        XmlNode interface_elem = service_elem.createAndAppendElement("implement-class");
        interface_elem.setAttrValue(us_name,*j);
        interfaces_to_service[*j].add(service_info);
      }
    }
    
    {
      XmlNode classes_elem = services_elem.createAndAppendElement("services-class");
      std::map<String,List<IServiceInfo*> >::const_iterator begin = interfaces_to_service.begin();
      std::map<String,List<IServiceInfo*> >::const_iterator end = interfaces_to_service.end();
      for( ; begin!=end; ++begin ){
        XmlNode class_elem = classes_elem.createAndAppendElement("class");
        class_elem.setAttrValue(us_name,begin->first);
        for( List<IServiceInfo*>::Enumerator i(begin->second); ++i; ){
          IServiceInfo* service_info = *i;
          XmlNode service_elem = class_elem.createAndAppendElement(us_service);
          service_elem.setAttrValue(us_name,service_info->localName());
          {
            auto xml_file_base_name = service_info->caseOptionsFileName();
            if (!xml_file_base_name.null()){
              service_elem.setAttrValue("file-base-name",xml_file_base_name);
            }
          }
        }
      }
    }
  }

  Directory shared_dir(m_application->applicationInfo().dataDir());
  String filename = shared_dir.file("arcane_internal.xml");
  cerr << "** FILE IS " << filename << '\n';
  sub_domain->ioMng()->writeXmlFile(doc_holder.get(),filename);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void InternalInfosDumper::
dumpInternalAllInfos()
{
  m_application->traceMng()->info() << "Sortie des infos sur les boucles en temps";
  IMainFactory* main_factory = m_application->mainFactory();
  Ref<ICodeService> code_service = _getDefaultService();

  ByteConstSpan config_bytes = m_application->configBuffer();
  ScopedPtrT<IXmlDocumentHolder> config_doc(m_application->ioMng()->parseXmlBuffer(config_bytes,String()));
  if (!config_doc.get())
    ARCANE_FATAL("Can not parse code configuration file");
  XmlNode elem = config_doc->documentNode().documentElement();

  // La boucle en temps est spécifiée dans le fichier de config.
  // XmlNode elem = m_application->configRootElement();
  elem = elem.child("Execution");
  elem = elem.child("BouclesTemps");

  ScopedPtrT<IServiceLoader> service_loader(main_factory->createServiceLoader());

  ScopedPtrT<IXmlDocumentHolder> doc_holder(m_application->ressourceMng()->createXmlDocument());
  XmlNode doc_element = doc_holder->documentNode();
  XmlNode root_element = doc_element.createAndAppendElement(String("root"));

  String us_name("name");

  TimeLoopReader stl(m_application);
  stl.readTimeLoops();

  {
    const ApplicationInfo& app_info = m_application->applicationInfo();
    //! Informations générales
    XmlNode elem2(root_element.createAndAppendElement("general"));
    elem2.createAndAppendElement("codename",String(app_info.codeName()));
    elem2.createAndAppendElement("codefullversion",String(m_application->mainVersionStr()));
    elem2.createAndAppendElement("codeversion",String(m_application->majorAndMinorVersionStr()));
  }

  IParallelSuperMng* psm = m_application->parallelSuperMng();
  Ref<IParallelMng> world_pm = psm->internalCreateWorldParallelMng(0);

  for( TimeLoopCollection::Enumerator i(stl.timeLoops()); ++i; ){
    ITimeLoop* timeloop = *i;
    const String& timeloop_name = timeloop->name();
    
    ISession* session = nullptr;
    ISubDomain* sd = nullptr;
    try{
      session = code_service->createSession();
      service_loader->loadSessionServices(session);
      SubDomainBuildInfo sdbi(world_pm,0);
      sdbi.setCaseFileName(String());
      sdbi.setCaseBytes(ByteConstArrayView());
      sd = session->createSubDomain(sdbi);
      //service_loader->loadSubDomainServices(sd);

      ITimeLoopMng* loop_mng = sd->timeLoopMng();
      loop_mng->registerTimeLoop(timeloop);
      loop_mng->setUsedTimeLoop(timeloop_name);
    }
    catch(...){
      session = nullptr;
      sd = nullptr;
    }
    if (!sd)
      continue;
    XmlNode mng_element = root_element.createAndAppendElement("timeloop");
    mng_element.setAttrValue(us_name,timeloop_name);
    sd->dumpInternalInfos(mng_element);
  }

  Directory shared_dir(m_application->applicationInfo().dataDir());
  String filename = shared_dir.file("arcane-caseinfos.xml");
  cerr << "** FILE IS " << filename << '\n';
  m_application->ioMng()->writeXmlFile(doc_holder.get(),filename);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void InternalInfosDumper::
_dumpSubDomainInternalInfos(ISubDomain* sd,JSONWriter& json_writer)
{
  // Info sur la boucle en temps utilisée.
  {
    ITimeLoopMng* tlm = sd->timeLoopMng();
    ITimeLoop* time_loop = tlm->usedTimeLoop();
    JSONWriter::Object jo_timeloopinfo(json_writer,"timeloopinfo");
    json_writer.write("title",time_loop->title());
    json_writer.write("description",time_loop->description());

    JSONWriter::Array ja_userclass(json_writer,"userclass");
    for( StringCollection::Enumerator j(time_loop->userClasses()); ++j; ){
      json_writer.writeValue(*j);
    }
  }

  String ustr_module("module");
  String ustr_name("name");
  String ustr_activated("activated");
  String ustr_variable("variable");
  String ustr_variable_ref("variable-ref");
  String ustr_ref("ref");
  String ustr_datatype("datatype");
  String ustr_dimension("dimension");
  String ustr_kind("kind");
  String ustr_caseblock("caseblock");
  String ustr_tagname("tagname");

  IVariableMng* var_mng = sd->variableMng();

  // Liste des modules avec les variables qu'ils utilisent.
  {
    JSONWriter::Array ja_modules(json_writer,"modules");
    for( IModule* module : sd->moduleMng()->modules() ){
      JSONWriter::Object jo_module(json_writer);
      json_writer.write(ustr_name,module->name());
      json_writer.write(ustr_activated,module->used());

      VariableRefList var_ref_list;
      var_mng->variables(var_ref_list,module);
      JSONWriter::Array ja_variables(json_writer,"variables");
      for( VariableRef* vr : var_ref_list )
        json_writer.writeValue(vr->variable()->fullName());
    }
  }

  // Liste des variables.
  {
    VariableCollection var_prv_list = var_mng->variables();
    JSONWriter::Array jo_variables(json_writer,"variables");
    for( VariableCollection::Enumerator j(var_prv_list); ++j; ){
      IVariable* var = *j;
      JSONWriter::Object jo_variable(json_writer);
      json_writer.write(ustr_name,var->fullName());
      json_writer.write(ustr_datatype,dataTypeName(var->dataType()));
      json_writer.write(ustr_dimension,(Int64)var->dimension());
      json_writer.write(ustr_kind,itemKindName(var->itemKind()));
    }
  }

  // Liste des blocs d'options
  {
    ICaseMng* cm = sd->caseMng();
    CaseOptionsCollection blocks = cm->blocks();

    JSONWriter::Array ja_blocks(json_writer,"caseblocks");
    for( ICaseOptions* block : blocks ){
      JSONWriter::Object jo_block(json_writer);
      json_writer.write(ustr_tagname,block->rootTagName());

      IModule* block_module = block->caseModule();
      if (block_module)
        json_writer.write(ustr_module,block_module->name());
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Sauve les informations internes de %Arcane dans un fichier json.
 */
void InternalInfosDumper::
dumpArcaneDatabase()
{
  IApplication* app = m_application;
  ITraceMng* tr = app->traceMng();
  IMainFactory* main_factory = m_application->mainFactory();
  tr->info() << "Generating Arcane Database";
  Ref<ICodeService> code_service = _getDefaultService();
  ISession* session(code_service->createSession());
  IParallelSuperMng* psm = app->parallelSuperMng();
  Ref<IParallelMng> world_pm = psm->internalCreateWorldParallelMng(0);
  SubDomainBuildInfo sdbi(world_pm,0);
  sdbi.setCaseFileName(String());
  sdbi.setCaseBytes(ByteConstArrayView());
  ISubDomain* sub_domain(session->createSubDomain(sdbi));
  ScopedPtrT<IServiceLoader> service_loader(main_factory->createServiceLoader());

  ByteConstSpan config_bytes = m_application->configBuffer();
  ScopedPtrT<IXmlDocumentHolder> config_doc(app->ioMng()->parseXmlBuffer(config_bytes,String()));
  if (!config_doc.get())
    ARCANE_FATAL("Can not parse code configuration file");

  // Les services sont déjà enregistrées lors de la création du sous-domaine
  {
    TimeLoopReader stl(m_application);
    stl.readTimeLoops();
    stl.registerTimeLoops(sub_domain);
  }
  // Charge tous les modules disponibles
  service_loader->loadModules(sub_domain,true);

  // Conserve dans ce tableau tous les IServiceInfo créés
  UniqueArray<IServiceInfo*> service_infos;
  {
    std::set<IServiceInfo*> done_set;
    for( ServiceFactory2Collection::Enumerator j(m_application->serviceFactories2()); ++j; ){
      Internal::IServiceFactory2* sf2 = *j;
      IServiceInfo* s = sf2->serviceInfo();
      if (done_set.find(s)==done_set.end()){
        service_infos.add(s);
        done_set.insert(s);
      }
    }
  }

  IModuleMng* module_mng = sub_domain->moduleMng();
  IVariableMng* variable_mng = sub_domain->variableMng();

  String us_name("name");
  String us_ref("ref");
  String us_datatype("datatype");
  String us_dimension("dimension");
  String us_kind("kind");
  String us_root("root");
  String us_modules("modules");
  String us_module("module");
  String us_services("services");
  String us_service("service");

  ScopedPtrT<IXmlDocumentHolder> doc_holder(m_application->ressourceMng()->createXmlDocument());
  XmlNode doc_element = doc_holder->documentNode();
  XmlNode root_element = doc_element.createAndAppendElement(us_root);

  JSONWriter json_writer(JSONWriter::FormatFlags::None);
  json_writer.beginObject();
  json_writer.write("version","1");

  // Liste des modules avec les variables qu'ils utilisent.
  {
    JSONWriter::Array ja_modules(json_writer,"modules");
    for( IModule* module : module_mng->modules() ){
      JSONWriter::Object jo(json_writer);
      json_writer.write(us_name,module->name());
      VariableRefList var_ref_list;
      variable_mng->variables(var_ref_list,module);
      JSONWriter::Array ja_var_ref(json_writer,"variable-references");
      for( VariableRefList::Enumerator j(var_ref_list); ++j; ){
        json_writer.writeValue((*j)->name());
      }
    }
  }

  // Liste des variables avec leurs caractéristiques
  {
    VariableCollection var_prv_list(variable_mng->variables());
    JSONWriter::Array ja(json_writer,"variables");
    for( VariableCollection::Enumerator j(var_prv_list); ++j; ){
      IVariable* var = *j;

      JSONWriter::Object jo(json_writer);
      json_writer.write(us_name,var->name());
      json_writer.write(us_datatype,dataTypeName(var->dataType()));
      json_writer.write(us_dimension,(Int64)var->dimension());
      json_writer.write(us_kind,itemKindName(var->itemKind()));
    }
  }

  // Liste des services qui implémentent une interface donnée
  std::map<String,List<IServiceInfo*> > interfaces_to_service;

  // Liste des services
  {
    JSONWriter::Array ja_services(json_writer,us_services);
    
    for( IServiceInfo* service_info : service_infos ){
      JSONWriter::Object jo(json_writer);
      json_writer.write(us_name,service_info->localName());
      json_writer.writeIfNotNull("file-base-name",service_info->caseOptionsFileName());

      // Sauver le contenu en splittant en plusieurs chaînes de
      // caractères ce qui permet de ne pas avoir de lignes trop longues.
      Span<const Byte> content = service_info->axlContent().bytes();
      Int64 content_size = content.size();
      if (content_size>0){
        Int64 block_size = 80;
        Int64 nb_block = content_size / block_size;
        if ((content_size%block_size)!=0)
          ++nb_block;
        JSONWriter::Array ja_axl_content(json_writer,"axl-content");
        Int64 index = 0;
        for( Integer k=0; k<nb_block; ++k ){
          auto z = content.subSpan(index,block_size);
          json_writer.writeValue(z);
          index += block_size;
        }
      }

      // Sauve la listes des interfaces implémentées par ce service
      {
        JSONWriter::Array ja_implemented_interfaces(json_writer,"implemented-interfaces");
        for( StringCollection::Enumerator j(service_info->implementedInterfaces()); ++j; ){
          interfaces_to_service[*j].add(service_info);
          json_writer.writeValue(*j);
        }
      }
    }
  }

  // Liste des interfaces de services et des services les implémentant.
  {
    JSONWriter::Array jo_services_interfaces(json_writer,"service-interfaces");
    for( const auto& x : interfaces_to_service ){
      JSONWriter::Object jo_class(json_writer);
      json_writer.write(us_name,x.first);
      JSONWriter::Array ja_services(json_writer,"services");
      for( IServiceInfo* service_info : x.second ){
        JSONWriter::Object jo_service(json_writer);
        json_writer.write(us_name,service_info->localName());
        json_writer.writeIfNotNull("file-base-name",service_info->caseOptionsFileName());
      }
    }
  }

  TimeLoopReader stl(m_application);
  stl.readTimeLoops();

  {
    const ApplicationInfo& app_info = m_application->applicationInfo();
    JSONWriter::Object jo_general(json_writer,"general");
    json_writer.write("codename",app_info.codeName());
    json_writer.write("codefullversion",m_application->mainVersionStr());
    json_writer.write("codeversion",m_application->majorAndMinorVersionStr());
  }

  {
    JSONWriter::Array jo_timeloops(json_writer,"timeloops");
    for( TimeLoopCollection::Enumerator i(stl.timeLoops()); ++i; ){
      ITimeLoop* timeloop = *i;
      const String& timeloop_name = timeloop->name();
    
      JSONWriter::Object jo_timeloop(json_writer);
      json_writer.write(us_name,timeloop_name);

      ISession* session = nullptr;
      ISubDomain* sd = nullptr;
      try{
        session = code_service->createSession();
        service_loader->loadSessionServices(session);
        SubDomainBuildInfo sdbi(world_pm,0);
        sdbi.setCaseFileName(String());
        sdbi.setCaseBytes(ByteConstArrayView());
        sd = session->createSubDomain(sdbi);

        ITimeLoopMng* loop_mng = sd->timeLoopMng();
        loop_mng->registerTimeLoop(timeloop);
        loop_mng->setUsedTimeLoop(timeloop_name);
      }
      catch(...){
        session = nullptr;
        sd = nullptr;
      }
      // TODO: Indique si sd est nul.
      if (sd){
        _dumpSubDomainInternalInfos(sd,json_writer);
      }
    }
  }

  json_writer.endObject();

  {
    // Écrit le fichier JSON.
    Directory shared_dir(m_application->applicationInfo().dataDir());
    String json_filename = shared_dir.file("arcane_database.json");
    cerr << "** FILE2 IS " << json_filename << '\n';
    String buf(json_writer.getBuffer());
    // TODO: regarder s'il ne serait pas préférable de sauver le fichier
    // dans le répertoire courant.
    std::ofstream ofile(json_filename.localstr());
    buf.writeBytes(ofile);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
