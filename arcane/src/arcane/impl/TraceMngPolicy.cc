// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TraceMngPolicy.cc                                           (C) 2000-2019 */
/*                                                                           */
/* Politique de configuration des gestionnaires de trace.                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounter.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/String.h"
#include "arcane/utils/TraceClassConfig.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/Iterator.h"
#include "arcane/utils/ITraceMngPolicy.h"

#include "arcane/XmlNode.h"
#include "arcane/XmlNodeList.h"
#include "arcane/IApplication.h"
#include "arcane/IXmlDocumentHolder.h"

#include <map>
#include <mutex>
#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestion du comportement des traces.
 *
 * \warning Les instances de cette classe sont créées avant les ITraceMng.
 * Il ne faut donc pas utiliser m_application->traceMng() sauf lors de
 * l'appel à setDefaultClassConfigXmlBuffer().
 */
class TraceMngPolicy
: public ITraceMngPolicy
{
 public:
  TraceMngPolicy(IApplication* app)
  : m_application(app), m_is_parallel(false), m_is_debug(false),
    m_is_master_has_output_file(false),
    m_stdout_verbosity_level(Trace::UNSPECIFIED_VERBOSITY_LEVEL),
    m_verbosity_level(Trace::UNSPECIFIED_VERBOSITY_LEVEL),
    m_is_parallel_output(false)
  {
    m_default_config_doc = IXmlDocumentHolder::createNull();
    m_output_file_prefix = platform::getEnvironmentVariable("ARCANE_PARALLEL_OUTPUT_PREFIX");
  }

  ~TraceMngPolicy()
  {
  }

  void build() override {}
  void initializeTraceMng(ITraceMng* trace,Int32 rank) override;
  void initializeTraceMng(ITraceMng* trace,ITraceMng* parent_trace,const String& file_suffix) override;
  void setClassConfigFromXmlBuffer(ITraceMng* trace,ByteConstArrayView bytes)  override;
  void setIsParallel(bool v) override { m_is_parallel = v; }
  bool isParallel() const override { return m_is_parallel; }
  void setIsDebug(bool v) override { m_is_debug = v; }
  bool isDebug() const override { return m_is_debug; }
  void setIsParallelOutput(bool v) override { m_is_parallel_output = v; }
  bool isParallelOutput() const override { return m_is_parallel_output; }
  void setStandardOutputVerbosityLevel(Int32 level) override { m_stdout_verbosity_level = level; }
  Int32 standardOutputVerbosityLevel() const override { return m_stdout_verbosity_level; }
  void setVerbosityLevel(Int32 level) override { m_verbosity_level = level; }
  Int32 verbosityLevel() const override { return m_verbosity_level; }
  void setIsMasterHasOutputFile(bool active) override { m_is_master_has_output_file = active; }
  bool isMasterHasOutputFile() const override { return m_is_master_has_output_file; }
  void setDefaultVerboseLevel(ITraceMng* trace,Int32 minimal_level) override;
  void setDefaultClassConfigXmlBuffer(ByteConstSpan bytes) override;

 private:
  
  IApplication* m_application;
  bool m_is_parallel;
  bool m_is_debug;
  bool m_is_master_has_output_file;
  Int32 m_stdout_verbosity_level;
  Int32 m_verbosity_level;
  std::map<String,Arccore::ReferenceCounter<ITraceStream>> m_output_files;
  String m_output_file_prefix;
  bool m_is_parallel_output;
  ScopedPtrT<IXmlDocumentHolder> m_default_config_doc;
  std::mutex m_init_mutex;
  std::mutex m_getfile_mutex;

 private:

  ITraceStream* _getFile(const String& rank);
  void _initializeTraceClasses(ITraceMng* trace);
  void _setAllTraceClassConfig(ITraceMng* trace,ByteConstArrayView bytes,bool do_log);
  void _setAllTraceClassConfig(ITraceMng* trace,XmlNode root_element,bool do_log);
  void _setAllTraceClassConfig(ITraceMng* trace,IXmlDocumentHolder* doc,bool do_log);
  void _initializeTraceMng(ITraceMng* trace,bool is_master,const String& rank_str);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_IMPL_EXPORT ITraceMngPolicy*
arcaneCreateTraceMngPolicy(IApplication* app)
{
  ITraceMngPolicy* itmp = new TraceMngPolicy(app);
  itmp->build();
  return itmp;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMngPolicy::
initializeTraceMng(ITraceMng* trace,Int32 rank)
{
  _initializeTraceMng(trace,rank==0,String::fromNumber(rank));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMngPolicy::
_initializeTraceMng(ITraceMng* trace,bool is_master,const String& rank_str)
{
  bool is_output_file = false;
  {
    StringBuilder trace_id;
    trace_id += rank_str;
    trace_id += ",";
    trace_id += platform::getHostName();
    trace->setTraceId(trace_id.toString());
  }
  // Par défaut si rien n'est spécifié, seul le proc maitre sort les infos.
  bool is_info_disabled = !is_master;
  if (m_is_parallel_output){
    is_info_disabled = false;
    is_output_file = true;
  }
  if (is_master && m_is_master_has_output_file)
    is_output_file = true;

  trace->setInfoActivated(!is_info_disabled);

  if (is_output_file){
    ITraceStream* ofile = _getFile(rank_str);
    trace->setRedirectStream(ofile);
  }

  _initializeTraceClasses(trace);
  trace->setMaster(is_master);
  trace->finishInitialize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMngPolicy::
initializeTraceMng(ITraceMng* trace,ITraceMng* parent_trace,const String& file_suffix)
{
  _initializeTraceMng(trace,false,file_suffix);
  if (parent_trace){
    trace->setVerbosityLevel(parent_trace->verbosityLevel());
    trace->setStandardOutputVerbosityLevel(parent_trace->standardOutputVerbosityLevel());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMngPolicy::
setDefaultVerboseLevel(ITraceMng* trace,Int32 minimal_level)
{
  if (m_verbosity_level!=Trace::UNSPECIFIED_VERBOSITY_LEVEL){
    Int32 level = m_verbosity_level;
    if (minimal_level!=Trace::UNSPECIFIED_VERBOSITY_LEVEL)
      level = std::max(level,minimal_level);
    trace->setVerbosityLevel(level);
  }
  if (m_stdout_verbosity_level!=Trace::UNSPECIFIED_VERBOSITY_LEVEL){
    Int32 level = m_stdout_verbosity_level;
    if (minimal_level!=Trace::UNSPECIFIED_VERBOSITY_LEVEL)
      level = std::max(level,minimal_level);
    trace->setStandardOutputVerbosityLevel(level);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMngPolicy::
setClassConfigFromXmlBuffer(ITraceMng* trace,ByteConstArrayView bytes)
{
  _setAllTraceClassConfig(trace,bytes,false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMngPolicy::
setDefaultClassConfigXmlBuffer(ByteConstSpan bytes)
{
  if (bytes.empty()){
    m_default_config_doc = IXmlDocumentHolder::createNull();
    return;
  }

  ITraceMng* tm = m_application->traceMng();
  m_default_config_doc = IXmlDocumentHolder::loadFromBuffer(bytes,String(),tm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITraceStream* TraceMngPolicy::
_getFile(const String& rank)
{
  std::lock_guard<std::mutex> guard(m_getfile_mutex);

  auto i = m_output_files.find(rank);
  if (i!=m_output_files.end())
    return i->second.get();

  StringBuilder buf(m_output_file_prefix);
  if (!m_output_file_prefix.null()) {
    buf += "/";
  }
  buf += "output";
  buf += rank;
  String bufstr = buf.toString();

  ReferenceCounter<ITraceStream> stream(ITraceStream::createFileStream(bufstr));
  m_output_files.insert(std::make_pair(rank,stream));
  return stream.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMngPolicy::
_initializeTraceClasses(ITraceMng* trace)
{
  _setAllTraceClassConfig(trace,m_default_config_doc.get(),true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMngPolicy::
_setAllTraceClassConfig(ITraceMng* trace,ByteConstArrayView bytes,bool do_log)
{
  ScopedPtrT<IXmlDocumentHolder> config_doc;
  if (!bytes.empty()){
    // Trace pour afficher les informations lors de la lecture.
    // Ne pas confondre avec \a trace passé en argument
    ITraceMng* print_tm = m_application->traceMng();
    config_doc = IXmlDocumentHolder::loadFromBuffer(bytes,String(),print_tm);
  }
  _setAllTraceClassConfig(trace,config_doc.get(),do_log);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMngPolicy::
_setAllTraceClassConfig(ITraceMng* trace,IXmlDocumentHolder* doc,bool do_log)
{
  // Il faut toujours appeler _setAllTraceClassConfig même si on n'a pas
  // d'élément racine sinon l'initialisation par défaut est incorrecte.
  XmlNode root_element;
  if (doc){
    XmlNode document_node = m_default_config_doc->documentNode();
    if (!document_node.null())
      root_element = document_node.documentElement();
  }
  _setAllTraceClassConfig(trace,root_element,do_log);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TraceMngPolicy::
_setAllTraceClassConfig(ITraceMng* trace,XmlNode root_element,bool do_log)
{
  if (do_log)
    trace->logdate() << "Reading trace classes";

  trace->removeAllClassConfig();
  bool is_info_activated = trace->isInfoActivated();
  Trace::eDebugLevel debug_level = Trace::Medium;
  // Désactive les infos de debug si on ne sort pas sur un fichier
  // et que les infos sont désactivées. Cela évite que tous les PE écrivent
  // les infos de débug dans la sortie standard.
  if (!is_info_activated && !m_is_parallel_output)
    debug_level = Trace::None;
  TraceClassConfig medium_cc(is_info_activated,true,debug_level);
  trace->setClassConfig("*",medium_cc);
  if (root_element.null()){
    if (do_log)
      trace->log() << "No user configuration";
    return;
  }

  XmlNodeList children = root_element.child("traces").children("trace-class");
  String ustr_name("name");
  String ustr_info("info");
  String ustr_pinfo("parallel-info");
  String ustr_debug("debug");
  String ustr_true("true");
  String ustr_none("none");
  String ustr_lowest("lowest");
  String ustr_low("low");
  String ustr_medium("medium");
  String ustr_high("high");
  String ustr_highest("highest");
  String ustr_star("*");
  String ustr_print_class_name("print-class-name");
  String ustr_print_elapsed_time("print-elapsed-time");

  for( auto xnode : children.range() ){
    String module_name = xnode.attrValue(ustr_name);
    String activate_str = xnode.attrValue(ustr_info);
    String parallel_activate_str = xnode.attrValue(ustr_pinfo);
    String dbg_lvl_str = xnode.attrValue(ustr_debug);
    String print_class_name_str = xnode.attrValue(ustr_print_class_name);
    String print_elapsed_time_str = xnode.attrValue(ustr_print_elapsed_time);
    if (module_name.null())
      continue;
    TraceClassConfig def_config = trace->classConfig("*");
    bool is_activate = def_config.isActivated();
    builtInGetValue(is_activate,activate_str);
    bool is_parallel_activate = is_activate;
    // Si \a disable_info vaut true, désactive les messages d'info sauf si
    // \e parallel-info vaut \a true
    if (!is_info_activated){
      is_activate = false;
      if (parallel_activate_str==ustr_true)
        is_activate = true;
    }

    Trace::eDebugLevel dbg_lvl = def_config.debugLevel();
    {
      if (dbg_lvl_str==ustr_none)
        dbg_lvl = Trace::None;
      if (dbg_lvl_str==ustr_lowest)
        dbg_lvl = Trace::Lowest;
      if (dbg_lvl_str==ustr_low)
        dbg_lvl = Trace::Low;
      if (dbg_lvl_str==ustr_medium)
        dbg_lvl = Trace::Medium;
      if (dbg_lvl_str==ustr_high)
        dbg_lvl = Trace::High;
      if (dbg_lvl_str==ustr_highest)
        dbg_lvl = Trace::Highest;
    }

    bool is_print_class_name = true;
    builtInGetValue(is_print_class_name,print_class_name_str);
    bool is_print_elapsed_time = false;
    builtInGetValue(is_print_elapsed_time,print_elapsed_time_str);
    int flags = Trace::PF_Default;
    if (!is_print_class_name)
      flags |= Trace::PF_NoClassName;
    if (is_print_elapsed_time)
      flags |= Trace::PF_ElapsedTime;

    TraceClassConfig mc (is_activate,is_parallel_activate,dbg_lvl,flags);
    if (do_log)
      trace->log() << "Config " << mc.isActivated() << ' '
                   << mc.isParallelActivated() << ' ' << module_name;
    trace->setClassConfig(module_name,mc);
    if (do_log)
      trace->log() << "Config module class"
                   << " name=" << module_name
                   << " activated=" << is_activate
                   << " dbglvl=" << (int)dbg_lvl;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
