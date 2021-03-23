// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TimeLoopReader.cc                                           (C) 2000-2020 */
/*                                                                           */
/* Chargement d'une boucle en temps.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Iterator.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/List.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/impl/TimeLoopReader.h"

#include "arcane/IApplication.h"
#include "arcane/IXmlDocumentHolder.h"
#include "arcane/IIOMng.h"
#include "arcane/ISubDomain.h"
#include "arcane/XmlNode.h"
#include "arcane/XmlNodeList.h"
#include "arcane/XmlNodeIterator.h"
#include "arcane/ICaseDocument.h"
#include "arcane/ArcaneException.h"
#include "arcane/ITimeLoopMng.h"
#include "arcane/ITimeLoop.h"
#include "arcane/IMainFactory.h"
#include "arcane/SequentialSection.h"
#include "arcane/TimeLoopEntryPointInfo.h"
#include "arcane/TimeLoopSingletonServiceInfo.h"
#include "arcane/impl/ConfigurationReader.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeLoopReader::
TimeLoopReader(IApplication* sm)
: TraceAccessor(sm->traceMng())
, m_application(sm)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeLoopReader::
~TimeLoopReader()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopReader::
setUsedTimeLoop(ISubDomain* sub_domain)
{
  ITimeLoopMng* loop_mng = sub_domain->timeLoopMng();

  XmlNode time_loop_elem = sub_domain->caseDocument()->timeloopElement();

  {
    SequentialSection ss(sub_domain);
    try{
      String value;
      if (!time_loop_elem.null())
        value = time_loop_elem.value();
      if (value.null())
        ARCANE_FATAL("No time loop specified");
      if (value.empty())
        value = "ArcaneEmptyLoop";
      // La boucle en temps est spécifiée dans le fichier de config.
      m_time_loop_name = value;
      info() << "Using the time loop <" << m_time_loop_name << ">";
      loop_mng->setUsedTimeLoop(m_time_loop_name);
    }
    catch(const Exception& ex){
      error() << ex << '\n';
      ss.setError(true);
    }
    catch(...){
      ss.setError(true);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopReader::
readTimeLoops()
{
  IMainFactory* factory = m_application->mainFactory();

  ByteConstSpan config_bytes = m_application->configBuffer();
  ScopedPtrT<IXmlDocumentHolder> config_doc(m_application->ioMng()->parseXmlBuffer(config_bytes,String()));
  if (!config_doc.get())
    ARCANE_FATAL("Can not parse code configuration file");
  XmlNode root_elem = config_doc->documentNode().documentElement();

  //  XmlNode root_elem = m_application->configRootElement();
  XmlNode elem = root_elem.child(String("time-loops"));
  XmlNodeList timeloops = elem.children(String("time-loop"));

  StringList required_modules_list;
  StringList optional_modules_list;
  List<TimeLoopEntryPointInfo> entry_points;
  StringList user_classes;
  List<TimeLoopSingletonServiceInfo> singleton_services;

  List<TimeLoopSingletonServiceInfo> global_singleton_services;

  String ustr_name("name");
  String ustr_modules("modules");

  // Liste des services singletons globaux
  XmlNodeList global_singleton_elems = root_elem.children("singleton-services");
  info() << "CHECK GLOBAL SINGLETON SERVICES";
  for( auto i : global_singleton_elems.range() ){
    info() << "CHECK GLOBAL SINGLETON SERVICES 2 " << i.name();

   for( auto j_node : i ){
      if (j_node.name()=="service"){
        bool is_required = (j_node.attrValue("need")=="required");
        info() << "GLOBAL SINGLETON SERVICE name=" << j_node.attrValue(ustr_name) << " is_required?=" << is_required;
        global_singleton_services.add(TimeLoopSingletonServiceInfo(j_node.attrValue(ustr_name),is_required));
      }
    }
  }

  for( auto i : timeloops.range() ){
    optional_modules_list.clear();
    required_modules_list.clear();
    user_classes.clear();
    singleton_services.clone(global_singleton_services);

    String name = i.attrValue(ustr_name);

    if (name.null())
      continue;
    ITimeLoop* time_loop = factory->createTimeLoop(m_application, name);
    XmlNode timeloop_node = i;
    
    for( auto j_node : timeloop_node ){
      String elem_name = j_node.name();
      String elem_value = j_node.value();

      if (elem_name==ustr_modules){
        for(XmlNode::const_iter k (j_node) ; k() ; ++k)
          if (k->name()=="module"){
            if (k->attrValue("need")=="required")
              required_modules_list.add(k->attrValue(ustr_name));
            else
              optional_modules_list.add(k->attrValue(ustr_name));
          }
      }
      else if (elem_name=="singleton-services"){
        for(XmlNode::const_iter k (j_node) ; k() ; ++k)
          if (k->name()=="service"){
            bool is_required = (k->attrValue("need")=="required");
            //info() << "SINGLETON SERVICE name=" << k->attrValue(ustr_name) << " is_required?=" << is_required;
            singleton_services.add(TimeLoopSingletonServiceInfo(k->attrValue(ustr_name),is_required));
          }
      }
      else if (elem_name=="entry-points"){
        entry_points.clear();
        for( XmlNode::const_iter k (j_node) ; k() ; ++k){
          XmlNode k_node = *k;
          if (k_node.name()!="entry-point")
            continue;
          StringList depends;
          entry_points.add(TimeLoopEntryPointInfo(k_node.attrValue(ustr_name),depends));
        }

        String cwhere = j_node.attrValue("where");
        if (cwhere != ITimeLoop::WInit 
            && cwhere != ITimeLoop::WComputeLoop
            && cwhere != ITimeLoop::WRestore
            && cwhere != ITimeLoop::WExit
            && cwhere != ITimeLoop::WBuild
            && cwhere != ITimeLoop::WOnMeshChanged
            && cwhere != ITimeLoop::WOnMeshRefinement)
        {
          OStringStream s;
          s() << "Incorrect value for the attribute \"where\" (time loop ";
          s() << name << "): \"" << cwhere << "\".\n";
          s() << "Available values are: "
              << ITimeLoop::WInit
              << ", " << ITimeLoop::WComputeLoop
              << ", " << ITimeLoop::WRestore
              << ", " << ITimeLoop::WOnMeshChanged
              << ", " << ITimeLoop::WOnMeshRefinement
              << ", " << ITimeLoop::WBuild
              << ", " << ITimeLoop::WExit
              << ".";
          throw InternalErrorException(A_FUNCINFO,s.str());
        }
        time_loop->setEntryPoints(cwhere,entry_points);
      }
      else if (elem_name=="title")
      {
        time_loop->setTitle(elem_value);
      }
      else if (elem_name=="description")
      {
        time_loop->setDescription(elem_value);
      }
      else if (elem_name=="userclass")
      {
        user_classes.add(elem_value);
      }
      else if (elem_name=="configuration"){
        ConfigurationReader cr(traceMng(),time_loop->configuration());
        cr.addValuesFromXmlNode(j_node,ConfigurationReader::P_TimeLoop);
      }
    }

    time_loop->setRequiredModulesName(required_modules_list);
    time_loop->setOptionalModulesName(optional_modules_list);
    time_loop->setUserClasses(user_classes);
    time_loop->setSingletonServices(singleton_services);

    m_time_loops.add(time_loop);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TimeLoopReader::
registerTimeLoops(ISubDomain* sd)
{
  ITimeLoopMng* loop_mng = sd->timeLoopMng();
  for( TimeLoopList::Enumerator i(m_time_loops); ++i; )
    loop_mng->registerTimeLoop(*i);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

