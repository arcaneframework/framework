// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CodeService.cc                                              (C) 2000-2024 */
/*                                                                           */
/* Service du code.                                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/CodeService.h"

#include "arcane/utils/List.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/Exception.h"
#include "arcane/utils/String.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/IProfilingService.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/JSONWriter.h"

#include "arcane/core/IApplication.h"
#include "arcane/core/ISession.h"
#include "arcane/core/IServiceLoader.h"
#include "arcane/core/IMainFactory.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/ServiceBuildInfo.h"
#include "arcane/core/ICheckpointMng.h"
#include "arcane/core/ICaseMng.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/CheckpointInfo.h"
#include "arcane/core/internal/IVariableMngInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CodeServicePrivate
{
 public:
  CodeServicePrivate(IApplication* application,IServiceInfo* si);
 public:
  IServiceInfo* m_service_info;
  IApplication* m_application;
  StringList m_extensions;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CodeServicePrivate::
CodeServicePrivate(IApplication* application,IServiceInfo* si)
: m_service_info(si)
, m_application(application)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CodeService::
CodeService(const ServiceBuildInfo& sbi)
: m_p(new CodeServicePrivate(sbi.application(),sbi.serviceInfo()))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CodeService::
~CodeService()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool CodeService::
allowExecution() const
{
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StringCollection CodeService::
validExtensions() const
{
  return m_p->m_extensions;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISubDomain* CodeService::
createAndLoadCase(ISession* session,const SubDomainBuildInfo& sdbi)
{
  ITraceMng* trace = session->traceMng();

  ISubDomain* sub_domain = session->createSubDomain(sdbi);

  // Permet à la classe dérivée de faire ses modifs
  _preInitializeSubDomain(sub_domain);

  try{
    sub_domain->readCaseMeshes();
  }
  catch(const Exception& ex)
  {
    trace->error() << ex;
    throw;
  }
  catch(...){
    trace->error() << "Unknown exception thrown";
    throw;
  }
  
  return sub_domain;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CodeService::
initCase(ISubDomain* sub_domain,bool is_continue)
{
  IProfilingService* ps = nullptr;
  bool is_profile_init = false;
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_PROFILE_INIT", true))
    is_profile_init = (v.value()!=0);
  if (is_profile_init)
    ps = platform::getProfilingService();
  {
    ProfilingSentryWithInitialize ps_sentry(ps);
    ps_sentry.setPrintAtEnd(true);
    /*
     * Les différentes phases de l'initialisation sont :
     * - allocation des structures gérant les maillages.
     * - relecture des protections (si on est en reprise).
     * - lecture (phase1, donc sans les éléments de maillage) du jeu de données.
     * - appel des points d'entrée de type 'build'.
     * - lecture des maillages (si init) ou relecture (si reprise).
     */
    if (is_continue)
      sub_domain->setIsContinue();
    sub_domain->allocateMeshes();
    if (is_continue){
      ICheckpointMng* cm = sub_domain->checkpointMng();
      CheckpointInfo ci = cm->readDefaultCheckpointInfo();
      ci.setIsRestart(true);
      cm->readCheckpoint(ci);
    }
    ICaseMng* case_mng = sub_domain->caseMng();
    // Lecture du jeu de donnée (phase1)
    case_mng->readOptions(true);
    ITimeLoopMng* loop_mng = sub_domain->timeLoopMng();
    loop_mng->execBuildEntryPoints();
    sub_domain->readOrReloadMeshes();

    IVariableMng* vm = sub_domain->variableMng();
    vm->_internalApi()->initializeVariables(is_continue);
    if (!is_continue)
      sub_domain->initializeMeshVariablesFromCaseFile();
    // Lecture du jeu de donnée (phase2)
    case_mng->readOptions(false);
    case_mng->printOptions();
    // Effectue le partitionnement initial ou de reprise
    sub_domain->doInitMeshPartition();
    loop_mng->execInitEntryPoints(is_continue);
    sub_domain->setIsInitialized();
  }

  if (is_profile_init && ps){
    // Génère un fichier JSON avec les informations de profiling
    // de l'initialisation.
    JSONWriter json_writer(JSONWriter::FormatFlags::None);
    json_writer.beginObject();
    ps->dumpJSON(json_writer);
    json_writer.endObject();
    String file_name = String("profiling_init-") + platform::getProcessId() + String(".json");
    std::ofstream ofile(file_name.localstr());
    ofile << json_writer.getBuffer();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IServiceInfo* CodeService::
serviceInfo() const
{
  return m_p->m_service_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IBase* CodeService::
serviceParent() const
{
  return m_p->m_application;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CodeService::
_addExtension(const String& extension)
{
  m_p->m_extensions.add(extension);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IApplication* CodeService::
_application() const
{
  return m_p->m_application;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
