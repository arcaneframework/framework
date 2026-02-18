// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneSimpleExecutor.cc                                     (C) 2000-2026 */
/*                                                                           */
/* Classe permettant d'exécuter du code directement via Arcane.              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/ArcaneSimpleExecutor.h"

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/TraceClassConfig.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ApplicationInfo.h"

#include "arcane/Parallel.h"

#include "arcane/impl/ArcaneMain.h"
#include "arcane/impl/MainFactory.h"
#include "arcane/impl/internal/ArcaneMainExecInfo.h"

#include "arcane/SubDomainBuildInfo.h"
#include "arcane/ISession.h"
#include "arcane/ICodeService.h"
#include "arcane/IApplication.h"
#include "arcane/IParallelSuperMng.h"
#include "arcane/IParallelMng.h"
#include "arcane/ITimeStats.h"
#include "arcane/IIOMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
extern "C++" ARCANE_IMPL_EXPORT Ref<ICodeService>
createArcaneCodeService(IApplication* app);

class ArcaneSimpleExecutorMainFactory
: public MainFactory
{
 public:
  // NOTE: Il faut implémenter cette méthode pour hériter de MainFactory
  // mais elle ne sera pas utilisée dans le cas d'une exécution directe
  // (c'est toujours une instance de ArcaneMain() qui sera créée)
  IArcaneMain* createArcaneMain(const ApplicationInfo& app_info) override
  { 
    return new ArcaneMain(app_info,this);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ArcaneSimpleExecutor::Impl
{
 public:
  Impl()
  {
    ArcaneMain::arcaneInitialize();
  }
  ~Impl() noexcept(false)
  {
    for (ITimeStats* ts : m_time_stats_list){
      ts->endGatherStats();
      delete ts;
    }
    if (m_arcane_main){
      m_arcane_main->finalize();
      delete m_arcane_main;
    }
    delete m_main_factory;
    ArcaneMain::arcaneFinalize();
  }
 public:
  ArcaneMainExecInfo* m_arcane_main = nullptr;
  ArcaneSimpleExecutorMainFactory* m_main_factory = nullptr;
  ApplicationBuildInfo m_application_build_info;
  bool m_has_minimal_verbosity_level = false;
  bool m_has_output_level = false;
  UniqueArray<ITimeStats*> m_time_stats_list;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneSimpleExecutor::
ArcaneSimpleExecutor()
: m_p(new Impl())
{
  const ApplicationInfo& app_info = ArcaneMain::defaultApplicationInfo();
  const CommandLineArguments& cmd_line_args = app_info.commandLineArguments();
  auto& build_info = m_p->m_application_build_info;
  build_info.parseArguments(cmd_line_args);
  // Par défaut il n'y a pas de fichiers de configuration dans le cas
  // d'une exécution directe.
  build_info.setConfigFileName(String());

  // Par défaut, limite le niveau de verbosité de l'initialisation
  // Cela permet d'éviter d'afficher les infos sur les versions
  _setDefaultVerbosityLevel(1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneSimpleExecutor::
~ArcaneSimpleExecutor() noexcept(false)
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneSimpleExecutor::
_checkInit()
{
  if (!m_p->m_arcane_main)
    ARCANE_FATAL("This instance is not initialized. Call initialize() method");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Positionne le niveau des traces à \a level si elles ne sont
 * pas positionnées.
 */
void ArcaneSimpleExecutor::
_setDefaultVerbosityLevel(Integer level)
{
  auto& build_info = m_p->m_application_build_info;

  m_p->m_has_minimal_verbosity_level = (build_info.minimalVerbosityLevel()!=Trace::UNSPECIFIED_VERBOSITY_LEVEL);
  m_p->m_has_output_level = (build_info.outputLevel()!=Trace::UNSPECIFIED_VERBOSITY_LEVEL);

  if (!m_p->m_has_minimal_verbosity_level)
    build_info.setMinimalVerbosityLevel(level);
  if (!m_p->m_has_output_level)
    build_info.setOutputLevel(level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int ArcaneSimpleExecutor::
initialize()
{
  if (m_p->m_arcane_main)
    ARCANE_FATAL("This instance is already initialized");

  const ApplicationInfo& app_info = ArcaneMain::defaultApplicationInfo();
  auto factory = new ArcaneSimpleExecutorMainFactory();
  m_p->m_main_factory = factory;
  m_p->m_arcane_main = new ArcaneMainExecInfo(app_info,m_p->m_application_build_info,factory);

  int r = m_p->m_arcane_main->initialize();
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int ArcaneSimpleExecutor::
runCode(IFunctor* f)
{
  if (!m_p->m_arcane_main)
    ARCANE_FATAL("This instance is not yet initialized");
  if (!f)
    return 0;
  IArcaneMain* am = m_p->m_arcane_main->arcaneMainClass();
  bool clean_abort = false;
  bool is_print = true;
  return ArcaneMain::callFunctorWithCatchedException(f,am,&clean_abort,is_print);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ApplicationBuildInfo& ArcaneSimpleExecutor::
applicationBuildInfo()
{
  return m_p->m_application_build_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const ApplicationBuildInfo& ArcaneSimpleExecutor::
applicationBuildInfo() const
{
  return m_p->m_application_build_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISubDomain* ArcaneSimpleExecutor::
createSubDomain(const String& case_file_name)
{
  _checkInit();
  IApplication* app = m_p->m_arcane_main->arcaneMainClass()->application();
  ITraceMng* tr = app->traceMng();
  if (!m_p->m_has_minimal_verbosity_level)
    tr->setVerbosityLevel(Trace::DEFAULT_VERBOSITY_LEVEL);
  if (!m_p->m_has_output_level)
    tr->setStandardOutputVerbosityLevel(Trace::DEFAULT_VERBOSITY_LEVEL);
  IMainFactory* main_factory = app->mainFactory();
  // TODO: utiliser le service de code spécifié dans ApplicationInfo.
  Ref<ICodeService> code_service = createArcaneCodeService(app);
  ISession* session(code_service->createSession());
  IParallelSuperMng* psm = app->parallelSuperMng();
  Ref<IParallelMng> world_pm = psm->internalCreateWorldParallelMng(0);

  SubDomainBuildInfo sdbi(world_pm,0);
  UniqueArray<Byte> case_bytes;
  bool has_case_file = !case_file_name.empty();
  if (has_case_file){
    bool is_bad = app->ioMng()->collectiveRead(case_file_name,case_bytes);
    if (is_bad)
      ARCANE_FATAL("Can not read case file '{0}'",case_file_name);
    sdbi.setCaseFileName(case_file_name);
    sdbi.setCaseBytes(case_bytes);
    tr->info() << "Create sub domain with case file '" << case_file_name << "'";
  }
  else{
    sdbi.setCaseFileName(String());
    sdbi.setCaseBytes(ByteConstArrayView());
  }
  // Le service de statistiques doit être détruit explicitement
  ITimeStats* time_stat = main_factory->createTimeStats(world_pm->timerMng(),tr,"Stats");
  time_stat->beginGatherStats();
  m_p->m_time_stats_list.add(time_stat);
  world_pm->setTimeStats(time_stat);

  ISubDomain* sub_domain(session->createSubDomain(sdbi));
  if (has_case_file){
    code_service->initCase(sub_domain,false);
  }
  return sub_domain;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
