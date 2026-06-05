// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneMainBatch.cc                                          (C) 2000-2026 */
/*                                                                           */
/* Batch execution management.                                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Iostream.h"
#include "arcane/utils/Ptr.h"
#include "arcane/utils/StdHeader.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/List.h"
#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/String.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/IMemoryInfo.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/IFunctor.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/IProcessorAffinityService.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/CStringUtils.h"
#include "arcane/utils/ITraceMngPolicy.h"
#include "arcane/utils/CommandLineArguments.h"
#include "arcane/utils/CriticalSection.h"
#include "arccore/common/internal/ParameterListPropertyReader.h"
#include "arccore/common/internal/Property.h"

#include "arcane/impl/ArcaneMain.h"
#include "arcane/impl/ParallelReplication.h"

#include "arcane/core/IIOMng.h"
#include "arcane/core/ICodeService.h"
#include "arcane/core/ISession.h"
#include "arcane/core/Timer.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IApplication.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/ITimeStats.h"
#include "arcane/core/SequentialSection.h"
#include "arcane/core/IParallelSuperMng.h"
#include "arcane/core/ITimeHistoryMng.h"
#include "arcane/core/IDirectExecution.h"
#include "arcane/core/IDirectSubDomainExecuteFunctor.h"
#include "arcane/core/ICaseMng.h"
#include "arcane/core/ServiceFinder2.h"
#include "arcane/core/SubDomainBuildInfo.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IMainFactory.h"
#include "arcane/core/ApplicationBuildInfo.h"
#include "arcane/core/CaseDatasetSource.h"

#include "arcane/core/ServiceUtils.h"

#include "arcane/core/IVariableMng.h"
#include "arcane/core/VariableCollection.h"
#include "arcane/core/internal/IVariableMngInternal.h"

#include "arcane/impl/ExecutionStatsDumper.h"
#include "arcane/impl/TimeLoopReader.h"

#include "arccore/common/accelerator/internal/RunnerInternal.h"

#include <thread>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Properties associated with ArcaneMain
class ArcaneMainBatchProperties
{
  ARCANE_DECLARE_PROPERTY_CLASS(ArcaneMainBatchProperties);

 public:

  Int32 m_max_iteration = 0;
  bool m_is_continue = false;
  String m_idle_service_name; //!< Service name for unused CPUs
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Batch execution of a code.
 */
class ArcaneMainBatch
: public ArcaneMain
{
 public:

  //! Execution information for a session.
  class SessionExec
  {
    //! Info per subdomain that must be destroyed at the end of execution
    class SubInfo
    {
     public:

      SubInfo()
      : m_sub_domain(nullptr)
      , m_time_stats(nullptr)
      , m_want_print_stats(false)
      {}
      ~SubInfo()
      {
        // ITimeStats must be destroyed first because it uses
        // the TimerMng of IParallelMng.
        delete m_time_stats;
        //delete m_rank_parallel_mng;
        //m_world_parallel_mng.reset();
        // The subdomain is destroyed when the session ends
      }
      SubInfo(const SubInfo&) = delete;
      void operator=(const SubInfo&) = delete;

     public:

      Ref<IParallelMng> m_world_parallel_mng;
      Ref<IParallelMng> m_rank_parallel_mng;
      ISubDomain* m_sub_domain;
      ITimeStats* m_time_stats;
      bool m_want_print_stats;
    };

   public:

    SessionExec(ArcaneMainBatch* arcane_main, ISession* session, Int32 nb_local_rank)
    : m_arcane_main(arcane_main)
    , m_session(session)
    , m_has_sub_domain_threads(m_arcane_main->m_has_sub_domain_threads)
    , m_direct_test_name(m_arcane_main->m_direct_test_name)
    , m_properties(m_arcane_main->m_properties)
    , m_code_service(m_arcane_main->m_code_service)
    , m_sub_infos(nb_local_rank)
    , m_direct_sub_domain_execute_functor(m_arcane_main->_directExecuteFunctor())
    {
      const CaseDatasetSource& dataset_source = m_arcane_main->applicationBuildInfo().caseDatasetSource();
      m_case_file = dataset_source.fileName();
      m_case_bytes = dataset_source.content();
      // The sub_infos for each thread are created in executeRank()
      m_sub_infos.fill(nullptr);
    }
    ~SessionExec()
    {
      for (Integer i = 0, n = m_sub_infos.size(); i < n; ++i)
        delete m_sub_infos[i];
    }

   public:

    // Collective over the process threads
    void executeRank(Int32 local_rank);

   private:

    IApplication* _application() { return m_arcane_main->application(); }

   private:

    ArcaneMainBatch* m_arcane_main;
    ISession* m_session;
    bool m_has_sub_domain_threads; //!< indicates if threads are used to manage subdomains
    String m_direct_test_name;
    String m_case_file; //!< Name of the file containing the case.
    UniqueArray<std::byte> m_case_bytes; //!< Content of the case dataset as an XML document.
    const ArcaneMainBatchProperties m_properties; //!< Execution properties.
    Ref<ICodeService> m_code_service; //!< Code service.
    UniqueArray<SubInfo*> m_sub_infos;
    IDirectSubDomainExecuteFunctor* m_direct_sub_domain_execute_functor;

   private:

    void _execDirectTest(IParallelMng* pm, const String& test_name, bool is_collective);
    void _printStats(ISubDomain* sd, ITraceMng* trace, ITimeStats* time_stat);
    void _createAndRunSubDomain(SubInfo* sub_info, Ref<IParallelMng> pm, Ref<IParallelMng> all_replica_pm, Int32 local_rank);
  };

  class ExecFunctor
  : public IFunctor
  {
   public:

    ExecFunctor(SessionExec* session_exec, Int32 local_rank)
    : m_session_exec(session_exec)
    , m_local_rank(local_rank)
    {
    }

   public:

    void executeFunctor() override
    {
      m_session_exec->executeRank(m_local_rank);
    }

   private:

    SessionExec* m_session_exec;
    Int32 m_local_rank;
  };

 public:

  ArcaneMainBatch(const ApplicationInfo&, IMainFactory*);
  ~ArcaneMainBatch() override;

  void build() override;
  void initialize() override;
  int execute() override;
  void doAbort() override;
  bool parseArgs(StringList args) override;
  void finalize() override;

 private:

  ISession* m_session = nullptr; //! Session
  ArcaneMainBatchProperties m_properties;
  bool m_init_only; //!< \a true if only initialization is performed.
  bool m_check_case_only; //!< \a true if only dataset verification is performed.
  bool m_has_sub_domain_threads; //!< indicates if threads are used to manage subdomains
  String m_case_name; //!< Case name
  String m_direct_exec_name;
  String m_direct_test_name;
  Ref<ICodeService> m_code_service; //!< Code service.
  SessionExec* m_session_exec = nullptr;

 private:

  bool _sequentialParseArgs(StringList args);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_IMPL_EXPORT IArcaneMain*
createArcaneMainBatch(const ApplicationInfo& app_info, IMainFactory* main_factory)
{
  return new ArcaneMainBatch(app_info, main_factory);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneMainBatch::
ArcaneMainBatch(const ApplicationInfo& exe_info, IMainFactory* main_factory)
: ArcaneMain(exe_info, main_factory)
, m_init_only(false)
, m_check_case_only(false)
, m_has_sub_domain_threads(false)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMainBatch::
build()
{
  ArcaneMain::build();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMainBatch::
initialize()
{
  ArcaneMain::initialize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneMainBatch::
~ArcaneMainBatch()
{
  // Normally finalize() should have been called to release the
  // various objects (m_session, m_code_service, ...).
  // If this is not the case, it is probably due to an exception and in
  // this case we do nothing to avoid destroying objects whose internal
  // state we do not know well.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ArcaneMainBatch::
parseArgs(StringList args)
{
  if (ArcaneMain::parseArgs(args))
    return true;

  bool r = _sequentialParseArgs(args);
  return r;
}

/*****************************************************************************
 * The variables ARCANE_NB_SUB_DOMAIN & ARCANE_IDLE_SERVICE are less prioritized
 * than the arguments passed to the executable.
 *****************************************************************************/
bool ArcaneMainBatch::
_sequentialParseArgs(StringList args)
{
  ITraceMng* trace = _application()->traceMng();

  String us_arcane_opt("-arcane_opt");
  String us_init_only("init_only");
  String us_check_case_only("check_case_only");
  String us_continue("continue");
  String us_max_iteration("max_iteration");
  String us_casename("casename");
  String us_direct_exec("direct_exec");
  String us_direct_test("direct_test");
  String us_direct_mesh("direct_mesh");
  String us_tool_arg("tool_arg");
  String us_direct_exec_mesh_arg("direct_exec_mesh_arg");
  String us_nb_sub_domain("nb_sub_domain");
  String us_nb_replication("nb_replication");
  String us_idle_service("idle_service");

  // Fills 'm_properties' based on command line parameters
  // TODO: This mechanism has been available since January 2021. Eventually, it will be
  // obsolete and the possibility of specifying options via '-arcane_opt' will be removed.
  properties::readFromParameterList(applicationInfo().commandLineArguments().parameters(), m_properties);

  CaseDatasetSource& dataset_source = _applicationBuildInfo().caseDatasetSource();
  // Indicates if we have a dataset.
  bool has_case_dataset_content = !(dataset_source.fileName().empty() && dataset_source.content().empty());
  Integer nb_arg = args.count();
  if (nb_arg < 2 && !has_case_dataset_content) {
    trace->info() << "Usage: program input_data ; for more information: program -arcane_opt help";
    trace->pfatal() << "No input data specified.";
  }

  StringList unknown_args;
  StringBuilder tool_args_xml;
  StringBuilder direct_exec_mesh_args_xml;
  String tool_mesh;

  String nb_sub_domain_str;
  String nb_replication_str;
  String idle_service_name = platform::getEnvironmentVariable("ARCANE_IDLE_SERVICE");
  if (!idle_service_name.null())
    m_properties.m_idle_service_name = idle_service_name;

  for (Integer i = 1, s = nb_arg - 1; i < s; ++i) {
    // cerr << "** ARGS ARGS " << i << ' ' << args[i] << '\n';
    if (args[i] != us_arcane_opt) {
      unknown_args.add(args[i]);
      continue;
    }
    bool is_valid_opt = false;
    ++i;
    String str;
    if (i < s)
      str = args[i];
    if (str == us_init_only) {
      m_init_only = true;
      is_valid_opt = true;
    }
    else if (str == us_check_case_only) {
      m_check_case_only = true;
      is_valid_opt = true;
    }
    else if (str == us_continue) {
      m_properties.m_is_continue = true;
      is_valid_opt = true;
    }
    else if (str == us_max_iteration) {
      ++i;
      if (i < s) {
        m_properties.m_max_iteration = CStringUtils::toInteger(args[i].localstr());
        //cerr << "** MAX ITER " << m_max_iteration << '\n';
        is_valid_opt = true;
      }
      else
        trace->pfatal() << "Option 'max_iteration' must specify the number of iterations";
    }
    // Case name.
    else if (str == us_casename) {
      ++i;
      if (i < s) {
        m_case_name = args[i];
        is_valid_opt = true;
      }
    }
    else if (str == us_direct_exec) {
      ++i;
      if (i < s) {
        m_direct_exec_name = args[i];
        //trace->info()<<"[ArcaneMainBatch] m_direct_exec_name="<<args[i];
        is_valid_opt = true;
      }
    }
    else if (str == us_direct_test) {
      ++i;
      if (i < s) {
        m_direct_test_name = args[i];
        //trace->info()<<"[ArcaneMainBatch] m_direct_test_name="<<args[i];
        is_valid_opt = true;
      }
    }
    else if (str == us_tool_arg || str == us_direct_exec_mesh_arg) {
      ++i;
      String arg;
      String value;
      if (i < s) {
        arg = args[i];
      }
      ++i;
      if (i < s) {
        value = args[i];
        is_valid_opt = true;
        String to_add = String::format("<{0}>{1}</{2}>\n", arg, value, arg);
        if (str == us_tool_arg)
          tool_args_xml += to_add;
        else if (str == us_direct_exec_mesh_arg)
          direct_exec_mesh_args_xml += to_add;
      }
    }
    else if (str == us_nb_sub_domain) {
      ++i;
      if (i < s) {
        nb_sub_domain_str = args[i];
        //trace->info()<<"[ArcaneMainBatch] nb_sub_domain_str="<<args[i];
        is_valid_opt = true;
      }
    }
    else if (str == us_nb_replication) {
      ++i;
      if (i < s) {
        nb_replication_str = args[i];
        //trace->info()<<"[ArcaneMainBatch] nb_sub_domain_str="<<args[i];
        is_valid_opt = true;
      }
    }
    else if (str == us_idle_service) {
      ++i;
      if (i < s) {
        m_properties.m_idle_service_name = args[i];
        //trace->info()<<"[ArcaneMainBatch] m_idle_service_name="<<args[i];
        is_valid_opt = true;
      }
    }
    if (!is_valid_opt) {
      trace->pfatal() << "Unknown Arcane option <" << str << ">\n";
    }
  }

  bool use_direct_test = (!m_direct_test_name.null());
  bool use_direct_exec = (!m_direct_exec_name.null());

  if (use_direct_test) {
  }
  else if (use_direct_exec) {
    // In this case, the last argument of the command line is
    // the mesh name.
    tool_mesh = args[nb_arg - 1];
    dataset_source.setFileName("Dummy.arc");
  }
  else {
    // The case name is contained in the last argument of the command line.
    // We take this argument unless a filename has already been set before
    // initializing Arcane.
    if (dataset_source.fileName().empty() && dataset_source.content().empty())
      dataset_source.setFileName(args[nb_arg - 1]);
  }

  if (!nb_sub_domain_str.null()) {
    Int32 nb_sub_domain = 0;
    bool is_bad = builtInGetValue(nb_sub_domain, nb_sub_domain_str);
    if (is_bad || nb_sub_domain <= 0) {
      trace->pfatal() << "Invalid number of subdomains : " << nb_sub_domain;
    }
    trace->info() << "Use '" << nb_sub_domain << "' subdomains";
    _applicationBuildInfo().setNbProcessusSubDomain(nb_sub_domain);
  }

  if (!nb_replication_str.null()) {
    Int32 nb_replication = 0;
    bool is_bad = builtInGetValue(nb_replication, nb_replication_str);
    if (is_bad || nb_replication < 0) {
      trace->pfatal() << "Invalid number of replication : " << nb_replication;
    }
    trace->info() << "Use replication of subdomains nb_replication=" << nb_replication;
    _applicationBuildInfo().setNbReplicationSubDomain(nb_replication);
  }

  if (_applicationBuildInfo().nbReplicationSubDomain() != 0 && _applicationBuildInfo().nbProcessusSubDomain() != 0)
    trace->pfatal() << "The subdomains number of replication and restriction options are incompatible.";

  if (!use_direct_test) {
    String case_file = dataset_source.fileName();
    //trace->info()<<"[ArcaneMainBatch] !use_direct_test, getCodeService";
    m_code_service = _application()->getCodeService(case_file);

    if (!m_code_service) {
      trace->info() << "The file `" << case_file << "' is not a known file type.";
      case_file = args[nb_arg - 2];

      m_code_service = _application()->getCodeService(case_file);
      if (!m_code_service) {
        trace->pfatal() << "File extension not valid.";
      }
    }
  }

  if (use_direct_exec) {
    //trace->info()<<"[ArcaneMainBatch] use_direct_test!";
    // Analyzes the arguments corresponding to direct execution options
    // and builds an XML file from them.
    StringBuilder s;
    s += "<?xml version=\"1.0\"?>\n";
    s += "<case codename=\"ArcaneDriver\" xml:lang=\"en\" codeversion=\"1.0\">";
    s += " <arcane>\n";
    s += "  <title>DirectExec</title>\n";
    s += "  <description>DirectExec</description>\n";
    s += "  <timeloop>ArcaneDirectExecutionLoop</timeloop>\n";
    s += " </arcane>\n";
    s += " <meshes>\n";
    s += "   <mesh>\n";
    s += String::format("  <filename>{0}</filename>\n", tool_mesh);
    s += direct_exec_mesh_args_xml;
    s += "   </mesh>\n";
    s += " </meshes>\n";
    s += " <arcane-direct-execution>\n";
    s += String::format("  <tool name='{0}'>\n", m_direct_exec_name);
    s += tool_args_xml;
    s += "  </tool>\n";
    s += " </arcane-direct-execution>\n";
    s += "</case>\n";
    dataset_source.setFileName("(None)");
    String buf = s;
    dataset_source.setContent(buf.utf8());
    trace->info() << "Direct exec xml file=" << s;
  }

  if (m_code_service.get()) {
    bool is_bad = m_code_service->parseArgs(unknown_args);
    if (is_bad)
      return true;
  }

  if (!unknown_args.empty()) {
    trace->info() << "Unknown command line option: " << unknown_args[0];
  }

  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  struct LaunchThreadInfo
  {
    ArcaneMainBatch* arcane_main;
    ArcaneMainBatch::SessionExec* session_exec;
    IApplication* application;
    Int32 thread_index;
  };
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * This function is called when a thread is created.
 */
void _ThreadWrapper(LaunchThreadInfo* lti)
{
  ArcaneMainBatch* amb = lti->arcane_main;
  IApplication* main_app = lti->application;
  ArcaneMainBatch::ExecFunctor functor(lti->session_exec, lti->thread_index);
  bool clean_abort = false;
  bool is_master = lti->thread_index == 0;
  int r = ArcaneMain::callFunctorWithCatchedException(&functor, amb, &clean_abort, is_master);
  if (r != 0 && !clean_abort) {
    // The thread has finished but since it is the only one that crashed,
    // it is possible that the others are blocked.
    // In this case, we perform an abort to prevent blocking
    // TODO: try to kill the other threads correctly.
    if (main_app) {
      IParallelSuperMng* psm = main_app->parallelSuperMng();
      psm->tryAbort();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int ArcaneMainBatch::
execute()
{
  ITraceMng* trace = _application()->traceMng();

  if (m_code_service.get())
    m_session = m_code_service->createSession();

  IParallelSuperMng* psm = _application()->parallelSuperMng();
  Int32 nb_total_rank = psm->commSize();
  const Integer nb_wanted_sub_domain = applicationBuildInfo().nbReplicationSubDomain();
  CaseDatasetSource& dataset_source = _applicationBuildInfo().caseDatasetSource();
  if (nb_wanted_sub_domain > nb_total_rank)
    ARCANE_THROW(ArgumentException, "Number of subdomain '{0}' > number of allocated cores '{1}",
                 nb_wanted_sub_domain, nb_total_rank);

  Integer nb_local_rank = psm->nbLocalSubDomain();
  trace->info() << "NB_LOCAL_RANK=" << nb_local_rank;
  if (nb_local_rank >= 1)
    m_has_sub_domain_threads = true;
  int return_value = 0;

  // Reading dataset data.
  if (dataset_source.content().empty() && m_direct_test_name.null()) {
    String case_file = dataset_source.fileName();
    trace->info() << "Reading input data '" << case_file << "'";
    IIOMng* io_mng = _application()->ioMng();
    UniqueArray<Byte> case_bytes;
    bool is_bad = io_mng->collectiveRead(case_file, case_bytes);
    if (is_bad)
      ARCANE_THROW(ParallelFatalErrorException, "Cannot read input data file '{0}'", case_file);
    dataset_source.setContent(case_bytes);
  }

  m_session_exec = new SessionExec(this, m_session, nb_local_rank);

  UniqueArray<LaunchThreadInfo> thinfo(nb_local_rank);
  for (Integer i = 0; i < nb_local_rank; ++i) {
    thinfo[i].arcane_main = this;
    thinfo[i].session_exec = m_session_exec;
    thinfo[i].application = _application();
    thinfo[i].thread_index = i;
  }

  if (nb_local_rank > 1) {
    UniqueArray<std::thread*> gths(nb_local_rank);
    for (Integer i = 0; i < nb_local_rank; ++i) {
      gths[i] = new std::thread(_ThreadWrapper, &thinfo[i]);
    }
    for (Integer i = 0; i < nb_local_rank; ++i) {
      gths[i]->join();
      delete gths[i];
    }
  }
  else {
    m_has_sub_domain_threads = false;
    m_session_exec->executeRank(0);
  }

  // TODO: remove because it is useless as it always equals 0.
  return return_value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * In mode with one subdomain per thread, this function is called
 * by each thread (potentially concurrently) for its subdomain.
 * \a local_rank indicates the local rank of the thread, which is between 0 and \a nb_local_sub_domain (as defined in execute()).
 */
void ArcaneMainBatch::SessionExec::
executeRank(Int32 local_rank)
{
  // ATTENTION:
  // This function must be reentrant...

  auto sub_info = new SubInfo();
  m_sub_infos[local_rank] = sub_info;

  IProcessorAffinityService* pas = platform::getProcessorAffinityService();
  if (pas && m_has_sub_domain_threads) {
    // CPU binding should only occur if requested and only if
    // the total number of threads (across all processes)
    // does not exceed the number of cores on the machine.
    if (!platform::getEnvironmentVariable("ARCANE_BIND_THREADS").null()) {
      ITraceMng* tm = _application()->traceMng();
      tm->info() << "Binding threads";
      pas->bindThread(local_rank);
    }
  }

  // Creation of the parallelism manager for all allocated ranks.
  IParallelSuperMng* psm = _application()->parallelSuperMng();
  Ref<IParallelMng> world_pm = psm->internalCreateWorldParallelMng(local_rank);
  sub_info->m_world_parallel_mng = world_pm;

  if (!m_direct_test_name.null()) {
    _execDirectTest(world_pm.get(), m_direct_test_name, true);
    return;
  }

  // Checks if we want to run the calculation on a subset
  // of the allocated resources. For now, it is only possible
  // to choose a number of subdomains. If so, only
  // ranks from 0 up to the desired number of subdomains minus 1 are
  // used. Higher ranks do not have subdomains
  // and instead use a service that implements IDirectExecution

  // Creation of the execution statistics manager.
  ITraceMng* trace = world_pm->traceMng();
  String stat_name = "Rank";
  stat_name = stat_name + world_pm->commRank();
  ITimeStats* time_stat = _application()->mainFactory()->createTimeStats(world_pm->timerMng(), trace, stat_name);
  sub_info->m_time_stats = time_stat;
  time_stat->beginGatherStats();
  world_pm->setTimeStats(time_stat);

  Ref<IParallelMng> pm = world_pm;
  Ref<IParallelMng> all_replica_pm = pm;

  const Integer nb_wanted_sub_domain = _application()->applicationBuildInfo().nbProcessusSubDomain();
  const Integer nb_wanted_replication = _application()->applicationBuildInfo().nbReplicationSubDomain();
  // We are in parallel and we want fewer subdomains than allocated processes
  if (world_pm->isParallel()) {
    // For now, we cannot mix subdomain replication with
    // a number of subdomains different from the number of allocated processors.
    // TODO: when this is no longer the case, we will need to create an all_replica_pm that
    // contains all subdomains and replicas.

    if (nb_wanted_replication > 1) {
      Int32 comm_size = world_pm->commSize();
      Int32 nb_sub_part = comm_size / nb_wanted_replication;
      trace->info() << "Using sub-domain replication nb_sub_part=" << nb_sub_part;
      if ((comm_size % nb_wanted_replication) != 0)
        ARCANE_FATAL("The number of replication '{0}' must be a common factor of the number of allocated cores '{1}",
                     nb_wanted_replication, comm_size);
      // First, we create a communicator containing the replicas of each subdomain
      // This communicator will therefore contain \a m_nb_wanted_replication objects
      Ref<IParallelMng> replicate_pm;
      trace->info() << "Building replicated parallel mng";
      {
        Int32UniqueArray kept_ranks(nb_wanted_replication);
        for (Integer i_sd = 0; i_sd < nb_sub_part; ++i_sd) {
          for (Int32 i = 0; i < nb_wanted_replication; ++i) {
            kept_ranks[i] = i_sd + (i * nb_sub_part);
            trace->info() << "Rank r=" << kept_ranks[i];
          }
          Ref<IParallelMng> new_pm = world_pm->createSubParallelMngRef(kept_ranks);
          if (new_pm.get()) {
            replicate_pm = new_pm;
            replicate_pm->setTimeStats(time_stat);
            trace->info() << " Building own replicated parallel mng";
          }
          else {
            trace->info() << "!pm";
          }
          trace->flush();
        }
      }
      if (!replicate_pm)
        ARCANE_FATAL("Null replicated parallel mng");

      // Now, we create an IParallelMng that corresponds to the set
      // of ranks of a single replica. This IParallelMng will be assigned to
      // the subdomain that will be created later.
      trace->info() << "Building sub-domain parallel mng";
      {
        Int32UniqueArray kept_ranks(nb_sub_part);
        for (Integer i_repl = 0; i_repl < nb_wanted_replication; ++i_repl) {
          for (Int32 i = 0; i < nb_sub_part; ++i) {
            kept_ranks[i] = i + (i_repl * nb_sub_part);
            trace->info() << "Rank r=" << kept_ranks[i];
          }
          Ref<IParallelMng> new_pm = world_pm->createSubParallelMngRef(kept_ranks);
          if (new_pm.get()) {
            pm = new_pm;
            if (nb_sub_part == 1) {
              // We must take the sequential version to make the calculation
              // appear sequential. This manager will be destroyed at the same time
              // as \a new_pm
              pm = new_pm->sequentialParallelMngRef();
            }
            trace->info() << "pm: setting time_stat & m_rank_parallel_mng for replica rank=" << i_repl;
            trace->flush();
            pm->setTimeStats(time_stat);
            sub_info->m_rank_parallel_mng = new_pm;
            auto pr = new ParallelReplication(i_repl, nb_wanted_replication, replicate_pm);
            pm->setReplication(pr);
          }
          else {
            trace->info() << "!pm";
            trace->flush();
          }
        }
      }
    }
    else if (nb_wanted_sub_domain != 0) {
      const Int32 nb_sub_part = nb_wanted_sub_domain;
      Int32UniqueArray kept_ranks(nb_sub_part);
      for (Int32 i = 0; i < nb_sub_part; ++i)
        kept_ranks[i] = i;
      pm = world_pm->createSubParallelMngRef(kept_ranks);
      if (pm.get()) {
        trace->info() << "pm: setting time_stat & m_rank_parallel_mng";
        trace->flush();
        pm->setTimeStats(time_stat);
        sub_info->m_rank_parallel_mng = pm;
        all_replica_pm = pm;
      }
      else {
        trace->info() << "!pm";
        trace->flush();
      }
    }
  }

  bool print_stats = false;
  ISubDomain* sub_domain = nullptr;

  if (!pm) {
    // If this is a rank that does not own a subdomain.
    // In this case, execute the service given by 'm_idle_service_name'
    // (if specified, otherwise do nothing)
    trace->info() << "The rank doesn't own any subdomain!";
    if (m_properties.m_idle_service_name.empty()) {
      trace->info() << "No idle service specified";
      trace->flush();
    }
    else {
      trace->info() << "execDirectTest: " << m_properties.m_idle_service_name;
      trace->flush();
      _execDirectTest(world_pm.get(), m_properties.m_idle_service_name, false);
      // We exit the directTest() via the broadcast(This is the end), so we must return
      return;
    }
    print_stats = true;
  }
  else {
    _createAndRunSubDomain(sub_info, pm, all_replica_pm, local_rank);
    sub_domain = sub_info->m_sub_domain;
    print_stats = sub_info->m_want_print_stats;
  }

  time_stat->endGatherStats();

  if (print_stats && sub_domain) {
    // Ensures everyone is here before stopping the profiling
    // TODO: Since profiling is local to the process, it would be sufficient
    // a priori to perform the barrier on the local IParallelMngs.
    IParallelMng* pm = sub_domain->parallelMng();
    pm->barrier();
    if (local_rank == 0)
      Accelerator::RunnerInternal::stopAllProfiling();
    pm->barrier();
    _printStats(sub_domain, trace, time_stat);

    // We must destroy the shared memory variables here because their
    // destruction is performed collectively.
    // We cannot destroy all variables because some are
    // used afterward (GlobalIteration, for example).
    // If, one day, we put certain "Global" variables in shared memory,
    // this part will cause problems.
    sub_domain->variableMng()->_internalApi()->removeAllShMemVariables();
  }

  //BaseForm[Hash["This is the end", "CRC32"], 16]
  // We inform the 'other' capabilities that they must leave now!
  world_pm->broadcast(UniqueArray<unsigned long>(1, 0xdfeb699fl).view(), 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMainBatch::SessionExec::
_createAndRunSubDomain(SubInfo* sub_info, Ref<IParallelMng> pm, Ref<IParallelMng> all_replica_pm, Int32 local_rank)
{
  // This is a rank that has a subdomain.
  // It is created and execution begins.
  SubDomainBuildInfo sdbi(pm, local_rank, all_replica_pm);
  sdbi.setCaseFileName(m_case_file);
  sdbi.setCaseContent(m_case_bytes);
  ISubDomain* sub_domain = m_code_service->createAndLoadCase(m_session, sdbi);
  sub_info->m_sub_domain = sub_domain;

  ITraceMng* trace = _application()->traceMng();
  ITraceMng* sd_trace = sub_domain->traceMng();
  ITraceMngPolicy* trace_policy = _application()->getTraceMngPolicy();

  // In case of replication, disable the output curves
  // of the replicas.
  trace->info() << "REPLICATION: rank=" << pm->replication()->replicationRank();

  if (!pm->replication()->isMasterRank()) {
    trace->info() << "Disable output curves for replicates.";
    sub_domain->timeHistoryMng()->setDumpActive(false);
  }

  // TODO:
  // Destroy the subdomain at the end of the function, but this requires
  // modifying ISession to support the deletion
  // of a subdomain (and then destroying ISession).

  IProcessorAffinityService* pas = platform::getProcessorAffinityService();
  if (pas) {
    String cpu_set = pas->cpuSetString();
    trace->info() << " CpuSet=" << cpu_set;
  }

  if (m_arcane_main->m_check_case_only) {
    trace->info() << "Checking the input data";
    // Initializes the time loop modules
    {
      TimeLoopReader stl(_application());
      stl.readTimeLoops();
      stl.registerTimeLoops(sub_domain);
      stl.setUsedTimeLoop(sub_domain);
    }
    ICaseMng* cm = sub_domain->caseMng();
    cm->readOptions(true);
  }
  else {
    Timer init_timer(sub_domain, "InitTimer", Timer::TimerReal);
    Timer loop_timer(sub_domain, "LoopTimer", Timer::TimerReal);

    {
      Timer::Action ts_action(sub_domain, "Init");
      Timer::Sentry ts(&init_timer);

      m_code_service->initCase(sub_domain, m_properties.m_is_continue);
    }

    if (m_properties.m_max_iteration > 0)
      trace->info() << "Option 'max_iteration' activated with " << m_properties.m_max_iteration;

    // Redirects signals.
    // This is also done at initialization but here we might be in another
    // thread and some libraries might have redirected signals
    // during the init
    {
      CriticalSection cs(pm->threadMng());
      ArcaneMain::redirectSignals();
    }
    int ret_compute_loop = 0;

    IDirectExecution* direct_exec = sub_domain->directExecution();
    if (direct_exec && direct_exec->isActive()) {
      trace->info() << "Direct execution activated";
      direct_exec->execute();
    }
    else if (m_arcane_main->m_init_only) {
      trace->info() << "Option 'init_only' activated";
      sub_info->m_want_print_stats = true;
    }
    else {
      sub_info->m_want_print_stats = true;
      Timer::Action ts_action(sub_domain, "Loop");
      Timer::Sentry ts(&loop_timer);
      // During the calculation loop, do not force the display of traces at a given level
      // (which is done during application initialization.
      trace_policy->setDefaultVerboseLevel(sd_trace, Trace::UNSPECIFIED_VERBOSITY_LEVEL);
      if (m_direct_sub_domain_execute_functor) {
        m_direct_sub_domain_execute_functor->setSubDomain(sub_domain);
        m_direct_sub_domain_execute_functor->execute();
        sub_domain->parallelMng()->barrier();
      }
      else {
        ret_compute_loop = sub_domain->timeLoopMng()->doComputeLoop(m_properties.m_max_iteration);
        if (ret_compute_loop < 0)
          //TODO: DO NOT FILL THIS FUNCTION DIRECTLY BECAUSE IT DOES NOT WORK
          // IN MULTI-THREAD
          m_arcane_main->setErrorCode(8);
      }
    }
    {
      Real init_time = init_timer.totalTime();
      Real loop_time = loop_timer.totalTime();
      trace->info(0) << "TotalReel = " << (init_time + loop_time)
                     << " seconds (init: "
                     << init_time << "  loop: " << loop_time << " )";
    }
    {
      Timer::Action ts_action(sub_domain, "Exit");
      trace_policy->setDefaultVerboseLevel(sd_trace, Trace::DEFAULT_VERBOSITY_LEVEL);
      sub_domain->doExitModules();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMainBatch::SessionExec::
_printStats(ISubDomain* sub_domain, ITraceMng* trace, ITimeStats* time_stat)
{
  ExecutionStatsDumper exec_dumper(trace);
  exec_dumper.dumpStats(sub_domain, time_stat);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMainBatch::SessionExec::
_execDirectTest(IParallelMng* world_pm, const String& test_name, bool is_collective)
{
  ITraceMng* trace = world_pm->traceMng();
  trace->info() << "Direct test name=" << test_name;
  trace->flush();
  ServiceFinder2T<IDirectExecution, IApplication> sf(_application(), _application());
  Ref<IDirectExecution> exec(sf.createReference(test_name));
  if (!exec) {
    String msg = String::format("Can not find 'IDirectExecution' service name '{0}'", test_name);
    if (is_collective)
      throw ParallelFatalErrorException(A_FUNCINFO, msg);
    else
      throw FatalErrorException(A_FUNCINFO, msg);
  }
  else {
    trace->info() << "Begin execution of direct service";
    trace->flush();
  }
  exec->setParallelMng(world_pm);
  exec->execute();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMainBatch::
finalize()
{
  if (m_session) {
    m_session->endSession(errorCode());
    _application()->removeSession(m_session);
    delete m_session;
    m_session = nullptr;
  }
  m_code_service.reset();
  delete m_session_exec;
  m_session_exec = nullptr;

  ITraceMng* tm = _application()->traceMng();
  Accelerator::RunnerInternal::finalize(tm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMainBatch::
doAbort()
{
  if (m_session)
    m_session->doAbort();
  else {
    // To finish cleanly even if stopped before session creation
    // or after session destruction.
    IParallelSuperMng* psm = application()->parallelSuperMng();
    if (psm)
      psm->tryAbort();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename V> void ArcaneMainBatchProperties::
_applyPropertyVisitor(V& p)
{
  auto b = p.builder();

  p << b.addInt32("MaxIteration")
       .addDescription("Maximum number of iteration")
       .addCommandLineArgument("MaxIteration")
       .addGetter([](auto a) { return a.x.m_max_iteration; })
       .addSetter([](auto a) { a.x.m_max_iteration = a.v; });

  p << b.addBool("Continue")
       .addDescription("True if continue from previous execution (restart)")
       .addCommandLineArgument("Continue")
       .addGetter([](auto a) { return a.x.m_is_continue; })
       .addSetter([](auto a) { a.x.m_is_continue = a.v; });

  p << b.addString("IdleService")
       .addDescription("Name of the idle service for additionnal cores")
       .addCommandLineArgument("IdleService")
       .addGetter([](auto a) { return a.x.m_idle_service_name; })
       .addSetter([](auto a) { a.x.m_idle_service_name = a.v; });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_PROPERTY_CLASS(ArcaneMainBatchProperties, ());

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
