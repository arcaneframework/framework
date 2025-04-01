// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneMainBatch.cc                                          (C) 2000-2024 */
/*                                                                           */
/* Gestion de l'exécution en mode Batch.                                     */
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
#include "arcane/utils/Property.h"
#include "arcane/utils/ParameterListPropertyReader.h"
#include "arcane/utils/CommandLineArguments.h"
#include "arcane/utils/CriticalSection.h"

#include "arcane/impl/ArcaneMain.h"
#include "arcane/impl/ParallelReplication.h"

#include "arcane/IIOMng.h"
#include "arcane/ICodeService.h"
#include "arcane/ISession.h"
#include "arcane/Timer.h"
#include "arcane/ISubDomain.h"
#include "arcane/IApplication.h"
#include "arcane/ITimeLoopMng.h"
#include "arcane/ITimeStats.h"
#include "arcane/SequentialSection.h"
#include "arcane/IParallelSuperMng.h"
#include "arcane/ITimeHistoryMng.h"
#include "arcane/IDirectExecution.h"
#include "arcane/IDirectSubDomainExecuteFunctor.h"
#include "arcane/ICaseMng.h"
#include "arcane/ServiceFinder2.h"
#include "arcane/SubDomainBuildInfo.h"
#include "arcane/IParallelMng.h"
#include "arcane/IMainFactory.h"
#include "arcane/ApplicationBuildInfo.h"
#include "arcane/CaseDatasetSource.h"

#include "arcane/ServiceUtils.h"

#include "arcane/impl/ExecutionStatsDumper.h"
#include "arcane/impl/TimeLoopReader.h"

#include "arcane/accelerator/core/internal/RunnerInternal.h"

#include <thread>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Propriétés associées à ArcaneMain
class ArcaneMainBatchProperties
{
  ARCANE_DECLARE_PROPERTY_CLASS(ArcaneMainBatchProperties);
 public:
  Int32 m_max_iteration = 0;
  bool m_is_continue = false;
  String m_idle_service_name; //!< Nom du service pour les CPU non utilisés
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Exécution en mode batch d'un code.
 */
class ArcaneMainBatch
: public ArcaneMain
{
 public:

  //! Informations d'exécution pour une session.
  class SessionExec
  {
    //! Infos par sous-domaine qui doivent être détruites à la fin de l'exécution
    class SubInfo
    {
     public:
      SubInfo()
      : m_sub_domain(nullptr), m_time_stats(nullptr), m_want_print_stats(false) {}
      ~SubInfo()
      {
        // Il faut d'abord détruire la ITimeStats car il utilise
        // les TimerMng des IParallelMng.
        delete m_time_stats;
        //delete m_rank_parallel_mng;
        //m_world_parallel_mng.reset();
        // Le sous-domaine est détruit lorsque la session se termine
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
    SessionExec(ArcaneMainBatch* arcane_main,ISession* session,Int32 nb_local_rank)
    : m_arcane_main(arcane_main),
      m_session(session),
      m_has_sub_domain_threads(m_arcane_main->m_has_sub_domain_threads),
      m_direct_test_name(m_arcane_main->m_direct_test_name),
      m_properties(m_arcane_main->m_properties),
      m_code_service(m_arcane_main->m_code_service),
      m_sub_infos(nb_local_rank),
      m_direct_sub_domain_execute_functor(m_arcane_main->_directExecuteFunctor())
    {
      const CaseDatasetSource& dataset_source = m_arcane_main->applicationBuildInfo().caseDatasetSource();
      m_case_file = dataset_source.fileName();
      m_case_bytes = dataset_source.content();
      // Les sub_infos de chaque thread sont créés executeRank()
      m_sub_infos.fill(nullptr);
    }
    ~SessionExec()
    {
      for( Integer i=0, n=m_sub_infos.size(); i<n; ++i )
        delete m_sub_infos[i];
    }
   public:
    // Collective sur les threads du processus
    void executeRank(Int32 local_rank);
   private:
    IApplication* _application() { return m_arcane_main->application(); }
   private:
    ArcaneMainBatch* m_arcane_main;
    ISession* m_session;
    bool m_has_sub_domain_threads; //!< indique si on utilise des threads pour gérer des sous-domaines
    String m_direct_test_name;
    String m_case_file; //!< Nom du fichier contenant le cas.
    UniqueArray<std::byte> m_case_bytes; //!< Contenu du jeu de données du cas sous forme d'un document XML.
    const ArcaneMainBatchProperties m_properties; //!< Propriétés d'exécution.
    Ref<ICodeService> m_code_service; //!< Service du code.
    UniqueArray<SubInfo*> m_sub_infos;
    IDirectSubDomainExecuteFunctor* m_direct_sub_domain_execute_functor;
   private:
    void _execDirectTest(IParallelMng* pm,const String& test_name,bool is_collective);
    void _printStats(ISubDomain* sd,ITraceMng* trace,ITimeStats* time_stat);
    void _createAndRunSubDomain(SubInfo* sub_info,Ref<IParallelMng> pm,Ref<IParallelMng> all_replica_pm,Int32 local_rank);
  };

  class ExecFunctor
  : public IFunctor
  {
   public:
    ExecFunctor(SessionExec* session_exec,Int32 local_rank)
    : m_session_exec(session_exec), m_local_rank(local_rank)
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

  ArcaneMainBatch(const ApplicationInfo&,IMainFactory*);
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
  bool m_init_only; //!< \a true si on ne fait que l'initialisation.
  bool m_check_case_only; //!< \a true si on ne fait que vérifier le jeu de données.
  bool m_has_sub_domain_threads; //!< indique si on utilise des threads pour gérer des sous-domaines
  String m_case_name; //!< Nom du cas
  String m_direct_exec_name;
  String m_direct_test_name;
  Ref<ICodeService> m_code_service; //!< Service du code.
  SessionExec* m_session_exec = nullptr;

 private:
  
  bool _sequentialParseArgs(StringList args);
};
  
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_IMPL_EXPORT IArcaneMain*
createArcaneMainBatch(const ApplicationInfo& app_info,IMainFactory* main_factory)
{
  return new ArcaneMainBatch(app_info,main_factory);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneMainBatch::
ArcaneMainBatch(const ApplicationInfo& exe_info,IMainFactory* main_factory)
: ArcaneMain(exe_info,main_factory)
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
  // Normalement finalize() doit avoir été appelé pour libérer les
  // différents objets (m_session, m_code_service, ...).
  // Si ce n'est pas le cas, c'est probablement du à une exception et dans
  // ce cas on ne fait rien pour éviter de détruire des objets dont on ne
  // connait pas trop l'état interne.
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
 * Les variables ARCANE_NB_SUB_DOMAIN & ARCANE_IDLE_SERVICE sont moins prioritaires
 * que les arguments passés à l'exécutable.
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
  String us_nb_sub_domain("nb_sub_domain");
  String us_nb_replication("nb_replication");
  String us_idle_service("idle_service");

  // Remplit 'm_properties' en fonction des paramètres de la ligne de commande
  // TODO: Ce mécanisme est disponible depuis janvier 2021. A terme, il faudra
  // rendre obsolète et supprimer la possibilité de spécifier les options
  // via '-arcane_opt'.
  properties::readFromParameterList(applicationInfo().commandLineArguments().parameters(),m_properties);

  CaseDatasetSource& dataset_source = _applicationBuildInfo().caseDatasetSource();
  // Indique si on a un jeu de données.
  bool has_case_dataset_content = !(dataset_source.fileName().empty() && dataset_source.content().empty());
  Integer nb_arg = args.count();
  if (nb_arg<2 && !has_case_dataset_content){
    trace->info() << "Usage: programm input_data ; for more information: program -arcane_opt help";
    trace->pfatal() << "No input data specified.";
  }
  
  StringList unknown_args;
  StringBuilder tool_args_xml;
  String tool_mesh;
  
  String nb_sub_domain_str;
  String nb_replication_str;
  String idle_service_name = platform::getEnvironmentVariable("ARCANE_IDLE_SERVICE");
  if (!idle_service_name.null())
    m_properties.m_idle_service_name = idle_service_name;
      
  for( Integer i=1, s=nb_arg-1; i<s; ++i ){
    // cerr << "** ARGS ARGS " << i << ' ' << args[i] << '\n';
    if (args[i]!=us_arcane_opt){
      unknown_args.add(args[i]);
      continue;
    }
    bool is_valid_opt = false;
    ++i;
    String str;
    if (i<s)
      str = args[i];
    if (str==us_init_only){
      m_init_only = true;
      is_valid_opt = true;
    }
    else if (str==us_check_case_only){
      m_check_case_only = true;
      is_valid_opt = true;
    }
    else if (str==us_continue){
      m_properties.m_is_continue = true;
      is_valid_opt = true;
    }
    else if (str==us_max_iteration){
      ++i;
      if (i<s){
        m_properties.m_max_iteration = CStringUtils::toInteger(args[i].localstr());
        //cerr << "** MAX ITER " << m_max_iteration << '\n';
        is_valid_opt = true;
      }
      else
        trace->pfatal() << "Option 'max_iteration' must specify the number of iterations";
    }
    // Nom du cas.
    else if (str==us_casename){
      ++i;
      if (i<s){
        m_case_name = args[i];
        is_valid_opt = true;
      }
    }
    else if (str==us_direct_exec){
      ++i;
      if (i<s){
        m_direct_exec_name = args[i];
        //trace->info()<<"[ArcaneMainBatch] m_direct_exec_name="<<args[i];
        is_valid_opt = true;
      }
    }
    else if (str==us_direct_test){
      ++i;
      if (i<s){
        m_direct_test_name = args[i];
        //trace->info()<<"[ArcaneMainBatch] m_direct_test_name="<<args[i];
        is_valid_opt = true;
      }
    }
    else if (str==us_tool_arg){
      ++i;
      String arg;
      String value;
      if (i<s){
        arg = args[i];
      }
      ++i;
      if (i<s){
        value = args[i];
        is_valid_opt = true;
        tool_args_xml += String::format("<{0}>{1}</{2}>\n",arg,value,arg);
      }
    }
    else if (str==us_nb_sub_domain){
      ++i;
      if (i<s){
        nb_sub_domain_str = args[i];
        //trace->info()<<"[ArcaneMainBatch] nb_sub_domain_str="<<args[i];
        is_valid_opt = true;
      }
    }
    else if (str==us_nb_replication){
      ++i;
      if (i<s){
        nb_replication_str = args[i];
        //trace->info()<<"[ArcaneMainBatch] nb_sub_domain_str="<<args[i];
        is_valid_opt = true;
      }
    }
    else if (str==us_idle_service){
      ++i;
      if (i<s){
        m_properties.m_idle_service_name = args[i];
        //trace->info()<<"[ArcaneMainBatch] m_idle_service_name="<<args[i];
        is_valid_opt = true;
      }
    }
    if (!is_valid_opt){
      trace->pfatal() << "Unknown Arcane option <" << str << ">\n";
    }
  }

  bool use_direct_test = (!m_direct_test_name.null());
  bool use_direct_exec = (!m_direct_exec_name.null());

  if (use_direct_test){
  }
  else if (use_direct_exec){
    // Dans ce cas, le dernier argument de la ligne de commande est
    // le nom du maillage.
    tool_mesh = args[nb_arg-1];
    dataset_source.setFileName("Dummy.arc");
  }
  else{
    // Le nom du cas est contenu dans le dernier argument de la ligne
    // de commande. On prend cet argument sauf si un nom de fichier
    // a déjà été positionné avant d'initialiser Arcane.
    if (dataset_source.fileName().empty() && dataset_source.content().empty())
      dataset_source.setFileName(args[nb_arg-1]);
  }

  if (!nb_sub_domain_str.null()){
    Int32 nb_sub_domain = 0;
    bool is_bad = builtInGetValue(nb_sub_domain,nb_sub_domain_str);
    if (is_bad || nb_sub_domain<=0){
      trace->pfatal() << "Invalid number of subdomains : " << nb_sub_domain;
    }
    trace->info() << "Use '" << nb_sub_domain << "' subdomains";
    _applicationBuildInfo().setNbProcessusSubDomain(nb_sub_domain);
  }

  if (!nb_replication_str.null()){
    Int32 nb_replication = 0;
    bool is_bad = builtInGetValue(nb_replication,nb_replication_str);
    if (is_bad || nb_replication<0){
      trace->pfatal() << "Invalid number of replication : " << nb_replication;
    }
    trace->info() << "Use replication of subdomains nb_replication=" << nb_replication;
    _applicationBuildInfo().setNbReplicationSubDomain(nb_replication);
  }

  if (_applicationBuildInfo().nbReplicationSubDomain()!=0 && _applicationBuildInfo().nbProcessusSubDomain()!=0)
    trace->pfatal() << "The subdomains number of replication and restriction options are incompatible.";

  if (!use_direct_test){
    String case_file = dataset_source.fileName();
    //trace->info()<<"[ArcaneMainBatch] !use_direct_test, getCodeService";
    m_code_service = _application()->getCodeService(case_file);

    if (!m_code_service){
      trace->info()  << "The file `" << case_file << "' is not a known file type.";
      case_file = args[nb_arg-2];
  
      m_code_service = _application()->getCodeService(case_file);
      if (!m_code_service){
        trace->pfatal() << "File extension not valid.";
      }
    }
  }

  if (use_direct_exec){
    //trace->info()<<"[ArcaneMainBatch] use_direct_test!";
    // Analyse les arguments qui correspondent aux options d'exécution directes
    // et construit un fichier xml à partir de la.
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
    s += String::format("  <filename>{0}</filename>\n",tool_mesh);
    s += "   </mesh>\n";
    s += " </meshes>\n";
    s += " <arcane-direct-execution>\n";
    s += String::format("  <tool name='{0}'>\n",m_direct_exec_name);
    s += tool_args_xml;
    s += "  </tool>\n";
    s += " </arcane-direct-execution>\n";
    s += "</case>\n";
    dataset_source.setFileName("(None)");
    String buf = s;
    dataset_source.setContent(buf.utf8());
    trace->info() << "Direct exec xml file=" << s;
  }
  
  if (m_code_service.get()){
    bool is_bad = m_code_service->parseArgs(unknown_args);
    if (is_bad)
      return true;
  }

  if (!unknown_args.empty()){
    trace->info()<< "Unknown command line option: " << unknown_args[0];
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
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * Cette fonction est celle appelée lors de la création d'un thread.
 */
void
_ThreadWrapper(LaunchThreadInfo* lti)
{
  ArcaneMainBatch* amb = lti->arcane_main;
  IApplication* main_app = lti->application;
  ArcaneMainBatch::ExecFunctor functor(lti->session_exec,lti->thread_index);
  bool clean_abort = false;
  bool is_master = lti->thread_index == 0;
  int r = ArcaneMain::callFunctorWithCatchedException(&functor,amb,&clean_abort,is_master);
  if (r!=0 && !clean_abort){
    // Le thread est terminé mais comme il est le seul à avoir planté,
    // il est possible que les autres soient bloqués.
    // Dans ce cas, on fait un abort pour éviter un blocage
    // TODO: essayer de tuer les autres threads correctement.
    if (main_app){
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
  if (nb_wanted_sub_domain>nb_total_rank)
    ARCANE_THROW(ArgumentException,"Number of subdomain '{0}' > number of allocated cores '{1}",
                 nb_wanted_sub_domain,nb_total_rank);

  Integer nb_local_rank = psm->nbLocalSubDomain();
  trace->info() << "NB_LOCAL_RANK=" << nb_local_rank;
  if (nb_local_rank>=1)
    m_has_sub_domain_threads = true;
  int return_value = 0;

  // Lecture des données du jeu de données.
  if (dataset_source.content().empty() && m_direct_test_name.null()){
    String case_file = dataset_source.fileName();
    trace->info() << "Reading input data '" << case_file << "'";
    IIOMng* io_mng = _application()->ioMng();
    UniqueArray<Byte> case_bytes;
    bool is_bad = io_mng->collectiveRead(case_file,case_bytes);
    if (is_bad)
      ARCANE_THROW(ParallelFatalErrorException,"Cannot read input data file '{0}'",case_file);
    dataset_source.setContent(case_bytes);
  }

  m_session_exec = new SessionExec(this,m_session,nb_local_rank);

  UniqueArray<LaunchThreadInfo> thinfo(nb_local_rank);
  for( Integer i=0; i<nb_local_rank; ++i ){
    thinfo[i].arcane_main = this;
    thinfo[i].session_exec = m_session_exec;
    thinfo[i].application = _application();
    thinfo[i].thread_index = i;
  }

  if (nb_local_rank>1){
    UniqueArray<std::thread*> gths(nb_local_rank);
    for( Integer i=0; i<nb_local_rank; ++i ){
      gths[i] = new std::thread(_ThreadWrapper,&thinfo[i]);
    }
    for( Integer i=0; i<nb_local_rank; ++i ){
      gths[i]->join();
      delete gths[i];
    }
  }
  else{
    m_has_sub_domain_threads = false;
    m_session_exec->executeRank(0);
  }

  // TODO: supprimer car inutile car vaut toujours 0.
  return return_value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * En mode avec un sous-domaine par thread,cette fonction est appelée
 * par chaque thread (de manière potentiellement concurrente) pour son sous-domaine.
 * \a local_rank indique le rang local du thread, qui est compris
 * entre 0 et \a nb_local_sub_domain (tèl que définit dans execute()).
 */
void ArcaneMainBatch::SessionExec::
executeRank(Int32 local_rank)
{
  // ATTENTION:
  // Cette fonction doit etre reentrente...

  auto sub_info = new SubInfo();
  m_sub_infos[local_rank] = sub_info;

  IProcessorAffinityService* pas = platform::getProcessorAffinityService();
  if (pas && m_has_sub_domain_threads){
    // Il ne faut binder les CPU que si demandé et uniquement si
    // le nombre de threads au total (sur l'ensemble des processus)
    // ne dépasse pas le nombre de coeur de la machine.
    if (!platform::getEnvironmentVariable("ARCANE_BIND_THREADS").null()){
      ITraceMng* tm = _application()->traceMng();
      tm->info() << "Binding threads";
      pas->bindThread(local_rank);
    }
  }

  // Création du gestionnaire de parallélisme pour l'ensemble des rangs alloués.
  IParallelSuperMng* psm = _application()->parallelSuperMng();
  Ref<IParallelMng> world_pm = psm->internalCreateWorldParallelMng(local_rank);
  sub_info->m_world_parallel_mng = world_pm;
  
  if (!m_direct_test_name.null()){
    _execDirectTest(world_pm.get(),m_direct_test_name,true);
    return;
  }

  // Regarde si on souhaite exécuter le calcul sur un sous-ensemble
  // des ressources allouées. Pour l'instant, il est uniquement possible
  // de choisir un nombre de sous-domaine. Si c'est le cas, seuls
  // les rangs de 0 au nombre de sous-domaine souhaité moins 1 sont
  // utilisés. Les rangs supérieurs n'ont pas de sous-domaines
  // et à la place utilisent un service qui implémente IDirectExecution

  // Création du gestionnaire des statistiques d'exécution.
  ITraceMng* trace = world_pm->traceMng();
  String stat_name = "Rank";
  stat_name = stat_name + world_pm->commRank();
  ITimeStats* time_stat = _application()->mainFactory()->createTimeStats(world_pm->timerMng(),trace,stat_name);
  sub_info->m_time_stats = time_stat;
  time_stat->beginGatherStats();
  world_pm->setTimeStats(time_stat);

  Ref<IParallelMng> pm = world_pm;
  Ref<IParallelMng> all_replica_pm = pm;

  const Integer nb_wanted_sub_domain = _application()->applicationBuildInfo().nbProcessusSubDomain();
  const Integer nb_wanted_replication = _application()->applicationBuildInfo().nbReplicationSubDomain();
  // On est en parallèle et on souhaite moins de sous-domaines que de processus alloués
  if (world_pm->isParallel()){
    // Pour l'instant, on ne peut pas mélanger la réplication de sous-domaines avec
    // un nombre de sous-domaines différent du nombre de processeurs alloués.
    // TODO: lorsque ce ne sera plus le cas, il faudra faire un all_replica_pm qui
    // contiendra l'ensemble des sous-domaines et des réplica.

    if (nb_wanted_replication>1){
      Int32 comm_size = world_pm->commSize();
      Int32 nb_sub_part = comm_size / nb_wanted_replication;
      trace->info() << "Using sub-domain replication nb_sub_part=" << nb_sub_part;
      if ((comm_size % nb_wanted_replication)!=0)
        ARCANE_FATAL("The number of replication '{0}' must be a common factor of the number of allocated cores '{1}",
                     nb_wanted_replication,comm_size);
      // D'abord, on créé un communicateur contenant les réplicats de chaque sous-domaine
      // Ce communicateur contiendra donc \a m_nb_wanted_replication objets
      Ref<IParallelMng> replicate_pm;
      trace->info() << "Building replicated parallel mng";
      {
        Int32UniqueArray kept_ranks(nb_wanted_replication);
        for( Integer i_sd=0; i_sd<nb_sub_part; ++i_sd ){
          for( Int32 i=0; i<nb_wanted_replication; ++i ){
            kept_ranks[i] = i_sd + (i*nb_sub_part);
            trace->info() << "Rank r=" << kept_ranks[i];
          }
          Ref<IParallelMng> new_pm = world_pm->createSubParallelMngRef(kept_ranks);
          if (new_pm.get()){
            replicate_pm = new_pm;
            replicate_pm->setTimeStats(time_stat);
            trace->info() << " Building own replicated parallel mng";
          }
          else{
            trace->info()<<"!pm";
          }
          trace->flush();
        }
      }
      if (!replicate_pm)
        ARCANE_FATAL("Null replicated parallel mng");

      // Maintenant, on créé un IParallelMng qui correspond à l'ensemble
      // des rangs d'un même réplica. Ce IParallelMng sera assigné au
      // sous-domaine qui sera créé par la suite.
      trace->info() << "Building sub-domain parallel mng";
      {
        Int32UniqueArray kept_ranks(nb_sub_part);
        for( Integer i_repl=0; i_repl<nb_wanted_replication; ++i_repl ){
          for( Int32 i=0; i<nb_sub_part; ++i ){
            kept_ranks[i] = i + (i_repl*nb_sub_part);
            trace->info() << "Rank r=" << kept_ranks[i];
          }
          Ref<IParallelMng> new_pm = world_pm->createSubParallelMngRef(kept_ranks);
          if (new_pm.get()){
            pm = new_pm;
            if (nb_sub_part==1){
              // Il faut prendre la version séquentielle pour faire comme si le calcul
              // était séquentiel. Ce gestionnaire sera détruit en même temps
              // que \a new_pm
              pm = new_pm->sequentialParallelMngRef();
            }
            trace->info()<<"pm: setting time_stat & m_rank_parallel_mng for replica rank=" << i_repl;
            trace->flush();
            pm->setTimeStats(time_stat);
            sub_info->m_rank_parallel_mng = new_pm;
            auto pr = new ParallelReplication(i_repl,nb_wanted_replication,replicate_pm);
            pm->setReplication(pr);
          }
          else{
            trace->info()<<"!pm";
            trace->flush();
          }
        }
      }
    }
    else if (nb_wanted_sub_domain!=0){
      const Int32 nb_sub_part = nb_wanted_sub_domain;
      Int32UniqueArray kept_ranks(nb_sub_part);
      for( Int32 i=0; i<nb_sub_part; ++i )
        kept_ranks[i] = i;
      pm = world_pm->createSubParallelMngRef(kept_ranks);
      if (pm.get()){
        trace->info()<<"pm: setting time_stat & m_rank_parallel_mng";
        trace->flush();
        pm->setTimeStats(time_stat);
        sub_info->m_rank_parallel_mng = pm;
        all_replica_pm = pm;
      }
      else{
        trace->info()<<"!pm";
        trace->flush();
      }
    }
  }

  bool print_stats = false;
  ISubDomain* sub_domain = nullptr;

  if (!pm){
    // Si ici, il s'agit d'un rang qui ne possède pas de sous-domaine.
    // Dans ce cas, exécute le service donnée par 'm_idle_service_name'
    // (si spécifié, sinon ne fait rien)
    trace->info()<<"The rank doesn't own any subdomain!";
    if (m_properties.m_idle_service_name.empty()){
      trace->info() << "No idle service specified"; trace->flush();
    }
    else{
      trace->info()<<"execDirectTest: "<< m_properties.m_idle_service_name;
      trace->flush();
      _execDirectTest(world_pm.get(),m_properties.m_idle_service_name,false);
      // On sort de l'execute() du directTest grâce au broadcast(This is the end), il faut s'en retourner
      return;
    }
    print_stats = true;
  }
  else {
    _createAndRunSubDomain(sub_info,pm,all_replica_pm,local_rank);
    sub_domain = sub_info->m_sub_domain;
    print_stats = sub_info->m_want_print_stats;
  }

  time_stat->endGatherStats();

  if (print_stats && sub_domain){
    // S'assure que tout le monde est ici avant d'arêter le profiling
    // TODO: Comme le profiling est local au processus, il suffirait
    // a priori de faire la barrière sur les IParallelMng locaux.
    IParallelMng* pm = sub_domain->parallelMng();
    pm->barrier();
    if (local_rank==0)
      Accelerator::RunnerInternal::stopAllProfiling();
    pm->barrier();
    _printStats(sub_domain,trace,time_stat);
  }

  //BaseForm[Hash["This is the end", "CRC32"], 16]
  // On informes les 'autres' capacités qu'il faut s'en aller, maintenant!
  world_pm->broadcast(UniqueArray<unsigned long>(1,0xdfeb699fl).view(),0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMainBatch::SessionExec::
_createAndRunSubDomain(SubInfo* sub_info,Ref<IParallelMng> pm,Ref<IParallelMng> all_replica_pm,Int32 local_rank)
{
  // Il s'agit d'un rang qui a un sous-domaine.
  // Celui-ci est créé et l'exécution commence.
  SubDomainBuildInfo sdbi(pm,local_rank,all_replica_pm);
  sdbi.setCaseFileName(m_case_file);
  sdbi.setCaseContent(m_case_bytes);
  ISubDomain* sub_domain = m_code_service->createAndLoadCase(m_session,sdbi);
  sub_info->m_sub_domain = sub_domain;

  ITraceMng* trace = _application()->traceMng();
  ITraceMng* sd_trace = sub_domain->traceMng();
  ITraceMngPolicy* trace_policy = _application()->getTraceMngPolicy();

  // En cas de réplication, désactive les sorties courbes
  // des réplicats.
  trace->info() << "REPLICATION: rank=" << pm->replication()->replicationRank();

  if (!pm->replication()->isMasterRank()){
    trace->info() << "Disable output curves for replicates.";
    sub_domain->timeHistoryMng()->setDumpActive(false);
  }

  // TODO:
  // Détruire le sous-domaine à la fin de la fonction mais il
  // faut pour cela modifier ISession pour supporter la suppression
  // d'un sous-domaine (et ensuite détruire ISession).

  IProcessorAffinityService* pas = platform::getProcessorAffinityService();
  if (pas){
    String cpu_set = pas->cpuSetString();
    trace->info() << " CpuSet=" << cpu_set;
  }

  if (m_arcane_main->m_check_case_only){
    trace->info() << "Checking the input data";
    // Initialise les modules de la boucle en temps
    {
      TimeLoopReader stl(_application());
      stl.readTimeLoops();
      stl.registerTimeLoops(sub_domain);
      stl.setUsedTimeLoop(sub_domain);
    }
    ICaseMng* cm = sub_domain->caseMng();
    cm->readOptions(true);
  }
  else{
    Timer init_timer(sub_domain,"InitTimer",Timer::TimerReal);
    Timer loop_timer(sub_domain,"LoopTimer",Timer::TimerReal);

    {
      Timer::Action ts_action(sub_domain,"Init");
      Timer::Sentry ts(&init_timer);

      m_code_service->initCase(sub_domain,m_properties.m_is_continue);
    }

    if (m_properties.m_max_iteration>0)
      trace->info() << "Option 'max_iteration' activated with " << m_properties.m_max_iteration;

    // Redirige les signaux.
    // Cela se fait aussi a l'initialisation mais ici on peut être dans un autre
    // thread et de plus certaines bibliothèques ont pu rediriger les signaux
    // lors de l'init
    {
      CriticalSection cs(pm->threadMng());
      ArcaneMain::redirectSignals();
    }
    int ret_compute_loop = 0;

    IDirectExecution* direct_exec = sub_domain->directExecution();
    if (direct_exec && direct_exec->isActive()){
      trace->info() << "Direct execution activated";
      direct_exec->execute();
    }
    else if (m_arcane_main->m_init_only){
      trace->info() << "Option 'init_only' activated";
      sub_info->m_want_print_stats = true;
    }
    else{
      sub_info->m_want_print_stats = true;
      Timer::Action ts_action(sub_domain,"Loop");
      Timer::Sentry ts(&loop_timer);
      // Lors de la boucle de calcul, ne force pas l'affichage des traces à un niveau
      // donné (ce qui est fait lors de l'initialisation de l'application.
      trace_policy->setDefaultVerboseLevel(sd_trace,Trace::UNSPECIFIED_VERBOSITY_LEVEL);
      if (m_direct_sub_domain_execute_functor){
        m_direct_sub_domain_execute_functor->setSubDomain(sub_domain);
        m_direct_sub_domain_execute_functor->execute();
        sub_domain->parallelMng()->barrier();
      }
      else{
        ret_compute_loop = sub_domain->timeLoopMng()->doComputeLoop(m_properties.m_max_iteration);
        if (ret_compute_loop<0)
          //TODO: NE PAS REMPLIR DIRECTEMENT CETTE FONCTION CAR CELA NE MARCHE
          // PAS EN MULTI-THREAD
          m_arcane_main->setErrorCode(8);
      }
    }
    {
      Real init_time = init_timer.totalTime();
      Real loop_time = loop_timer.totalTime();
      trace->info(0) << "TotalReel = " << (init_time+loop_time)
                     << " secondes (init: "
                     << init_time << "  loop: " << loop_time << " )";
    }
    {
      Timer::Action ts_action(sub_domain,"Exit");
      trace_policy->setDefaultVerboseLevel(sd_trace,Trace::DEFAULT_VERBOSITY_LEVEL);
      sub_domain->doExitModules();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMainBatch::SessionExec::
_printStats(ISubDomain* sub_domain,ITraceMng* trace,ITimeStats* time_stat)
{
  ExecutionStatsDumper exec_dumper(trace);
  exec_dumper.dumpStats(sub_domain,time_stat);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMainBatch::SessionExec::
_execDirectTest(IParallelMng* world_pm,const String& test_name,bool is_collective)
{
  ITraceMng* trace = world_pm->traceMng();
  trace->info() << "Direct test name=" << test_name;
  trace->flush();
  ServiceFinder2T<IDirectExecution,IApplication> sf(_application(),_application());
  Ref<IDirectExecution> exec(sf.createReference(test_name));
  if (!exec){
   String msg = String::format("Can not find 'IDirectExecution' service name '{0}'",test_name);
   if (is_collective)
     throw ParallelFatalErrorException(A_FUNCINFO,msg);
   else
     throw FatalErrorException(A_FUNCINFO,msg);
  }
  else{
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
  if (m_session){
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
  else{
    // Pour finir proprement même si arrêt avant la création de la session
    // ou après la destruction de la session.
    IParallelSuperMng* psm = application()->parallelSuperMng();
    if (psm)
      psm->tryAbort();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename V> void ArcaneMainBatchProperties::
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

ARCANE_REGISTER_PROPERTY_CLASS(ArcaneMainBatchProperties,());

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
