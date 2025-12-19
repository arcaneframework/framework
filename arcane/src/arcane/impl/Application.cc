// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Application.cc                                              (C) 2000-2025 */
/*                                                                           */
/* Superviseur.                                                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/StringUtils.h"

#include "arcane/utils/Iostream.h"
#include "arcane/utils/Iterator.h"
#include "arcane/utils/List.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/Deleter.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/TraceClassConfig.h"
#include "arcane/utils/IMemoryInfo.h"
#include "arcane/utils/IStackTraceService.h"
#include "arcane/utils/ISymbolizerService.h"
#include "arcane/utils/IProcessorAffinityService.h"
#include "arcane/utils/IOnlineDebuggerService.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/IProfilingService.h"
#include "arcane/utils/IThreadImplementationService.h"
#include "arccore/base/internal/IDynamicLibraryLoader.h"
#include "arcane/utils/IPerformanceCounterService.h"
#include "arcane/utils/ITraceMngPolicy.h"
#include "arcane/utils/JSONReader.h"
#include "arcane/utils/Profiling.h"

#include "arccore/base/internal/DependencyInjection.h"
#include "arccore/concurrency/internal/TaskFactoryInternal.h"

#include "arcane/core/ArcaneVersion.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IIOMng.h"
#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/IParallelSuperMng.h"
#include "arcane/core/IMainFactory.h"
#include "arcane/core/IArcaneMain.h"
#include "arcane/core/IRessourceMng.h"
#include "arcane/core/IServiceLoader.h"
#include "arcane/core/IServiceMng.h"
#include "arcane/core/ICodeService.h"
#include "arcane/core/ISession.h"
#include "arcane/core/IDataFactory.h"
#include "arcane/core/IDataFactoryMng.h"
#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/XmlNode.h"
#include "arcane/core/XmlNodeList.h"
#include "arcane/core/ItemTypeMng.h"
#include "arcane/core/Concurrency.h"
#include "arcane/core/IPhysicalUnitSystemService.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/Configuration.h"
#include "arcane/core/Directory.h"
#include "arcane/core/IServiceAndModuleFactoryMng.h"
#include "arcane/core/ApplicationBuildInfo.h"

#include "arcane/core/IItemEnumeratorTracer.h"
#include "arcane/impl/Application.h"
#include "arcane/impl/ConfigurationReader.h"
#include "arcane/impl/ArcaneMain.h"

// Ces fichiers ne sont utilisés que pour afficher des tailles
// des classes définies dans ces fichiers
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/Item.h"
#include "arcane/core/IndexedItemConnectivityView.h"
#include "arcane/core/UnstructuredMeshConnectivity.h"

#include "arccore_version.h"

#ifdef ARCANE_OS_WIN32
#include <windows.h>
#endif

#include <vector>
#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" IPhysicalUnitSystemService*
createNullPhysicalUnitSystemService();
extern "C++" IConfigurationMng*
arcaneCreateConfigurationMng(ITraceMng* tm);
extern "C++" ARCANE_IMPL_EXPORT IServiceAndModuleFactoryMng*
arcaneCreateServiceAndModuleFactoryMng(ITraceMng* tm);
extern "C++" ARCANE_IMPL_EXPORT Ref<IItemEnumeratorTracer>
arcaneCreateItemEnumeratorTracer(ITraceMng* tm,Ref<IPerformanceCounterService> perf_counter);
extern "C++" ARCANE_IMPL_EXPORT Ref<ICodeService>
createArcaneCodeService(IApplication* app);
extern "C++" ARCANE_CORE_EXPORT void
arcaneSetSingletonItemEnumeratorTracer(Ref<IItemEnumeratorTracer> tracer);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" IApplication*
arcaneCreateApplication(IArcaneMain* am)
{
  IApplication* sm = new Application(am);
  sm->build();
  return sm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Construit un superviseur avec les informations d'exécution \a ex.
 * \warning Il n'existe qu'une instance de Application par exécution du code.
 */
Application::
Application(IArcaneMain* am)
: m_exe_info(am->applicationInfo())
, m_namespace_uri(arcaneNamespaceURI())
, m_local_name("Application")
, m_arcane_main(am)
, m_main_factory(am->mainFactory())
, m_service_mng(nullptr)
, m_sequential_parallel_super_mng(nullptr)
, m_trace(nullptr)
, m_ressource_mng(nullptr)
, m_io_mng(nullptr)
, m_configuration_mng(nullptr)
, m_main_service_factory_infos(am->registeredServiceFactoryInfos())
, m_main_module_factory_infos(am->registeredModuleFactoryInfos())
, m_has_garbage_collector(am->hasGarbageCollector())
, m_trace_policy(nullptr)
, m_is_init(false)
, m_is_master(false)
, m_service_and_module_factory_mng(nullptr)
{
  // Initialise les threads avec un service qui ne fait rien.
  platform::setThreadImplementationService(&m_null_thread_implementation);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Détruit le gestionnaire.
 *
 * Détruit le gestionnaire de message et les gestionnaires de configuration.
 */
Application::
~Application()
{
  TaskFactory::terminate();
  m_task_implementation.reset();

  // Supprime les services que l'instance a positionné
  if (platform::getProcessorAffinityService()==m_processor_affinity_service.get())
    platform::setProcessorAffinityService(nullptr);

  if (platform::getStackTraceService()==m_stack_trace_service.get())
    platform::setStackTraceService(nullptr);

  if (platform::getSymbolizerService()==m_symbolizer_service.get())
    platform::setSymbolizerService(nullptr);

  if (platform::getProfilingService()==m_profiling_service.get())
    platform::setProfilingService(nullptr);

  if (platform::getPerformanceCounterService()==m_performance_counter_service.get())
    platform::setPerformanceCounterService(nullptr);

  delete m_service_and_module_factory_mng;
  
  m_sessions.each(Deleter());

  delete m_configuration_mng;
  delete m_ressource_mng;

  m_owned_sequential_parallel_super_mng.reset();
  m_parallel_super_mng.reset();

  m_data_factory_mng.reset();
  delete m_io_mng;
  delete m_service_mng;

  m_trace = nullptr;
  // Il faut détruire le m_trace_policy après m_trace car ce dernier peut
  // l'utiliser.
  delete m_trace_policy;

  // Supprime la référence au gestionnaire de thread. Il faut le faire en dernier car
  // les autres gestionnaires peuvent l'utiliser.
  if (platform::getThreadImplementationService()==m_thread_implementation.get())
    platform::setThreadImplementationService(nullptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

UniqueArray<String> Application::
_stringListToArray(const StringList& slist) const
{
  UniqueArray<String> a;
  for( const String& s : slist )
    a.add(s);
  return a;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * Essaie d'instancier un service implémentant \a InterfaceType avec
 * la liste de nom de services \a names.  Retourne l'instance trouvée
 * si elle existe et remplit \a found_name (si non nul) avec le nom de
 * l'instance. Dès qu'une instance est trouvée, on la retourne.
 * Retourne nulle si aucune instance n'est disponible.
 *
 * \note Cette méthode n'est plus utilisée (janvier 2025) et on utilise
 * _tryCreateServiceUsingInjector() à la place.
 */
template<typename InterfaceType> Ref<InterfaceType> Application::
_tryCreateService(const StringList& names,String* found_name)
{
  if (found_name)
    (*found_name) = String();
  ServiceBuilder<InterfaceType> sf(this);
  for( String s : names ){
    auto t = sf.createReference(s,SB_AllowNull);
    if (t.get()){
      if (found_name)
        (*found_name) = s;
      return t;
    }
  }
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * Essaie d'instancier un service implémentant \a InterfaceType avec
 * la liste de nom de services \a names.  Retourne l'instance trouvée
 * si elle existe et remplit \a found_name (si non nul) avec le nom de
 * l'instance. Dès qu'une instance est trouvée, on la retourne.
 * Retourne \a nullptr si aucune instance n'est disponible.
 */
template<typename InterfaceType> Ref<InterfaceType> Application::
_tryCreateServiceUsingInjector(const StringList& names,String* found_name,bool has_trace)
{
  DependencyInjection::Injector injector;
  injector.fillWithGlobalFactories();
  // Ajoute une instance de ITraceMng* pour les services qui en ont besoin
  if (has_trace)
    injector.bind(m_trace.get());

  if (found_name)
    (*found_name) = String();
  for( String s : names ){
    auto t = injector.createInstance<InterfaceType>(s,true);
    if (t.get()){
      if (found_name)
        (*found_name) = s;
      return t;
    }
  }
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void
arcaneSetPauseOnError(bool v);

void Application::
build()
{
  const ApplicationBuildInfo& build_info = applicationBuildInfo();

  // Création du TraceMngPolicy. L'initialisation complète se fera plus tard
  // car on a besoin d'informations supplémentaires comme le rang MPI et
  // d'avoir lu des fichiers de configuration.
  m_trace_policy = m_main_factory->createTraceMngPolicy(this);

  // Toujours le premier après TraceMngPolicy
  m_trace = m_main_factory->createTraceMng();

  {
    // TODO: positionner ces informations dans ApplicationBuildInfo.
    Int32 output_level = build_info.outputLevel();
    if (output_level!=Trace::UNSPECIFIED_VERBOSITY_LEVEL){
      m_trace_policy->setVerbosityLevel(output_level);
      m_trace_policy->setStandardOutputVerbosityLevel(output_level);
    }
    Int32 verbosity_level = build_info.verbosityLevel();
    if (verbosity_level!=Trace::UNSPECIFIED_VERBOSITY_LEVEL){
      m_trace_policy->setVerbosityLevel(verbosity_level);
    }

    bool has_output_file = build_info.isMasterHasOutputFile();
    m_trace_policy->setIsMasterHasOutputFile(has_output_file);

    // Positionne le niveau de verbosité en laissant au minimum le niveau
    // par défaut. Sans cela, certains messages d'initialisation peuvent ne
    // pas s'afficher ce qui peut être génant en cas de problèmes ou de plantage.
    Int32 minimal_verbosity_level = build_info.minimalVerbosityLevel();
    if (minimal_verbosity_level==Trace::UNSPECIFIED_VERBOSITY_LEVEL)
      minimal_verbosity_level = Trace::DEFAULT_VERBOSITY_LEVEL;
    m_trace_policy->setDefaultVerboseLevel(m_trace.get(),minimal_verbosity_level);
  }

  arcaneGlobalMemoryInfo()->setTraceMng(traceMng());

  {
    // Affiche quelques informations à l'initialisation dès le niveau 4
    m_trace->info(4) << "*** Initialization informations:";
    m_trace->info(4) << "*** PID: " << platform::getProcessId();
    m_trace->info(4) << "*** Host: " << platform::getHostName();

    IDynamicLibraryLoader* dynamic_library_loader = IDynamicLibraryLoader::getDefault();
    if (dynamic_library_loader){
      String os_dir(m_exe_info.dataOsDir());
#ifdef ARCANE_OS_WIN32
      {
        // Sous windows, si le processus est lancé via 'dotnet' par exemple,
        // les chemins de recherche pour LoadLibrary() ont pu être modifiés
        // et on ne vas plus chercher par défaut dans le répertoire courant
        // pour charger les bibliothèques natives. Pour corrige ce problème
        // on repositionne le comportement qui autorise à ajouter des chemins
        // utilisateurs et on ajoute 'os_dir' à ce chemin.
        // Sans cela, les dépendances aux bibliothèques chargées par LoadLibrary()
        // ne seront pas trouvées. Par exemple 'arcane_thread.dll' dépend de 'tbb.dll' et
        // tous les deux sont le même répertoire mais si répertoire n'est pas
        // dans la liste autorisé, alors 'tbb.dll' ne sera pas trouvé.
        //
        // NOTE: On pourrait peut-être éviter cela en utilisant LoadLibraryEx() et en
        // spécifiant LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR comme flag de recherche.
        // A vérifier si cela fonctionne lorsqu'on n'utilisera plus le chargeur
        // dynamique de la glib.
        m_trace->info(4) << "Adding '" << os_dir << "' to search library path";
        std::wstring wide_os_dir = StringUtils::convertToStdWString(os_dir);
        SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        AddDllDirectory(wide_os_dir.c_str());
      }
#endif
      for( StringCollection::Enumerator i(m_exe_info.dynamicLibrariesName()); ++i; ){
        String name = *i;
        m_trace->info(4) << "*** Trying to load dynamic library: " << name;
        IDynamicLibrary* dl = dynamic_library_loader->open(os_dir,name);
        if (!dl)
          m_trace->info(4) << "WARNING: Can not load library '" << name << "'";
      }
    }

#ifdef ARCANE_OS_WIN32
    if (dynamic_library_loader){
      String os_dir(m_exe_info.dataOsDir());
      // TODO: Ajouter le répertoire contenant 'arcane_impl' qui est connu dans ArcaneMain dans m_arcane_lib_path.
      String dyn_lib_names[5] = { "arcane_mpi", "arcane_std", "arcane_mesh",
                                  "arcane_thread", "arcane_mpithread",
                                };
      for( Integer i=0; i<5; ++i )
        dynamic_library_loader->open(os_dir,dyn_lib_names[i]);
    }
#endif

    m_configuration_mng = arcaneCreateConfigurationMng(traceMng());

    {
      m_service_and_module_factory_mng = arcaneCreateServiceAndModuleFactoryMng(traceMng());
      for( ServiceFactoryInfoCollection::Enumerator i(m_main_service_factory_infos); ++i; )
        m_service_and_module_factory_mng->addGlobalFactory(*i);
      for( ModuleFactoryInfoCollection::Enumerator i(m_main_module_factory_infos); ++i; )
        m_service_and_module_factory_mng->addGlobalFactory(*i);

      m_service_and_module_factory_mng->createAllServiceRegistererFactories();
    }

    m_service_mng = m_main_factory->createServiceMng(this);

    String pause_on_error = platform::getEnvironmentVariable("ARCANE_PAUSE_ON_ERROR");
    if (!pause_on_error.null()){
      arcaneSetPauseOnError(true);
    }

    // Recherche le service utilisé pour connaitre la pile d'appel
    bool has_dbghelp = false;
    {
      ServiceBuilder<IStackTraceService> sf(this);
      Ref<IStackTraceService> sv;
      const auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_USE_BACKWARDCPP", true);
      if (v && v.value() != 0) {
        sv = sf.createReference("BackwardCppStackTraceService", SB_AllowNull);
      }
      else {
        sv = sf.createReference("LibUnwind", SB_AllowNull);
      }
      if (!sv.get()){
        sv = sf.createReference("DbgHelpStackTraceService", SB_AllowNull);
        if (sv.get())
          has_dbghelp = true;
      }
      if (sv.get()) {
        m_stack_trace_service = sv;
        platform::setStackTraceService(sv.get());
      }
    }

    // Recherche le service utilisé pour connaitre les infos sur les symboles
    // du code source. Pour l'instant on ne supporte que LLVM et on n'active ce service
    // que si la variable d'environnement ARCANE_LLVMSYMBOLIZER_PATH est définie.
    {
      Ref<ISymbolizerService> s;
      ServiceBuilder<ISymbolizerService> sf(this);
      if (!platform::getEnvironmentVariable("ARCANE_LLVMSYMBOLIZER_PATH").null()){
        s = sf.createReference("LLVMSymbolizer", SB_AllowNull);
      }
      if (!s.get() && has_dbghelp){
        s = sf.createReference("DbgHelpSymbolizerService", SB_AllowNull);
      }
      if (s.get()) {
        m_symbolizer_service = s;
        platform::setSymbolizerService(s.get());
      }
    }

    {
      StringList names = build_info.threadImplementationServices();
      String found_name;
      auto sv = _tryCreateServiceUsingInjector<IThreadImplementationService>(names,&found_name, false);
      if (sv.get()){
        m_thread_implementation_service = sv;
        m_thread_implementation = sv->createImplementation();
        platform::setThreadImplementationService(m_thread_implementation.get());
        m_thread_implementation->initialize();
        m_used_thread_service_name = found_name;
      }
      else {
        ARCANE_FATAL("Can not find implementation for 'IThreadImplementation' (names='{0}').",
                     _stringListToArray(names));
        arcaneSetHasThread(false);
      }
    }

    // Le gestionnaire de thread a pu changer et il faut donc
    // reinitialiser le gestionnaire de trace.
    m_trace->resetThreadStatus();

    // Recherche le service utilisé pour gérer les tâches
    {
      Integer nb_task_thread = build_info.nbTaskThread();
      if (nb_task_thread>=0){

        StringList names = build_info.taskImplementationServices();
        String found_name;
        auto sv = _tryCreateServiceUsingInjector<ITaskImplementation>(names,&found_name,false);
        if (sv.get()){
          TaskFactoryInternal::setImplementation(sv.get());
          //std::cout << "Initialize task with nb_thread=" << nb_task_thread << "\n";
          sv->initialize(nb_task_thread);
          m_used_task_service_name = found_name;
          m_task_implementation = sv;
        }
        else
          ARCANE_FATAL("Can not find task implementation service (names='{0}')."
                       " Please check if Arcane is configured with Intel TBB library",
                       _stringListToArray(names));
      }

      if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_TASK_VERBOSE_LEVEL",true))
        TaskFactory::setVerboseLevel(v.value());
    }

    if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_LOOP_PROFILING_LEVEL",true))
      ProfilingRegistry::setProfilingLevel(v.value());

    // Recherche le service utilisé pour le profiling
    {
      String profile_str = platform::getEnvironmentVariable("ARCANE_PROFILING");
      if (!profile_str.null()){
        ServiceBuilder<IProfilingService> sf(this);
        auto sv = sf.createReference(profile_str+"ProfilingService",SB_AllowNull);
        if (sv.get()){
          m_profiling_service = sv;
          platform::setProfilingService(sv.get());
        }
        else
          ARCANE_FATAL("Can not find profiling service (name='{0}')",profile_str);
      }
    }

    // Par défaut, on accroche le service Hyoda
    {
      ServiceBuilder<IOnlineDebuggerService> sf(this);
      auto sv = sf.createReference("Hyoda",SB_AllowNull);
      if (sv.get()){
        m_online_debugger = sv;
        platform::setOnlineDebuggerService(sv.get());
      }
    }
    
    // Recherche le service utilisé pour la gestion de l'affinité processeur
    {
      StringList names;
      names.add("HWLoc");
      String found_name;
      auto sv = _tryCreateServiceUsingInjector<IProcessorAffinityService>(names,&found_name,true);
      if (sv.get()) {
        m_processor_affinity_service = sv;
        platform::setProcessorAffinityService(sv.get());
      }
      else {
        m_trace->info(4) << "Can not find implementation for IProcessorAffinityService "
                         << "(names=" << _stringListToArray(names) << ").";
      }
    }

    // Recherche le service utilisé pour le parallélisme
    String message_passing_service = build_info.messagePassingService();
    if (message_passing_service.null())
      message_passing_service = build_info.internalDefaultMessagePassingService();
    ServiceBuilder<IParallelSuperMng> sf(this);
    auto sm = sf.createReference(message_passing_service,SB_AllowNull);
    if (!sm)
      ARCANE_FATAL("Can not find message passing service '{0}'",message_passing_service);

    m_parallel_super_mng = sm;
    m_parallel_super_mng->initialize();

    IParallelSuperMng* seq_sm = nullptr;
    if (sm->isParallel()){
      m_owned_sequential_parallel_super_mng = sf.createReference("SequentialParallelSuperMng",SB_AllowNull);
      seq_sm = m_owned_sequential_parallel_super_mng.get();
      if (!seq_sm)
        ARCANE_FATAL("Can not find service 'SequentialParallelSuperMng'");
      seq_sm->initialize();
    }
    else
      seq_sm = m_parallel_super_mng.get();

    m_sequential_parallel_super_mng = seq_sm;

    m_ressource_mng = IRessourceMng::createDefault(this);
    m_io_mng = m_main_factory->createIOMng(this);
    m_data_factory_mng = m_main_factory->createDataFactoryMngRef(this);
  }

  {
    VersionInfo version_info = m_exe_info.codeVersion();
    int vmajor = version_info.versionMajor();
    int vminor = version_info.versionMinor();
    int vpatch = version_info.versionPatch();
    m_main_version_str = String::format("{0}.{1}.{2}",vmajor,vminor,vpatch);
    m_major_and_minor_version_str = String::format("{0}.{1}",vmajor,vminor);
    m_version_str = m_major_and_minor_version_str;
    if (vpatch!=0)
      m_version_str = m_major_and_minor_version_str + "." + vpatch;
  }

  m_targetinfo_str = m_exe_info.targetFullName();
  m_application_name = m_exe_info.applicationName();
  m_code_name = m_exe_info.codeName();

  // Récupère le nom de l'utilisateur
  m_user_name = platform::getUserName();

  // Récupère le chemin du répertoire de la configuration utilisateur
  // TODO: il faut changer car dans les nouvelles recommandations POSIX,
  // le répertoire de configuration est '.config/arcane'.
  m_user_config_path = platform::getEnvironmentVariable("ARCANE_CONFIG_PATH");
  if (m_user_config_path.null()) {
    Directory user_home_env(platform::getHomeDirectory());
    m_user_config_path = Directory(user_home_env, ".arcane").path();
  }

  {
    bool is_parallel = parallelSuperMng()->isParallel();
    bool is_debug = applicationInfo().isDebug();
    // Création et initialisation du TraceMngPolicy.
    m_trace_policy->setIsParallel(is_parallel);
    m_trace_policy->setIsDebug(is_debug);
    bool is_parallel_output = is_parallel && is_debug;
    // Permet de forcer les sorties meme en mode optimisé
    {
      String s = platform::getEnvironmentVariable("ARCANE_PARALLEL_OUTPUT");
      if (!s.null())
        is_parallel_output = true;
      if (s=="0")
        is_parallel_output = false;
    }
    m_trace_policy->setIsParallelOutput(is_parallel_output);
  }

  m_is_master = m_parallel_super_mng->commRank() == 0;
 
  m_trace->info(4) << "*** UserName: " << m_user_name;
  m_trace->info(4) << "*** HomeDirectory: " << platform::getHomeDirectory();

#ifdef ARCANE_CHECK_MEMORY
  arcaneGlobalMemoryInfo()->setTraceMng(m_trace);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Application::
initialize()
{
  if (m_is_init){
    m_trace->warning() << "Application is already initialised";
    return;
  }

  bool is_debug = m_exe_info.isDebug();

  // Analyse le fichier de configuration de l'utilisateur
  _openUserConfig();

  //m_trace->info() << "Application init trace mng rank=" << m_parallel_super_mng->traceRank();
  m_trace_policy->setDefaultClassConfigXmlBuffer(userConfigBuffer());
  m_trace_policy->initializeTraceMng(m_trace.get(),m_parallel_super_mng->traceRank());

  m_trace->logdate() << "Begin execution.";

  if (is_debug)
    m_trace->info() << "WARNING: Execution in DEBUG mode!";

#ifdef ARCANE_CHECK
  m_trace->info() << "WARNING: Compilation in CHECK mode !";
#endif

  // Active ou désactive un mode vérification partiel si la variable d'environnement
  // correspondante est positionnée.
  String check_str = platform::getEnvironmentVariable("ARCANE_CHECK");
  if (!check_str.null()){
    bool is_check = check_str != "0";
    m_trace->info() << "WARNING: Setting CHECK mode to " << is_check;
    arcaneSetCheck(is_check);
  }
  if (arcaneIsCheck()){
    m_trace->info() << "WARNING: Execution in CHECK mode!";
  }

#ifdef ARCANE_TRACE
  m_trace->info() << "WARNING: Execution in TRACE mode !";
#endif
#ifdef ARCANE_64BIT
  m_trace->info() << "Using 64bits version!";
#endif

  m_trace->info() << "** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- **";
  m_trace->info() << "** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- **";
  m_trace->info() << " ";
  m_trace->info() << "               "
                  << m_exe_info.applicationName();
  m_trace->info() << " ";
  VersionInfo version_info = m_exe_info.codeVersion();
  m_trace->info() << "               Version "
                  << version_info.versionMajor() << "."
                  << version_info.versionMinor() << "."
                  << version_info.versionPatch();

  m_trace->info() << " ";
  m_trace->info() << "** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- **";
  m_trace->info() << "** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- ** -- **";
  m_trace->info() << "Date: " << platform::getCurrentDateTime();
  m_trace->info() << "MemoryUsage: " << static_cast<Int64>(platform::getMemoryUsed() / 1.0e6) << " (MegaBytes)";
  m_trace->info() << "ArccoreVersion: " << ARCCORE_VERSION_STR_FULL;
  m_trace->info() << "ArcaneVersion: " << ARCANE_VERSION_STR_FULL
                  << " (Compiler: " << platform::getCompilerId() << ")";
  m_trace->info() << "Current process pid=" << platform::getProcessId()
                  << " machine=" << platform::getHostName();
  m_trace->info() << "MessagePassing service=" << applicationBuildInfo().messagePassingService();

  if (platform::getStackTraceService()){
    m_trace->info() << "Stack trace service is available";
  }
  if (platform::getSymbolizerService()){
    m_trace->info() << "Symbolizer service is available";
  }

#ifdef ARCANE_USE_LIBXML2
    m_trace->info() << "Using 'libxml2' for XML parsing";
#endif

    // Affiche les infos sur les processeurs
  {
    IProcessorAffinityService* pas = platform::getProcessorAffinityService();
    if (pas){
      pas->printInfos();
    }
  }

  // Affiche si l'on a un service de debug
  if (platform::getOnlineDebuggerService()){
    m_trace->info() << "Hyoda service is now hooked";
  }
  else{
    m_trace->info() << "Unknown online debugger service";
  }
  
  m_is_init = true;

  // Analyse le fichier de configuration du code.
  _readCodeConfigurationFile();

  {
    // Construction des types internes
    ItemTypeMng::_singleton()->build(m_parallel_super_mng.get(),traceMng());
  }

  {
    ByteConstSpan runtime_config = m_exe_info.runtimeConfigFileContent();
    if (!runtime_config.empty()){
      m_trace->info() << "Reading configuration parameters from runtime config file";
      JSONDocument jdoc;
      jdoc.parse(runtime_config);
      JSONValue config = jdoc.root().child("configuration");
      ConfigurationReader cr(m_trace.get(),m_configuration_mng->defaultConfiguration());
      cr.addValuesFromJSON(config,ConfigurationReader::P_GlobalRuntime);
    }
  }
  {
    if (!m_config_root_element.null()){
      XmlNode configuration_elem = m_config_root_element.child("configuration");
      if (!configuration_elem.null()){
        m_trace->info() << "Reading configuration parameters from code config file";
        ConfigurationReader cr(m_trace.get(),m_configuration_mng->defaultConfiguration());
        cr.addValuesFromXmlNode(configuration_elem,ConfigurationReader::P_Global);
      }
    }
  }

  _initDataInitialisationPolicy();
  
  {
    if (!m_used_thread_service_name.null())
      m_trace->info() << "Service used for thread management : '" << m_used_thread_service_name << "'";
    else
      m_trace->info() << "No thread management active";

    if (!m_used_task_service_name.null()){
      m_trace->info() << "Service used for task management : '" << m_used_task_service_name
                      << "' (max_task_thread=" << TaskFactory::nbAllowedThread() << ")";
      std::ostringstream ostr;
      TaskFactory::printInfos(ostr);
      m_trace->info() << "TaskManagement infos:" << ostr.str();
    }
    else
      m_trace->info() << "No task management active";
  }

  // Recherche le service utilisé pour gérer le système d'unité.
  {
    ServiceBuilder<IPhysicalUnitSystemService> sf(this);
    String service_name = "Udunits";
    auto sv = sf.createReference(service_name,SB_AllowNull);
    if (sv.get()){
      m_trace->info() << "UnitSystem service found name=" << service_name;
    }
    else{
      m_trace->info() << "No unit system service found";
      sv = makeRef(createNullPhysicalUnitSystemService());
    }
    m_physical_unit_system_service = sv;
  }

  // Recherche le service utilisé pour gérer les compteurs de performance.
  {
    String service_name = "LinuxPerfPerformanceCounterService";
    String env_service_name = platform::getEnvironmentVariable("ARCANE_PERFORMANCE_COUNTER_SERVICE");
    if (!env_service_name.null())
      service_name = env_service_name + "PerformanceCounterService";
    ServiceBuilder<IPerformanceCounterService> sbuilder(this);
    auto p = sbuilder.createReference(service_name,SB_AllowNull);
    m_performance_counter_service = p;
    if (p.get()){
      m_trace->info() << "PerformanceCounterService found name=" << service_name;
    }
    else{
      m_trace->info() << "No performance counter service found";
    }
  }

  // Initialise le traceur des énumérateurs.
  {
    bool force_tracer = false;
    String trace_str = platform::getEnvironmentVariable("ARCANE_TRACE_ENUMERATOR");
    if (!trace_str.null() || ProfilingRegistry::profilingLevel()>=1 || force_tracer){
      if (!TaskFactory::isActive()){
        ServiceBuilder<IPerformanceCounterService> sbuilder(this);
        auto p = m_performance_counter_service;
        if (p.get()){
          m_trace->info() << "Enumerator tracing is enabled";
          Ref<IItemEnumeratorTracer> tracer(arcaneCreateItemEnumeratorTracer(traceMng(),p));
          arcaneSetSingletonItemEnumeratorTracer(tracer);
          p->initialize();
          p->start();
        }
        else
          m_trace->info() << "WARNING: enumerator tracing is not available because no performance counter service is available.";
      }
      else
        m_trace->info() << "WARNING: enumerator tracing is not available when using multi-tasking.";
    }
  }

  m_trace->info() << "sizeof(ItemInternal)=" << sizeof(ItemInternal)
                  << " sizeof(ItemInternalConnectivityList)=" << sizeof(ItemInternalConnectivityList)
                  << " sizeof(ItemSharedInfo)=" << sizeof(ItemSharedInfo);
  m_trace->info() << "sizeof(ItemLocalId)=" << sizeof(ItemLocalId)
                  << " sizeof(ItemConnectivityContainerView)=" << sizeof(ItemConnectivityContainerView)
                  << " sizeof(UnstructuredMeshConnectivityView)=" << sizeof(UnstructuredMeshConnectivityView);
  m_trace->info() << "sizeof(Item)=" << sizeof(Item)
                  << " sizeof(ItemEnumerator)=" << sizeof(ItemEnumerator)
                  << " sizeof(ItemVectorView)=" << sizeof(ItemVectorView)
                  << " sizeof(ItemVectorViewConstIterator)=" << sizeof(ItemVectorViewConstIterator)
                  << " ItemEnumeratorVersion=" << ItemEnumerator::version();
  m_trace->info() << "sizeof(eItemKind)=" << sizeof(eItemKind)
                  << " sizeof(IndexedItemConnectivityViewBase)=" << sizeof(IndexedItemConnectivityViewBase);

  {
    Real init_time_accelerator = ArcaneMain::initializationTimeForAccelerator() * 1000.0;
    if (init_time_accelerator!=0.0)
      m_trace->info() << "Time (in ms) to initialize Accelerators = " << init_time_accelerator;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Application::
_readCodeConfigurationFile()
{
  // Analyse le fichier de configuration du code.
  const ApplicationBuildInfo& build_info = applicationBuildInfo();

  // Récupère le nom du fichier de configuration.
  // Si nul, cela indique qu'il n'y a pas de fichier de configure.
  // Si vide (le défaut), on récupère le nom à partir du nom du code.
  // Sinon, on utilise le nom spécifié dans la configuration.

  // A noter que cette valeur doit être la même pour tous les PE sinon cela
  // va bloquer (TODO: faire une éventuellem réduction)
  String config_file_name = build_info.configFileName();

  bool use_config_file = true;
  if (config_file_name.null()){
    use_config_file = false;
  }
  else if (config_file_name.empty()){
    // Regarde d'abord dans le répertoire courant, sinon dans le répertoire
    // des données partagées (share).
    // Pour des raisons de performances en parallèle, seul le processeur maitre
    // fait le test.
    StringBuilder buf;
    if (m_is_master){
      buf = m_exe_info.codeName();
      buf += ".config";
      if (!platform::isFileReadable(buf.toString())){
        buf = m_exe_info.dataDir();
        buf += "/";
        buf += m_exe_info.codeName();
        buf += ".config";
      }
      else{
        m_trace->info() << "Using configuration file in current directory.";
      }
    }
    config_file_name = buf.toString();
  }
  m_trace->info() << "Using configuration file: '" << config_file_name << "'";

  if (use_config_file){
    bool bad_file = m_io_mng->collectiveRead(config_file_name,m_config_bytes);
    if (bad_file)
      ARCANE_FATAL("Can not read configuration file '{0}'",config_file_name);
    m_config_document = m_io_mng->parseXmlBuffer(m_config_bytes,config_file_name);
    if (!m_config_document.get())
      ARCANE_FATAL("Can not parse configuration file '{0}'",config_file_name);
    m_config_root_element = m_config_document->documentNode().documentElement();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Analyse le fichier de configuration de l'utilisateur.
 */
void Application::
_openUserConfig()
{
  Directory user_config_dir(m_user_config_path);
  String buf = user_config_dir.file("config.xml");

  //ByteUniqueArray bytes;
  bool bad_file = m_io_mng->collectiveRead(buf,m_user_config_bytes);
  if (bad_file){
    if (m_is_master)
      m_trace->log() << "No user configuration file '" << buf << "'";
    return;
  }

  IXmlDocumentHolder* doc = m_io_mng->parseXmlBuffer(m_user_config_bytes,buf);
  if (!doc){
    if (m_is_master)
      m_trace->log() << "Can not parse user configuration file '" << buf << "'";
    return;
  }

  m_user_config_document = doc;
  m_user_config_root_element = doc->documentNode().documentElement();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
bool _hasExtension(ICodeService* service,const String& extension)
{
  StringCollection extensions = service->validExtensions();
  for( StringCollection::Enumerator j(extensions); ++j; ){
    if ((*j)==extension)
      return true;
  }
  return false;
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ICodeService> Application::
getCodeService(const String& u_file_name)
{
  ServiceBuilder<ICodeService> builder(this);
  auto services = builder.createAllInstances();

  // Regarde si un service gere l'extension '.arc'.
  // S'il n'y en a pas, utilise ArcaneCodeService pour
  // cette extension.
  {
    bool has_arc_extension = false;
    for( Integer i=0, n=services.size(); i<n; ++i ){
      ICodeService* code_service = services[i].get();
      if (_hasExtension(code_service,"arc")){
        has_arc_extension = true;
        break;
      }
    }
    if (!has_arc_extension){
      services.add(createArcaneCodeService(this));
    }
  }

  // Cherche l'extension du fichier et la conserve dans \a case_ext
  std::string_view fview = u_file_name.toStdStringView();
  std::size_t extension_pos = fview.find_last_of('.');
  if (extension_pos==std::string_view::npos)
    return {};
  fview.remove_prefix(extension_pos+1);
  String case_ext(fview);

  Ref<ICodeService> found_service;
  for( const auto& code_service : services ){
    StringCollection extensions = code_service->validExtensions();
    for( StringCollection::Enumerator j(extensions); ++j; ){
      if (case_ext==(*j)){
        found_service = code_service;
        break;
      }
    }
    if (found_service.get())
      break;
  }
  // TODO: retourner une référence.
  return found_service;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Application::
addSession(ISession* session)
{
  m_sessions.add(session);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Application::
removeSession(ISession* session)
{
  m_sessions.remove(session);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceFactory2Collection Application::
serviceFactories2()
{
  return m_service_and_module_factory_mng->serviceFactories2();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ModuleFactoryInfoCollection Application::
moduleFactoryInfos()
{
  return m_service_and_module_factory_mng->moduleFactoryInfos();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Application::
_initDataInitialisationPolicy()
{
  String data_init_policy = platform::getEnvironmentVariable("ARCANE_DATA_INIT_POLICY");
  eDataInitialisationPolicy init_policy = getGlobalDataInitialisationPolicy();
  bool is_changed = false;
  if (data_init_policy=="DEFAULT"){
    init_policy = DIP_InitWithDefault;
    is_changed = true;
  }
  else if (data_init_policy=="NAN"){
    init_policy = DIP_InitWithNan;
    is_changed = true;
  }
  else if (data_init_policy=="NONE"){
    init_policy = DIP_None;
    is_changed = true;
  }
  else if (data_init_policy=="LEGACY"){
    init_policy = DIP_Legacy;
    is_changed = true;
  }
  else if (data_init_policy=="NAN_AND_DEFAULT"){
    init_policy = DIP_InitInitialWithNanResizeWithDefault;
    is_changed = true;
  }
  if (is_changed){
    setGlobalDataInitialisationPolicy(init_policy);
    init_policy = getGlobalDataInitialisationPolicy();
    m_trace->info() << "Change data initialisation policy: " << data_init_policy
                    << " (" << (int)init_policy << ")";
  }
  m_trace->info() << "Data initialisation policy is : " << (int)init_policy;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const ApplicationBuildInfo& Application::
applicationBuildInfo() const
{
  return m_arcane_main->applicationBuildInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const DotNetRuntimeInitialisationInfo& Application::
dotnetRuntimeInitialisationInfo() const
{
  return m_arcane_main->dotnetRuntimeInitialisationInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const AcceleratorRuntimeInitialisationInfo& Application::
acceleratorRuntimeInitialisationInfo() const
{
  return m_arcane_main->acceleratorRuntimeInitialisationInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITraceMng* Application::
createAndInitializeTraceMng(ITraceMng* parent_trace,const String& file_suffix)
{
  ITraceMng* tm = mainFactory()->createTraceMng();
  ITraceMngPolicy* tmp = getTraceMngPolicy();
  tmp->initializeTraceMng(tm,parent_trace,file_suffix);
  return tm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IDataFactory* Application::
dataFactory()
{
  return m_data_factory_mng->deprecatedOldFactory();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IDataFactoryMng* Application::
dataFactoryMng() const
{
  return m_data_factory_mng.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
