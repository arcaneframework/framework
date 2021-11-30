﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneMain.cc                                               (C) 2000-2021 */
/*                                                                           */
/* Classe gérant l'exécution.                                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/ArcaneMain.h"

#include "arcane/utils/Iostream.h"
#include "arcane/utils/StdHeader.h"
#include "arcane/utils/List.h"
#include "arcane/utils/Iterator.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ParallelFatalErrorException.h"
#include "arcane/utils/StringImpl.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/SignalException.h"
#include "arcane/utils/TimeoutException.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ArithmeticException.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/IDynamicLibraryLoader.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/CommandLineArguments.h"
#include "arcane/utils/ApplicationInfo.h"

#include "arcane/ArcaneException.h"
#include "arcane/IMainFactory.h"
#include "arcane/IApplication.h"
#include "arcane/IServiceLoader.h"
#include "arcane/IParallelMng.h"
#include "arcane/IParallelSuperMng.h"
#include "arcane/IIOMng.h"
#include "arcane/ISession.h"
#include "arcane/ISubDomain.h"
#include "arcane/IRessourceMng.h"
#include "arcane/IModuleMng.h"
#include "arcane/IModule.h"
#include "arcane/IVariableMng.h"
#include "arcane/VariableRef.h"
#include "arcane/ITimeLoopMng.h"
#include "arcane/ITimeLoop.h"
#include "arcane/Directory.h"
#include "arcane/XmlNodeList.h"
#include "arcane/IXmlDocumentHolder.h"
#include "arcane/ItemTypeMng.h"
#include "arcane/ServiceUtils.h"
#include "arcane/ICodeService.h"
#include "arcane/CaseOptions.h"
#include "arcane/VariableCollection.h"
#include "arcane/ItemGroupImpl.h"
#include "arcane/SubDomainBuildInfo.h"
#include "arcane/ICaseMng.h"
#include "arcane/DotNetRuntimeInitialisationInfo.h"
#include "arcane/AcceleratorRuntimeInitialisationInfo.h"
#include "arcane/ApplicationBuildInfo.h"

#include "arcane/IServiceFactory.h"
#include "arcane/IModuleFactory.h"

#include "arcane/impl/TimeLoopReader.h"
#include "arcane/impl/MainFactory.h"
#include "arcane/impl/InternalInfosDumper.h"
#include "arcane/impl/internal/ArcaneMainExecInfo.h"
#include "arcane/impl/internal/ThreadBindingMng.h"

#include <signal.h>
#include <exception>
#ifndef ARCANE_OS_WIN32
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

#include <set>
#include <map>

#ifdef ARCANE_FLEXLM
#include "arcane/impl/FlexLMTools.h"
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace Arcane
{
class ArcaneMainStaticInfo
{
 public:
  List<IServiceFactoryInfo*> m_service_factory_infos;
  List<IModuleFactoryInfo*> m_module_factory_infos;
  List<IApplicationBuildInfoVisitor*> m_application_build_info_visitors;
  ApplicationBuildInfo m_app_build_info;
  DotNetRuntimeInitialisationInfo m_dotnet_init_info;
  AcceleratorRuntimeInitialisationInfo m_accelerator_init_info;
  bool m_has_dotnet_wrapper = false;
  String m_dotnet_assembly;
  String m_arcane_lib_path;
  IDirectSubDomainExecuteFunctor* m_direct_exec_functor = nullptr;
  std::atomic<Int32> m_nb_autodetect = 0;
};
}

namespace
{
Arcane::ArcaneMainStaticInfo* global_static_info = nullptr;
Arcane::ArcaneMainStaticInfo* _staticInfo()
{
  // TODO: voir s'il faut protéger en multi-threading.
  if (!global_static_info)
    global_static_info = new Arcane::ArcaneMainStaticInfo();
  return global_static_info;
}
void _deleteStaticInfo()
{
  delete global_static_info;
  global_static_info = nullptr;
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C"
void arcaneEndProgram()
{
  // Juste la pour faire un point d'entrée à third.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C"
{
  typedef void (*fSignalFunc)(int);
  void arcaneSignalHandler(int);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void
arcaneRedirectSignals(fSignalFunc sig_func);

extern "C++" ARCANE_UTILS_EXPORT void
arcaneCallDefaultSignal(int val);

extern "C++" ARCANE_UTILS_EXPORT void
initializeStringConverter();

extern "C++" ARCANE_IMPL_EXPORT IArcaneMain*
createArcaneMainBatch(const ApplicationInfo& exe_info,IMainFactory*);

extern "C++" ARCANE_IMPL_EXPORT IDynamicLibraryLoader*
createGlibDynamicLibraryLoader();

extern "C++" ARCANE_IMPL_EXPORT ICodeService*
createArcaneCodeService(IApplication* app);

Int32 ArcaneMain::m_nb_arcane_init = 0;
Int32 ArcaneMain::m_is_init_done = 0;
bool ArcaneMain::m_has_garbage_collector = false;
bool ArcaneMain::m_is_master_io = true;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ArcaneBatchMainFactory
: public MainFactory
{
 public:

  IArcaneMain* createArcaneMain(const ApplicationInfo& app_info) override
  { 
    return createArcaneMainBatch(app_info,this);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static int
_catchStd(const std::exception& ex,ITraceMng* msg)
{
  if (msg){
    msg->error() << "** A standard exception occured: " << ex.what() << ".\n"
                 << "** Can't continue with the execution.\n";
  }
  else{
    cerr << "** A standard exception occured: " << ex.what() << ".\n";
    cerr << "** Can't continue with the execution.\n";
  }
  return 2;
}

static int
_catchAny(ITraceMng* msg)
{
  int ret_val = 1;
  const char* msg_str = "** An unknowed error occured...\n** Can't continue with the execution.\n";
  if (msg){
    msg->error() << msg_str;
  }
  else{
    cerr << msg_str;
  }
  return ret_val;
}

static int
_catchArcane(const Exception& ex,ITraceMng* msg)
{
  if (msg){
    if (!ex.isCollective() || msg->isMaster())
      msg->error() << ex << '\n'
                   << "** (MAIN) Can't continue with the execution.\n";
  }
  else{
    cerr << ex << '\n';
    cerr << "** (MAIN) Can't continue with the execution.\n";
  }
  return 3;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ArcaneMain::Impl
{
 public:
  Impl(const ApplicationInfo& infos)
  : m_app_info(infos),
    m_application_build_info(ArcaneMain::defaultApplicationBuildInfo()),
    m_dotnet_info(ArcaneMain::defaultDotNetRuntimeInitialisationInfo()),
    m_accelerator_info(ArcaneMain::defaultAcceleratorRuntimeInitialisationInfo())
  {}
  Impl(const ApplicationInfo& infos,const ApplicationBuildInfo& build_infos,
       const DotNetRuntimeInitialisationInfo& dotnet_info,
       const AcceleratorRuntimeInitialisationInfo& accelerator_info)
  : m_app_info(infos), m_application_build_info(build_infos),
    m_dotnet_info(dotnet_info), m_accelerator_info(accelerator_info)
  {}
 public:
  ApplicationInfo m_app_info;
  ApplicationBuildInfo m_application_build_info;
  DotNetRuntimeInitialisationInfo m_dotnet_info;
  AcceleratorRuntimeInitialisationInfo m_accelerator_info;
  ThreadBindingMng m_thread_binding_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMain::
redirectSignals()
{
  bool redirect_signals = true;
  String rv = platform::getEnvironmentVariable("ARCANE_REDIRECT_SIGNALS");
  (void)builtInGetValue(redirect_signals,rv);
  if (redirect_signals){
    arcaneRedirectSignals(arcaneSignalHandler);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMain::
setDefaultMainFactory(IMainFactory* mf)
{
  m_default_main_factory = mf;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ArcaneMainExecFunctor
: public IFunctor
{
 public:
  ArcaneMainExecFunctor(const ApplicationInfo& app_info,IArcaneMain* exec_main)
  : m_app_info(app_info),
    m_exec_main(exec_main)
  {
  }
 public:
  void executeFunctor() override
  {
    StringList args;
    m_app_info.args(args);
    if (!m_exec_main->parseArgs(args))
      m_exec_main->execute();
  }
 private:
  const ApplicationInfo& m_app_info;
  IArcaneMain* m_exec_main;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour gérer les appels aux mécanismes d'auto-détection des runtimes (MPI,Accélérateurs).
 *
 * Cette classe permet de garantir que les mécanismes d'auto-détection ne sont
 * appelés qu'une fois. L'auto-détection se fait lors de l'appel à check().
 */
class ArcaneMainAutoDetectRuntimeHelper
{
 public:
  Int32 check()
  {
    auto* x = _staticInfo();
    if (x->m_nb_autodetect>0)
      return m_return_value;

    // TODO: rendre thread-safe
    {
      ArcaneMain::_checkAutoDetectMPI();

      m_return_value = ArcaneMain::_checkAutoDetectAccelerator();
      ++x->m_nb_autodetect;
    }
    return m_return_value;
  }
 public:
  Int32 m_return_value = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Création de l'instance 'IArcaneMain'.
 *
 * Si l'instance est déjà créée, cette méthode ne fait rien.
 *
 * En cas d'exception, cette méthode retourne une valeur non nulle.
 * En cas d'erreur dans le gestionnaire de licence, \a m_ret_val est
 * positionné à une valeur non nulle mais cette méthode retourne 0.
 */
int ArcaneMainExecInfo::
initialize()
{
  if (m_exec_main)
    return 0;

  m_ret_val = 0;
  m_clean_abort = false;

  ArcaneMain::redirectSignals();

  // Création de la classe d'exécution
  try{
    if (m_has_build_info){
      ArcaneMain* x = new ArcaneMain(m_app_info,m_main_factory,
                                     m_application_build_info,
                                     ArcaneMain::defaultDotNetRuntimeInitialisationInfo(),
                                     ArcaneMain::defaultAcceleratorRuntimeInitialisationInfo());
      m_exec_main = x;
    }
    else{
      m_exec_main = m_main_factory->createArcaneMain(m_app_info);
    }
    m_exec_main->build();
    ArcaneMain::m_is_master_io = m_exec_main->application()->parallelSuperMng()->isMasterIO();
    m_exec_main->initialize();
#ifndef ARCANE_USE_MPC
    IArcaneMain::setArcaneMain(m_exec_main);
#endif
  }
  catch(const ArithmeticException& ex){
    cerr << "** CATCH ARITHMETIC_EXCEPTION\n";
    return _catchArcane(ex,nullptr);
  }
  catch(const Exception& ex){
    return _catchArcane(ex,nullptr);
  }
  catch(const std::exception& ex){
    return _catchStd(ex,nullptr);
  }
  catch(...){
    return _catchAny(nullptr);
  }

  // Redirige a nouveau les signaux car certaines
  // routines d'initialisation (par exemple MPI) peuvent
  // les detourner.
  ArcaneMain::redirectSignals();

  m_ret_val = 0;
  m_clean_abort = false;

#ifdef ARCANE_FLEXLM
  try {
    IApplication* app = m_exec_main->application();
    ITraceMng* trace = app->traceMng();
    IParallelSuperMng * parallel_super_mng = app->parallelSuperMng();
    trace->info() << "Initializing license manager";
    FlexLMMng::instance()->init(parallel_super_mng);

    // La politique de licence parallèle est sous-traitée aux applications
    //     bool is_parallel = parallel_super_mng->isParallel();
    //     FlexLMTools<ArcaneFeatureModel> license_tool;
    //     Integer commSize = parallel_super_mng->commSize();
    //     if (is_parallel && commSize > 1)
    //       { // La fonctionnalité paralléle n'est activé que si nécessaire
    //         license_tool.getLicense(ArcaneFeatureModel::ArcaneParallel,commSize);
    //       }
  }
  catch(const Exception& ex){
    IApplication* app = m_exec_main->application();
    ITraceMng* trace = app->traceMng();
    m_ret_val = _catchArcane(ex,trace);
    if (ex.isCollective()){
      m_clean_abort = true;
    }
  }
#endif
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// NOTE: Cette méthode ne doit pas lancer d'exceptions
void ArcaneMainExecInfo::
execute()
{
  if (m_ret_val!=0)
    return;

  if (m_direct_exec_functor)
    m_exec_main->setDirectExecuteFunctor(m_direct_exec_functor);

  ArcaneMainExecFunctor exec_functor(m_app_info,m_exec_main);
  if (ArcaneMain::m_exec_override_functor){
    // Obsolète. Ne plus utiliser.
    IApplication* app = m_exec_main->application();
    ArcaneMain::m_exec_override_functor->m_application = app;
    ITraceMng* trace = app->traceMng();
    trace->info() << "Calling overriding functor";
    m_ret_val = ArcaneMain::callFunctorWithCatchedException(ArcaneMain::m_exec_override_functor->functor(),
                                                            m_exec_main,&m_clean_abort,true);
  }
  else
    m_ret_val = ArcaneMain::callFunctorWithCatchedException(&exec_functor,m_exec_main,&m_clean_abort,true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMainExecInfo::
finalize()
{
  // Si l'exécution s'est bien déroulée mais que l'utilisateur a spécifié un
  // code d'erreur, on le récupère.
  int exe_error_code = m_exec_main->errorCode();
  if (m_ret_val==0 && exe_error_code!=0){
    m_ret_val = exe_error_code;
  }
  else
    if (m_ret_val!=0)
      m_exec_main->setErrorCode(m_ret_val);

  m_exec_main->finalize();

  if (m_ret_val!=0 && !m_clean_abort)
    m_exec_main->doAbort();

  // Destruction du code.
  // Attention à ne pas détruire le gestionnaire avant car lorsqu'une exception
  // de l'architecture est générée, elle utilise un ITraceMng pour afficher
  // le message
  delete m_exec_main;
  m_exec_main = nullptr;
#ifndef ARCANE_USE_MPC
  IArcaneMain::setArcaneMain(m_exec_main);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Point d'entrée de l'exécutable.
 * \internal
 */
int ArcaneMain::
_arcaneMain(const ApplicationInfo& app_info,IMainFactory* factory)
{
  if (!factory)
    return 5;

  ArcaneMainExecInfo exec_info(app_info,factory);
  int r = exec_info.initialize();
  if (r!=0)
    return r;

  IDirectSubDomainExecuteFunctor* func = _staticInfo()->m_direct_exec_functor;
  if (func)
    exec_info.setDirectExecFunctor(func);
  exec_info.execute();
  exec_info.finalize();

  return exec_info.returnValue();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int ArcaneMain::
callFunctorWithCatchedException(IFunctor* functor,IArcaneMain* exec_main,
                                bool* clean_abort,bool is_print)
{
  int ret_val = 0;
  *clean_abort = false;
  IApplication* app = exec_main->application();
  ITraceMng* trace = app->traceMng();
  bool is_parallel = app->parallelSuperMng()->isParallel();
  bool is_master = app->parallelSuperMng()->isMasterIO();
  try{
    functor->executeFunctor();
  }
  catch(const FatalErrorException& ex)
  {
    if (ex.isCollective()){
      if (is_parallel){
        *clean_abort = true;
        ret_val = 5;
        if (is_master && is_print){
          ofstream ofile("fatal");
          ofile << ret_val << '\n';
          ofile.flush();
          trace->error() << "ParallelFatalErrorException caught in ArcaneMain::callFunctor: " << ex << '\n';
        }
      }
      else{
        trace->error() << "ParallelFatalErrorException caught in ArcaneMain::callFunctor: " << ex << '\n';
        ret_val = 4;
      }
    }
    else{
      trace->error() << Trace::Color::red() << "FatalErrorException caught in ArcaneMain::callFunctor: " << ex << '\n';
      ret_val = 4;
    }
  }
  catch(const SignalException& ex)
    {
      trace->error() << "SignalException caught in ArcaneMain::callFunctor: " << ex << '\n';
      ret_val = 6;
    }
  catch(const TimeoutException& ex)
    {
      trace->error() << "TimeoutException caught in ArcaneMain::callFunctor: " << ex << '\n';
      ret_val = 7;
    }
  catch(const ParallelFatalErrorException& ex)
    {
      // TODO: utiliser le code de FatalErrorException en mode collectif.n
      if (is_parallel){
        *clean_abort = true;
        ret_val = 5;
        if (is_master && is_print){
          ofstream ofile("fatal");
          ofile << ret_val << '\n';
          ofile.flush();
          trace->error() << "ParallelFatalErrorException caught in ArcaneMain::callFunctor: " << ex << '\n';
        }
      }
      else{
        trace->error() << "ParallelFatalErrorException caught in ArcaneMain::callFunctor: " << ex << '\n';
        ret_val = 4;
      }
    }
  catch(const ArithmeticException& ex){
    cerr << "** ARITHMETIC EXCEPTION!\n";
    ret_val = _catchArcane(ex,trace);
    if (ex.isCollective()){
      *clean_abort = true;
    }
  }
  catch(const Exception& ex){
    ret_val = _catchArcane(ex,trace);
    if (ex.isCollective()){
      *clean_abort = true;
    }
  }
  catch(const std::exception& ex){
    ret_val = _catchStd(ex,trace);
  }
  catch(...){
    ret_val = _catchAny(trace);
  }
  return ret_val;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMain::
_launchMissingInitException()
{
  cerr << "ArcaneMain: appel ArcaneMain::arcaneInitialize() manquant.\n";
  throw std::exception();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMain::
_checkHasInit()
{
  if (m_nb_arcane_init<=0)
    _launchMissingInitException();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void arcaneInitCheckMemory();
extern "C++" void arcaneExitCheckMemory();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMain::
setHasGarbageCollector()
{
  if (m_nb_arcane_init!=0){
    cerr << "WARNING: ArcaneMain::setHasGarbageCollector has to be called before arcaneInitialize\n";
    return;
  }
  m_has_garbage_collector = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMain::
setHasDotNETRuntime()
{
  if (m_nb_arcane_init!=0){
    cerr << "WARNING: ArcaneMain::setHasDotNETRuntime has to be called before arcaneInitialize\n";
    return;
  }
  platform::setHasDotNETRuntime(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMain::
setExecuteOverrideFunctor(ArcaneMainExecutionOverrideFunctor* functor)
{
  m_exec_override_functor = functor;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ArcaneMain::
hasDotNetWrapper()
{
  return _staticInfo()->m_has_dotnet_wrapper;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * \brief Positionne le chemin contenant les bibliothèques dynamiques de Arcane.
 *
 * Il ne faut appeler qu'une seule fois cette méthode.
 */
void ArcaneMain::
_setArcaneLibraryPath()
{
  String dir_name;
  String dll_full_path = platform::getLoadedSharedLibraryFullPath("arcane_impl");
  if (!dll_full_path.null())
    dir_name = platform::getFileDirName(dll_full_path);
  if (dir_name.null())
    dir_name = platform::getCurrentDirectory();
  _staticInfo()->m_arcane_lib_path = dir_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMain::
_checkCreateDynamicLibraryLoader()
{
  auto x = platform::getDynamicLibraryLoader();
  if (!x){
    platform::setDynamicLibraryLoader(createGlibDynamicLibraryLoader());
    _setArcaneLibraryPath();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMain::
arcaneInitialize()
{
  // Le premier thread qui arrive ici fait l'init.
  // Les autres doivent attendre que l'init soit terminée.
  if (AtomicInt32::increment(&m_nb_arcane_init)==1){
    (void)_staticInfo();
    Exception::staticInit();
    dom::DOMImplementation::initialize();
    platform::platformInitialize();
    // Crée le singleton gestionnaire des types
    ItemTypeMng::_singleton();
    initializeStringConverter();
    arcaneInitCheckMemory();
    // Initialise le singleton du groupe vide et garde une référence dessus.
    ItemGroupImpl::_buildSharedNull();
    _checkCreateDynamicLibraryLoader();
    AtomicInt32::setValue(&m_is_init_done,1);
  }
  else
    // Attend que le thread qui fait l'init ait terminé
    while (AtomicInt32::getValue(&m_is_init_done)==0)
      ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMain::
arcaneFinalize()
{
  _checkHasInit();

  if (AtomicInt32::decrement(&m_nb_arcane_init)==0){
    _deleteStaticInfo();

    //! Supprime notre référence sur ItemGroupImpl::shared_null.
    ItemGroupImpl::_destroySharedNull();

    {
      auto x = platform::getDynamicLibraryLoader();
      if (x){
        x->closeLibraries();
        delete x;
      }
    }
    arcaneExitCheckMemory();
    platform::platformTerminate();
    dom::DOMImplementation::terminate();
    ItemTypeMng::_destroySingleton();
    arcaneEndProgram();
#ifdef ARCANE_FLEXLM
    {
      FlexLMMng::instance()->releaseAllLicenses();
    }
#endif
    AtomicInt32::setValue(&m_is_init_done,0);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMain::
addServiceFactoryInfo(IServiceFactoryInfo* sri)
{
  _staticInfo()->m_service_factory_infos.add(sri);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMain::
addModuleFactoryInfo(IModuleFactoryInfo* mfi)
{
  _staticInfo()->m_module_factory_infos.add(mfi);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMain::
addApplicationBuildInfoVisitor(IApplicationBuildInfoVisitor* visitor)
{
  _staticInfo()->m_application_build_info_visitors.add(visitor);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ApplicationInfo& ArcaneMain::
defaultApplicationInfo()
{
  return _staticInfo()->m_app_build_info._internalApplicationInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DotNetRuntimeInitialisationInfo& ArcaneMain::
defaultDotNetRuntimeInitialisationInfo()
{
  return _staticInfo()->m_dotnet_init_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorRuntimeInitialisationInfo& ArcaneMain::
defaultAcceleratorRuntimeInitialisationInfo()
{
  return _staticInfo()->m_accelerator_init_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ApplicationBuildInfo& ArcaneMain::
defaultApplicationBuildInfo()
{
  return _staticInfo()->m_app_build_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const ApplicationBuildInfo& ArcaneMain::
applicationBuildInfo() const
{
  return m_p->m_application_build_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ApplicationBuildInfo& ArcaneMain::
_applicationBuildInfo()
{
  return m_p->m_application_build_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const DotNetRuntimeInitialisationInfo& ArcaneMain::
dotnetRuntimeInitialisationInfo() const
{
  return m_p->m_dotnet_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const AcceleratorRuntimeInitialisationInfo& ArcaneMain::
acceleratorRuntimeInitialisationInfo() const
{
  return m_p->m_accelerator_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int ArcaneMain::
arcaneMain(const ApplicationInfo& app_info,IMainFactory* factory)
{
  _checkHasInit();

  ScopedPtrT<IMainFactory> default_factory;
  if (!factory){
    factory = m_default_main_factory;
    if (!factory){
      factory = new ArcaneBatchMainFactory();
    }
    default_factory = factory;
  }

  int ret = _arcaneMain(app_info,factory);

  default_factory = nullptr;
  
  // Le code d'erreur 5 représente une erreur en parallèle pour tous les
  // processeurs.
  if (ret!=0 && ret!=5)
    cerr << "* Process return: " << ret << '\n';
  if (ret==5)
    ret = 4;
    
  return ret;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int ArcaneMain::
_internalRun(IDirectSubDomainExecuteFunctor* func)
{
  _staticInfo()->m_direct_exec_functor = func;
  return run();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int ArcaneMain::
_initRuntimes()
{
  ArcaneMainAutoDetectRuntimeHelper auto_detect_helper;
  return auto_detect_helper.check();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int ArcaneMain::
run()
{
  int r = _initRuntimes();
  if (r!=0)
    return r;

  DotNetRuntimeInitialisationInfo& dotnet_info = defaultDotNetRuntimeInitialisationInfo();

  // Si on arrive ici et que le runtime C# est déjà chargé
  // (parce que le Main est en C# par exemple), on ne lance pas le wrapper
  bool is_in_dotnet = platform::hasDotNETRuntime();
  if (!is_in_dotnet && dotnet_info.isUsingDotNetRuntime()){
    r = _runDotNet();
  }
  else{
    arcaneInitialize();
    r = arcaneMain(defaultApplicationInfo(),nullptr);
    arcaneFinalize();
  }
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int ArcaneMain::
_runDotNet()
{
  auto si = _staticInfo();
  si->m_has_dotnet_wrapper = true;

  // TODO: vérifier que l'init n'a pas été faite.

  // Pour pouvoir exécuter le code .Net, il faut
  // appeler la méthode 'arcane_mono_main' qui se trouve
  // dans la bibliothèque dynamique 'arcane_mono'.

  typedef int (*DotNetMainFunctor)(const CommandLineArguments& cmd_args,const String& assembly_name);

  const ApplicationInfo& app_info = defaultApplicationInfo();
  const DotNetRuntimeInitialisationInfo& dotnet_info = defaultDotNetRuntimeInitialisationInfo();

  DotNetMainFunctor my_functor = nullptr;
  String os_dir(si->m_arcane_lib_path);

  try{
    _checkCreateDynamicLibraryLoader();

    IDynamicLibraryLoader* dll_loader = platform::getDynamicLibraryLoader();
    if (!dll_loader)
      ARCANE_FATAL("No dynamic library available for running .Net");

    String dll_name = "arcane_mono";
    String symbol_name = "arcane_mono_main2";

    String runtime_name = dotnet_info.embeddedRuntime();

    if (runtime_name.null() || runtime_name=="mono")
      // Mono est le défaut si rien n'est spécifié.
      ;
    else if (runtime_name=="coreclr"){
      dll_name = "arcane_dotnet_coreclr";
      symbol_name = "arcane_dotnet_coreclr_main";
    }
    else
      ARCANE_FATAL("Unknown '.Net' runtime '{0}'. Valid values are 'mono' or 'coreclr'",runtime_name);

    IDynamicLibrary* dl = dll_loader->open(os_dir,dll_name);
    if (!dl)
      ARCANE_FATAL("Can not found dynamic library '{0}' for using .Net",dll_name);

    bool is_found = false;
    void* functor_addr = dl->getSymbolAddress(symbol_name,&is_found);
    if (!is_found)
      ARCANE_FATAL("Can not find symbol '{0}' in library '{1}'",symbol_name,dll_name);

    my_functor = reinterpret_cast<DotNetMainFunctor>(functor_addr);
  }
  catch(const Exception& ex){
    return _catchArcane(ex,nullptr);
  }
  catch(const std::exception& ex){
    return _catchStd(ex,nullptr);
  }
  catch(...){
    return _catchAny(nullptr);
  }

  if (my_functor){
    const CommandLineArguments& cmd_args = app_info.commandLineArguments();
    // TODO: vérifier que l'assembly 'Arcane.Main.dll' existe bien.
    String new_name = os_dir + "/Arcane.Main.dll";
    return (*my_functor)(cmd_args,new_name);
  }
  return (-1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Ne pas appeler directement mais passer par ArcaneMainAutoDetectHelper.
void ArcaneMain::
_checkAutoDetectMPI()
{
  auto si = _staticInfo();

  // Pour pouvoir automatiquement enregisrer MPI, il faut
  // appeler la méthode 'arcaneAutoDetectMessagePassingServiceMPI' qui se trouve
  // dans la bibliothèque dynamique 'arcane_mpi'.

  typedef void (*ArcaneAutoDetectMPIFunctor)();

  _checkCreateDynamicLibraryLoader();

  IDynamicLibraryLoader* dll_loader = platform::getDynamicLibraryLoader();
  if (!dll_loader)
    return;

  String os_dir(si->m_arcane_lib_path);
  String dll_name = "arcane_mpi";
  String symbol_name = "arcaneAutoDetectMessagePassingServiceMPI";
  IDynamicLibrary* dl = dll_loader->open(os_dir,dll_name);
  if (!dl)
    return;

  bool is_found = false;
  void* functor_addr = dl->getSymbolAddress(symbol_name,&is_found);
  if (!is_found)
    return;

  auto my_functor = reinterpret_cast<ArcaneAutoDetectMPIFunctor>(functor_addr);
  if (my_functor)
    (*my_functor)();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Détecte et charge la gestion du runtime des accélérateurs.
 *
 * \note Ne pas appeler directement mais passer par ArcaneMainAutoDetectHelper.
 */
int ArcaneMain::
_checkAutoDetectAccelerator()
{
  auto si = _staticInfo();
  AcceleratorRuntimeInitialisationInfo& init_info = si->m_accelerator_init_info;
  if (!init_info.isUsingAcceleratorRuntime())
    return 0;
  String runtime_name = init_info.acceleratorRuntime();
  //std::cout << "RUNTIME=" << runtime_name << "\n";
  if (runtime_name.empty())
    return 0;

  try{
    // Pour l'instant, seul le runtime 'cuda' est autorisé
    if (runtime_name!="cuda" && runtime_name!="hip")
      ARCANE_FATAL("Invalid accelerator runtime '{0}'. Only 'cuda' or 'hip' is allowed",runtime_name);

    // Pour pouvoir automatiquement enregisrer un runtime accélérateur de nom \a NAME,
    // il faut appeler la méthode 'arcaneRegisterAcceleratorRuntime${NAME}' qui se trouve
    // dans la bibliothèque dynamique 'arcane_${NAME}'.

    typedef void (*ArcaneAutoDetectAcceleratorFunctor)();

    _checkCreateDynamicLibraryLoader();

    IDynamicLibraryLoader* dll_loader = platform::getDynamicLibraryLoader();
    if (!dll_loader)
      ARCANE_FATAL("No dynamic library available for running accelerator runtime");

    String os_dir(si->m_arcane_lib_path);
    String dll_name = "arcane_accelerator_" + runtime_name + "_runtime";
    String symbol_name = "arcaneRegisterAcceleratorRuntime" + runtime_name;
    IDynamicLibrary* dl = dll_loader->open(os_dir,dll_name);
    if (!dl)
      ARCANE_FATAL("Can not found dynamic library '{0}' for using accelerator runtime",dll_name);

    bool is_found = false;
    void* functor_addr = dl->getSymbolAddress(symbol_name,&is_found);
    if (!is_found || !functor_addr)
      ARCANE_FATAL("Can not find symbol '{0}' in library '{1}'",symbol_name,dll_name);

    auto my_functor = reinterpret_cast<ArcaneAutoDetectAcceleratorFunctor>(functor_addr);
    (*my_functor)();
  }
  catch(const Exception& ex){
    return _catchArcane(ex,nullptr);
  }
  catch(const std::exception& ex){
    return _catchStd(ex,nullptr);
  }
  catch(...){
    return _catchAny(nullptr);
  }
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMainFactory* ArcaneMain::m_default_main_factory = nullptr;
ArcaneMainExecutionOverrideFunctor* ArcaneMain::m_exec_override_functor = nullptr;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneMain::
ArcaneMain(const ApplicationInfo& app_info,IMainFactory* factory)
: m_p(new Impl(app_info))
, m_main_factory(factory)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneMain::
ArcaneMain(const ApplicationInfo& app_info,IMainFactory* factory,
           const ApplicationBuildInfo& app_build_info,
           const DotNetRuntimeInitialisationInfo& dotnet_info,
           const AcceleratorRuntimeInitialisationInfo& accelerator_info)
: m_p(new Impl(app_info,app_build_info,dotnet_info,accelerator_info))
, m_main_factory(factory)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneMain::
~ArcaneMain()
{
  delete m_application;
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMain::
build()
{
  _parseApplicationBuildInfoArgs();
  m_application = m_main_factory->createApplication(this);
  m_p->m_thread_binding_mng.initialize(m_application->traceMng(),
                                       m_p->m_application_build_info.threadBindingStrategy());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const ApplicationInfo& ArcaneMain::
applicationInfo() const
{
  return m_p->m_app_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMain::
_parseApplicationBuildInfoArgs()
{
  ApplicationBuildInfo& abi = _applicationBuildInfo();
  abi.parseArguments(m_p->m_app_info.commandLineArguments());
  // Appelle les visiteurs enregistrés.
  {
    auto& x = _staticInfo()->m_application_build_info_visitors;
    for( IApplicationBuildInfoVisitor* v : x){
      if (v)
        v->visit(abi);
    }
  }
  abi.setDefaultServices();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMain::
initialize()
{
  m_application->initialize();

  ScopedPtrT<IServiceLoader> service_loader(m_main_factory->createServiceLoader());
  service_loader->loadApplicationServices(m_application);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ServiceFactoryInfoCollection ArcaneMain::
registeredServiceFactoryInfos()
{
  return _staticInfo()->m_service_factory_infos;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ModuleFactoryInfoCollection ArcaneMain::
registeredModuleFactoryInfos()
{
  return _staticInfo()->m_module_factory_infos;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int ArcaneMain::
execute()
{
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMain::
doAbort()
{
  ::abort();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMain::
setErrorCode(int errcode)
{
  m_error_code = errcode;
  if (errcode!=0){
    // Seul le proc maitre écrit le fichier sauf s'il s'agit d'un fatal car
    // dans ce cas n'importe quel PE peut le faire.
    if (ArcaneMain::m_is_master_io || errcode==4){
      String errname = "fatal_" + String::fromNumber(errcode);
      ofstream ofile(errname.localstr());
      ofile.close();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ArcaneMain::
parseArgs(StringList args)
{
  // Si vrai, affichage des informations internes
  bool arcane_internal = false;
  // Si vrai, affichage des informations internes pour chaque boucle en temps
  bool arcane_all_internal = false;
  // Si vrai, génère un fichier contenant les infos internes du code.
  bool arcane_database = false;
  // Si vrai, affichage des informations sur les modules et services présents et une aide sommaire
  bool arcane_help = false;

  String us_arcane_opt("-arcane_opt");
  String us_help("help");
  String us_arcane_internal("arcane_internal");
  String us_arcane_all_internal("arcane_all_internal");
  String us_arcane_database("arcane_database");

  StringList unknown_args;
  for( Integer i=0, s=args.count(); i<s; ++i ){
    if (args[i].startsWith("-A")){
      continue;
    }
    if (args[i]!=us_arcane_opt){
      unknown_args.add(args[i]);
      continue;
    }
    bool is_valid_opt = false;
    ++i;
    String str;
    if (i<s)
      str = args[i];
    if (str==us_arcane_internal){
      arcane_internal = true;
      is_valid_opt = true;
    }
    if (str==us_arcane_all_internal){
      arcane_all_internal = true;
      is_valid_opt = true;
    }
    if (str==us_arcane_database){
      arcane_database = true;
      is_valid_opt = true;
    }
    if (str==us_help){
      arcane_help = true;
      is_valid_opt = true;
    }
    if (!is_valid_opt){
      // Si l'option n'est pas valide, la rajoute à la liste des options
      // non traitées
      unknown_args.add(us_arcane_opt);
      if (!str.null())
        unknown_args.add(str);
      //trace->fatal() << "Unknown arcane option <" << str << ">\n";
    }
  }
  
  bool do_stop = false;
  if (arcane_database){
    InternalInfosDumper dumper(application());
    dumper.dumpArcaneDatabase();
    do_stop = true;
  }
  if (arcane_internal){
    InternalInfosDumper dumper(application());
    dumper.dumpInternalInfos();
    do_stop = true;
  }
  if (arcane_all_internal){
    InternalInfosDumper dumper(application());
    dumper.dumpInternalAllInfos();
    do_stop = true;
  }
  if (arcane_help){
    _dumpHelp();
    do_stop = true;
  }

  args.clear();
  for( StringList::Enumerator i(unknown_args); ++i; )
    args.add(*i);
  
  return do_stop;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMain::
_dumpHelp()
{
  // Utilise un multimap car plusieurs services peuvent avoir le même nom.
  typedef std::multimap<String,IServiceInfo*> ServiceList;
  ServiceList service_list;
  Integer max_name_size = 0;

  {
    // On veut lister par IServiceInfo. Comme il est possible qu'un service ait plusieurs
    // fabriques, on filtre grâce à done_set.
    std::set<IServiceInfo*> done_set;
    for( ServiceFactory2Collection::Enumerator j(application()->serviceFactories2()); ++j; ){
      IServiceInfo* si = (*j)->serviceInfo();
      if (done_set.find(si)!=done_set.end()){
        continue;
      }
      done_set.insert(si);
      const String& name = si->localName();
      max_name_size = math::max(max_name_size,CheckedConvert::toInteger(name.length()));
      service_list.insert(std::make_pair(name,si));
    }
  }

  UniqueArray<String> module_names;
  for( EnumeratorT<IModuleFactoryInfo*> e = application()->moduleFactoryInfos(); ++e ; ){
    IModuleFactoryInfo* mfi = (*e);
    const String& name = mfi->moduleName();
    max_name_size = math::max(max_name_size,CheckedConvert::toInteger(name.length()));
    module_names.add(name);
  }
  
  ITraceMng * trace = application()->traceMng();
  trace->info() << " ";
  trace->info() << std::setw(max_name_size) << "Module List";
  trace->info() << std::setw(max_name_size) << "-------------" << "--";
  for( int i=0, n=module_names.size(); i<n; ++i ){
    trace->info() << std::setw(max_name_size) << module_names[i];
  }

  trace->info() << " ";
  trace->info() << std::setw(max_name_size) << "Service List";
  trace->info() << std::setw(max_name_size) << "--------------" << "--";
  for( ServiceList::const_iterator i = service_list.begin(); i != service_list.end(); ++i) {
    IServiceInfo* si = i->second;
    OStringStream oss;
    oss() << std::setw(max_name_size) << i->first;
    StringCollection interfaces = si->implementedInterfaces();
    if (!interfaces.empty())
      oss() << " Implements : ";
    for(EnumeratorT<String> e(interfaces.enumerator());++e;) {
      oss() << e.current() << "  ";
    }      
    trace->info() << oss.str();
  }

  const Integer option_size = 20;
  trace->info() << " ";
  trace->info() << std::setw(max_name_size) << "Usage";
  trace->info() << std::setw(max_name_size) << "-------" << "--";
  trace->info() << application()->applicationName() << ".exe [-arcane_opt OPTION] dataset_file.arc";
  trace->info() << "Where OPTION is";
  trace->info() << std::setw(option_size) <<                "help" << " : this help page and abort";
  trace->info() << std::setw(option_size) <<     "arcane_internal" << " : save into a file internal Arcane informations and abort execution";
  trace->info() << std::setw(option_size) << "arcane_all_internal" << " : save into a file timeloop informations and abort execution";
  trace->info() << std::setw(option_size) <<     "arcane_database" << " : save internal database infos in file 'arcane_database.json'";
  trace->info() << std::setw(option_size) <<           "init_only" << " : only run initialization step";
  trace->info() << std::setw(option_size) <<            "continue" << " : continue an interrupted run";
  trace->info() << std::setw(option_size) <<       "max_iteration" << " : define maximum iteration number";
  trace->info() << std::setw(option_size) <<            "casename" << " : define case name";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonction appelee lorsque le programme est interrompu par le signal 'val'.
 *
 * Effectue une procedure d'urgence avant de quitter.
 */
extern "C" void
arcaneSignalHandler(int val)
{
  const char* signal_str = "Unknown";
	bool is_alarm = false;
  int written_signal_number = val;

  switch(val){
   case SIGSEGV: signal_str = "Segmentation Fault"; break;
   case SIGFPE:  signal_str = "Floating exception"; break;
#ifdef SIGBUS
   case SIGBUS:  signal_str = "Bus Error"; break;
#endif
#ifdef SIGSYS
   case SIGSYS:  signal_str = "System signal"; break;
#endif
#ifdef SIGPIPE
   case SIGPIPE: signal_str = "Broken pipe"; break;
#endif
#ifdef SIGALRM
   case SIGALRM:
     signal_str = "Sigalarm";
     is_alarm = true;
     break;
#endif
#ifdef SIGVTALRM
  case SIGVTALRM:
    signal_str = "Sigalarm(VirtualTime)";
    written_signal_number = SIGALRM; //! Utilise le même identifiant que le SIGALRM
    is_alarm = true;
    break;
#endif
  }

  cerr << "Signal Caught !!! number=" << val << " name=" << signal_str << ".\n";
#ifdef ARCANE_DEBUG
  //arcaneDebugPause("SIGNAL");
#endif

#ifndef ARCANE_OS_WIN32
  // Pour éviter que tous les PE écrivent le même fichier pour les SIGALRM,
  // seul le proc maitre le fait. Dans le cas des autres signaux, tout le monde
  // le fait.
  bool create_file = ArcaneMain::isMasterIO() || (!is_alarm);
  if (create_file){
    // Crée le fichier 'signal_*' pour indiquer en parallèle qu'un
    // signal a été envoyé
    mode_t mode = S_IRUSR | S_IWUSR;
    char path[256];
    sprintf(path,"signal_%d",written_signal_number);
    path[255] = '\0';
    int fd = ::open(path, O_WRONLY|O_CREAT|O_TRUNC, mode);
    if (fd!=(-1))
      ::close(fd);
  }
#endif

  // Repositionne les signaux pour la prochaine fois, si le signal est
  // un signal qui peut être reçu plusieurs fois.
  arcaneRedirectSignals(arcaneSignalHandler);

  Arcane::arcaneCallDefaultSignal(val);
  //::exit(val);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
