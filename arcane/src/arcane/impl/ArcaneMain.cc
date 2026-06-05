// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneMain.cc                                               (C) 2000-2026 */
/*                                                                           */
/* Execution management class.                                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/ArcaneMain.h"

#include "arcane/utils/Iostream.h"
#include "arcane/utils/List.h"
#include "arcane/utils/Iterator.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ParallelFatalErrorException.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/SignalException.h"
#include "arcane/utils/TimeoutException.h"
#include "arcane/utils/ArithmeticException.h"
#include "arcane/utils/StringBuilder.h"
#include "arccore/base/internal/IDynamicLibraryLoader.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/CommandLineArguments.h"
#include "arcane/utils/TestLogger.h"
#include "arcane/utils/MemoryUtils.h"

#include "arccore/base/internal/ConvertInternal.h"
#include "arccore/common/ExceptionUtils.h"
#include "arccore/common/internal/MemoryUtilsInternal.h"
#include "arccore/common/accelerator/internal/RuntimeLoader.h"

#include "arcane/core/IMainFactory.h"
#include "arcane/core/IApplication.h"
#include "arcane/core/IServiceLoader.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IParallelSuperMng.h"
#include "arcane/core/ISession.h"
#include "arcane/core/VariableRef.h"
#include "arcane/core/ItemTypeMng.h"
#include "arcane/core/ServiceUtils.h"
#include "arcane/core/CaseOptions.h"
#include "arcane/core/ItemGroupImpl.h"
#include "arcane/core/DotNetRuntimeInitialisationInfo.h"
#include "arcane/core/AcceleratorRuntimeInitialisationInfo.h"
#include "arcane/core/ApplicationBuildInfo.h"

#include "arcane/core/IServiceFactory.h"
#include "arcane/core/IModuleFactory.h"

#include "arcane/impl/MainFactory.h"
#include "arcane/impl/InternalInfosDumper.h"
#include "arcane/impl/internal/ArcaneMainExecInfo.h"
#include "arcane/impl/internal/ThreadBindingMng.h"

#include "arccore/common/accelerator/internal/RegisterRuntimeInfo.h"

#include "arcane_internal_config.h"

#include <signal.h>
#include <exception>
#ifndef ARCANE_OS_WIN32
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

#include <set>
#include <map>
#include <chrono>

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
  //! Number of times auto-detection for MPI and accelerators has been run
  std::atomic<Int32> m_nb_autodetect = 0;
  //! Return code for auto-detection
  Int32 m_autodetect_return_value = 0;
  //! Time spent (in seconds) during initialization for accelerators
  Real m_init_time_accelerator = 0.0;
};
} // namespace Arcane

namespace
{
Arcane::ArcaneMainStaticInfo* global_static_info = nullptr;
Arcane::ArcaneMainStaticInfo* _staticInfo()
{
  // TODO: see if it needs to be protected in multi-threading.
  if (!global_static_info)
    global_static_info = new Arcane::ArcaneMainStaticInfo();
  return global_static_info;
}
void _deleteStaticInfo()
{
  delete global_static_info;
  global_static_info = nullptr;
}
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C" void arcaneEndProgram()
{
  // Just for a third entry point.
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C" {
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
createArcaneMainBatch(const ApplicationInfo& exe_info, IMainFactory*);

extern "C++" ARCANE_IMPL_EXPORT ICodeService*
createArcaneCodeService(IApplication* app);

extern "C++" ARCCORE_COMMON_EXPORT void
arccorePrintSpecificMemoryStats();

std::atomic<Int32> ArcaneMain::m_nb_arcane_init(0);
std::atomic<Int32> ArcaneMain::m_is_init_done(0);
bool ArcaneMain::m_has_garbage_collector = false;
bool ArcaneMain::m_is_master_io = true;
bool ArcaneMain::m_is_use_test_logger = false;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ArcaneBatchMainFactory
: public MainFactory
{
 public:

  IArcaneMain* createArcaneMain(const ApplicationInfo& app_info) override
  {
    return createArcaneMainBatch(app_info, this);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ArcaneMain::Impl
{
 public:

  explicit Impl(const ApplicationInfo& infos)
  : m_app_info(infos)
  , m_application_build_info(ArcaneMain::defaultApplicationBuildInfo())
  , m_dotnet_info(ArcaneMain::defaultDotNetRuntimeInitialisationInfo())
  , m_accelerator_info(ArcaneMain::defaultAcceleratorRuntimeInitialisationInfo())
  {}
  Impl(const ApplicationInfo& infos, const ApplicationBuildInfo& build_infos,
       const DotNetRuntimeInitialisationInfo& dotnet_info,
       const AcceleratorRuntimeInitialisationInfo& accelerator_info)
  : m_app_info(infos)
  , m_application_build_info(build_infos)
  , m_dotnet_info(dotnet_info)
  , m_accelerator_info(accelerator_info)
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
  (void)builtInGetValue(redirect_signals, rv);
  if (redirect_signals) {
    arcaneRedirectSignals(arcaneSignalHandler);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMain::
setUseTestLogger(bool v)
{
  m_is_use_test_logger = v;
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

  ArcaneMainExecFunctor(const ApplicationInfo& app_info, IArcaneMain* exec_main)
  : m_app_info(app_info)
  , m_exec_main(exec_main)
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
 * \brief Class to manage calls to runtime auto-detection mechanisms (MPI, Accelerators).
 *
 * This class ensures that the auto-detection mechanisms are
 * called only once. Auto-detection occurs when check() is called.
 */
class ArcaneMainAutoDetectRuntimeHelper
{
 public:

  Int32 check()
  {
    auto* x = _staticInfo();
    if (x->m_nb_autodetect > 0)
      return x->m_autodetect_return_value;

    ArcaneMain::_setArcaneLibraryPath();

    std::chrono::high_resolution_clock clock;

    // TODO: make thread-safe
    {
      ArcaneMain::_checkAutoDetectMPI();

      bool has_accelerator = false;
      // Measures the initialization time.
      // Since Arcane has not been initialized yet, methods from the 'platform' namespace
      // should not be used here.
      auto start_time = clock.now();
      x->m_autodetect_return_value = ArcaneMain::_checkAutoDetectAccelerator(has_accelerator);
      auto end_time = clock.now();
      // Only retrieve the time if an accelerator was used
      if (has_accelerator)
        x->m_init_time_accelerator = _getTime(end_time, start_time);
      ++x->m_nb_autodetect;
    }
    return x->m_autodetect_return_value;
  }

  template <typename TimeType>
  Real _getTime(TimeType end_time, TimeType start_time)
  {
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    Real x = static_cast<Real>(duration.count());
    return x / 1.0e9;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creation of the 'IArcaneMain' instance.
 *
 * If the instance is already created, this method does nothing.
 *
 * In case of an exception, this method returns a non-zero value.
 * In case of a license manager error, \a m_ret_val is
 * set to a non-zero value but this method returns 0.
 */
int ArcaneMainExecInfo::
initialize()
{
  if (m_exec_main)
    return 0;

  m_ret_val = 0;
  m_clean_abort = false;

  ArcaneMain::redirectSignals();

  // Creation of the execution class
  try {
    if (m_has_build_info) {
      ArcaneMain* x = new ArcaneMain(m_app_info, m_main_factory,
                                     m_application_build_info,
                                     ArcaneMain::defaultDotNetRuntimeInitialisationInfo(),
                                     ArcaneMain::defaultAcceleratorRuntimeInitialisationInfo());
      m_exec_main = x;
    }
    else {
      m_exec_main = m_main_factory->createArcaneMain(m_app_info);
    }
    m_exec_main->build();
    ArcaneMain::m_is_master_io = m_exec_main->application()->parallelSuperMng()->isMasterIO();
    m_exec_main->initialize();
    IArcaneMain::setArcaneMain(m_exec_main);
  }
  catch (const ArithmeticException& ex) {
    cerr << "** CATCH ARITHMETIC_EXCEPTION\n";
    return ExceptionUtils::print(ex, nullptr);
  }
  catch (const Exception& ex) {
    return ExceptionUtils::print(ex, nullptr);
  }
  catch (const std::exception& ex) {
    return ExceptionUtils::print(ex, nullptr);
  }
  catch (...) {
    return ExceptionUtils::print(nullptr);
  }

  // Redirects signals again because certain
  // initialization routines (for example MPI) may
  // divert them.
  ArcaneMain::redirectSignals();

  m_ret_val = 0;
  m_clean_abort = false;

#ifdef ARCANE_FLEXLM
  try {
    IApplication* app = m_exec_main->application();
    ITraceMng* trace = app->traceMng();
    IParallelSuperMng* parallel_super_mng = app->parallelSuperMng();
    trace->info() << "Initializing license manager";
    FlexLMMng::instance()->init(parallel_super_mng);

    // The parallel license policy is delegated to the applications
    //     bool is_parallel = parallel_super_mng->isParallel();
    //     FlexLMTools<ArcaneFeatureModel> license_tool;
    //     Integer commSize = parallel_super_mng->commSize();
    //     if (is_parallel && commSize > 1)
    //       { // The parallel feature is only activated if necessary
    //         license_tool.getLicense(ArcaneFeatureModel::ArcaneParallel,commSize);
    //       }
  }
  catch (const Exception& ex) {
    IApplication* app = m_exec_main->application();
    ITraceMng* trace = app->traceMng();
    m_ret_val = arcanePrintArcaneException(ex, trace);
    if (ex.isCollective()) {
      m_clean_abort = true;
    }
  }
#endif
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// NOTE: This method must not throw exceptions
void ArcaneMainExecInfo::
execute()
{
  if (m_ret_val != 0)
    return;

  if (m_direct_exec_functor)
    m_exec_main->setDirectExecuteFunctor(m_direct_exec_functor);

  ArcaneMainExecFunctor exec_functor(m_app_info, m_exec_main);
  if (ArcaneMain::m_exec_override_functor) {
    // Obsolete. Do not use.
    IApplication* app = m_exec_main->application();
    ArcaneMain::m_exec_override_functor->m_application = app;
    ITraceMng* trace = app->traceMng();
    trace->info() << "Calling overriding functor";
    m_ret_val = ArcaneMain::callFunctorWithCatchedException(ArcaneMain::m_exec_override_functor->functor(),
                                                            m_exec_main, &m_clean_abort, true);
  }
  else
    m_ret_val = ArcaneMain::callFunctorWithCatchedException(&exec_functor, m_exec_main, &m_clean_abort, true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMainExecInfo::
finalize()
{
  // Disables floating exceptions
  platform::enableFloatingException(false);

  // If execution went well but the user specified an
  // error code, we retrieve it.
  int exe_error_code = m_exec_main->errorCode();
  if (m_ret_val == 0 && exe_error_code != 0) {
    m_ret_val = exe_error_code;
  }
  else if (m_ret_val != 0)
    m_exec_main->setErrorCode(m_ret_val);

  m_exec_main->finalize();

  if (m_ret_val != 0 && !m_clean_abort)
    m_exec_main->doAbort();

  // Code destruction.
  // Be careful not to destroy the manager beforehand because when an
  // architecture exception is generated, it uses an ITraceMng to display
  // the message
  delete m_exec_main;
  m_exec_main = nullptr;
#ifndef ARCANE_USE_MPC
  IArcaneMain::setArcaneMain(m_exec_main);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Executable entry point.
 * \internal
 */
int ArcaneMain::
_arcaneMain(const ApplicationInfo& app_info, IMainFactory* factory)
{
  if (!factory)
    return 5;

  ArcaneMainExecInfo exec_info(app_info, factory);
  int r = exec_info.initialize();
  if (r != 0)
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
callFunctorWithCatchedException(IFunctor* functor, IArcaneMain* exec_main,
                                bool* clean_abort, bool is_print)
{
  int ret_val = 0;
  *clean_abort = false;
  IApplication* app = exec_main->application();
  ITraceMng* trace = app->traceMng();
  bool is_parallel = app->parallelSuperMng()->isParallel();
  bool is_master = app->parallelSuperMng()->isMasterIO();
  try {
    functor->executeFunctor();
  }
  catch (const FatalErrorException& ex) {
    if (ex.isCollective()) {
      if (is_parallel) {
        *clean_abort = true;
        ret_val = 5;
        if (is_master && is_print) {
          std::ofstream ofile("fatal");
          ofile << ret_val << '\n';
          ofile.flush();
          trace->error() << "ParallelFatalErrorException caught in ArcaneMain::callFunctor: " << ex << '\n';
        }
      }
      else {
        trace->error() << "ParallelFatalErrorException caught in ArcaneMain::callFunctor: " << ex << '\n';
        ret_val = 4;
      }
    }
    else {
      trace->error() << Trace::Color::red() << "FatalErrorException caught in ArcaneMain::callFunctor: " << ex << '\n';
      ret_val = 4;
    }
  }
  catch (const SignalException& ex) {
    trace->error() << "SignalException caught in ArcaneMain::callFunctor: " << ex << '\n';
    ret_val = 6;
  }
  catch (const TimeoutException& ex) {
    trace->error() << "TimeoutException caught in ArcaneMain::callFunctor: " << ex << '\n';
    ret_val = 7;
  }
  catch (const ParallelFatalErrorException& ex) {
    // TODO: use the FatalErrorException code in collective mode.n
    if (is_parallel) {
      *clean_abort = true;
      ret_val = 5;
      if (is_master && is_print) {
        std::ofstream ofile("fatal");
        ofile << ret_val << '\n';
        ofile.flush();
        trace->error() << "ParallelFatalErrorException caught in ArcaneMain::callFunctor: " << ex << '\n';
      }
    }
    else {
      trace->error() << "ParallelFatalErrorException caught in ArcaneMain::callFunctor: " << ex << '\n';
      ret_val = 4;
    }
  }
  catch (const ArithmeticException& ex) {
    cerr << "** ARITHMETIC EXCEPTION!\n";
    ret_val = ExceptionUtils::print(ex, trace);
    if (ex.isCollective()) {
      *clean_abort = true;
    }
  }
  catch (const Exception& ex) {
    ret_val = ExceptionUtils::print(ex, trace);
    if (ex.isCollective()) {
      *clean_abort = true;
    }
  }
  catch (const std::exception& ex) {
    ret_val = ExceptionUtils::print(ex, trace);
  }
  catch (...) {
    ret_val = ExceptionUtils::print(trace);
  }
  return ret_val;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMain::
_launchMissingInitException()
{
  std::cerr << "ERROR: ArcaneMain: missing call to ArcaneMain::arcaneInitialize().\n";
  throw std::runtime_error("Missing call to ArcaneMain::arcaneInitialize()");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMain::
_checkHasInit()
{
  if (m_nb_arcane_init <= 0)
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
  if (m_nb_arcane_init != 0) {
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
  if (m_nb_arcane_init != 0) {
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
 * \brief Positions the path containing Arcane dynamic libraries.
 *
 * This method must only be called once.
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
arcaneInitialize()
{
  // The first thread that arrives here performs the init.
  // Others must wait for the init to finish.
  if (m_nb_arcane_init.fetch_add(1) == 0) {
    (void)_staticInfo();
    Exception::staticInit();
    dom::DOMImplementation::initialize();
    platform::platformInitialize();

    // Checks if we want to use the old mechanism (before 3.15)
    // to convert character strings to numeric types
    if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_USE_LEGACY_BUILTINVALUECONVERT", true))
      Convert::Impl::ConvertPolicy::setUseFromChars(v.value() == 0);

    // Creates the type manager singleton
    ItemTypeMng::_singleton();
    initializeStringConverter();
    arcaneInitCheckMemory();
    // Initializes the empty group singleton and keeps a reference to it.
    ItemGroupImpl::_buildSharedNull();
    m_is_init_done = 1;
  }
  else
    // Waits for the thread performing the init to finish
    while (m_is_init_done.load() == 0)
      ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMain::
arcaneFinalize()
{
  _checkHasInit();

  if (m_nb_arcane_init.fetch_sub(1) == 1) {
    _deleteStaticInfo();

    //! Removes our reference on ItemGroupImpl::shared_null.
    ItemGroupImpl::_destroySharedNull();

    {
      auto x = IDynamicLibraryLoader::getDefault();
      if (x) {
        x->closeLibraries();
      }
    }
    arccorePrintSpecificMemoryStats();
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
    m_is_init_done = 0;
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

Real ArcaneMain::
initializationTimeForAccelerator()
{
  return _staticInfo()->m_init_time_accelerator;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int ArcaneMain::
arcaneMain(const ApplicationInfo& app_info, IMainFactory* factory)
{
  _checkHasInit();

  ScopedPtrT<IMainFactory> default_factory;
  if (!factory) {
    factory = m_default_main_factory;
    if (!factory) {
      factory = new ArcaneBatchMainFactory();
      default_factory = factory;
    }
  }

  int ret = _arcaneMain(app_info, factory);

  default_factory = nullptr;

  // Error code 5 represents a parallel error for all
  // processors.
  if (ret != 0 && ret != 5)
    cerr << "* Process return: " << ret << '\n';
  if (ret == 5)
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
_checkTestLoggerResult()
{
  if (!m_is_use_test_logger)
    return 0;
  if (!m_is_master_io)
    return 0;
  return TestLogger::compare();
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
  if (r != 0)
    return r;

  DotNetRuntimeInitialisationInfo& dotnet_info = defaultDotNetRuntimeInitialisationInfo();

  // If we arrive here and the C# runtime is already loaded
  // (because Main is in C# for example), we do not launch the wrapper
  bool is_in_dotnet = platform::hasDotNETRuntime();
  if (!is_in_dotnet && dotnet_info.isUsingDotNetRuntime()) {
    r = _runDotNet();
    // Before version 3.7.8 we called arcaneFinalize() because it could
    // cause problems with the '.Net' Garbage Collector. Normally these
    // problems are fixed but we allow the previous behavior just in case.
    bool do_finalize = false;
    String x = platform::getEnvironmentVariable("ARCANE_DOTNET_USE_LEGACY_DESTROY");
    if (x == "1")
      do_finalize = false;
    if (x == "0")
      do_finalize = true;
    if (do_finalize)
      arcaneFinalize();
  }
  else {
    arcaneInitialize();
    r = arcaneMain(defaultApplicationInfo(), nullptr);
    arcaneFinalize();
  }
  if (r != 0)
    return r;
  return _checkTestLoggerResult();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int ArcaneMain::
_runDotNet()
{
  auto si = _staticInfo();
  si->m_has_dotnet_wrapper = true;

  // TODO: check that init has not been done.

  // To be able to execute .Net code, it is necessary
  // to call the method 'arcane_mono_main' which is located
  // in the dynamic library 'arcane_mono'.

  typedef int (*DotNetMainFunctor)(const CommandLineArguments& cmd_args, const String& assembly_name);

  const ApplicationInfo& app_info = defaultApplicationInfo();
  const DotNetRuntimeInitialisationInfo& dotnet_info = defaultDotNetRuntimeInitialisationInfo();

  DotNetMainFunctor my_functor = nullptr;
  String os_dir(si->m_arcane_lib_path);

  try {
    IDynamicLibraryLoader* dll_loader = IDynamicLibraryLoader::getDefault();

    String dll_name = "arcane_mono";
    String symbol_name = "arcane_mono_main2";

    String runtime_name = dotnet_info.embeddedRuntime();

    if (runtime_name.null() || runtime_name == "mono")
      // Mono is the default if nothing is specified.
      ;
    else if (runtime_name == "coreclr") {
      dll_name = "arcane_dotnet_coreclr";
      symbol_name = "arcane_dotnet_coreclr_main";
    }
    else
      ARCANE_FATAL("Unknown '.Net' runtime '{0}'. Valid values are 'mono' or 'coreclr'", runtime_name);

    IDynamicLibrary* dl = dll_loader->open(os_dir, dll_name);
    if (!dl)
      ARCANE_FATAL("Can not found dynamic library '{0}' for using .Net", dll_name);

    bool is_found = false;
    void* functor_addr = dl->getSymbolAddress(symbol_name, &is_found);
    if (!is_found)
      ARCANE_FATAL("Can not find symbol '{0}' in library '{1}'", symbol_name, dll_name);

    my_functor = reinterpret_cast<DotNetMainFunctor>(functor_addr);
  }
  catch (const Exception& ex) {
    return ExceptionUtils::print(ex, nullptr);
  }
  catch (const std::exception& ex) {
    return ExceptionUtils::print(ex, nullptr);
  }
  catch (...) {
    return ExceptionUtils::print(nullptr);
  }

  if (my_functor) {
    const CommandLineArguments& cmd_args = app_info.commandLineArguments();
    // TODO: check that the assembly 'Arcane.Main.dll' exists.
    String new_name = os_dir + "/Arcane.Main.dll";
    return (*my_functor)(cmd_args, new_name);
  }
  return (-1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Do not call directly but go through ArcaneMainAutoDetectHelper.
void ArcaneMain::
_checkAutoDetectMPI()
{
  auto si = _staticInfo();
  // To automatically register MPI, it is necessary
  // to call the method 'arcaneAutoDetectMessagePassingServiceMPI' which is located
  // in the dynamic library 'arcane_mpi'.

  typedef void (*ArcaneAutoDetectMPIFunctor)();

  IDynamicLibraryLoader* dll_loader = IDynamicLibraryLoader::getDefault();

  String os_dir(si->m_arcane_lib_path);
  String dll_name = "arcane_mpi";
  String symbol_name = "arcaneAutoDetectMessagePassingServiceMPI";
  IDynamicLibrary* dl = dll_loader->open(os_dir, dll_name);
  if (!dl)
    return;

  bool is_found = false;
  void* functor_addr = dl->getSymbolAddress(symbol_name, &is_found);
  if (!is_found)
    return;

  auto my_functor = reinterpret_cast<ArcaneAutoDetectMPIFunctor>(functor_addr);
  if (my_functor)
    (*my_functor)();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Detects and loads accelerator runtime management.
 *
 * In return, has_accelerator is true if an accelerator runtime was loaded.
 *
 * \retval 0 if everything is OK
 *
 * \note Do not call this method directly but
 * go through ArcaneMainAutoDetectHelper.
 */
int ArcaneMain::
_checkAutoDetectAccelerator(bool& has_accelerator)
{
  has_accelerator = false;
  String default_runtime_name;
#if defined(ARCANE_ACCELERATOR_RUNTIME)
  default_runtime_name = ARCANE_ACCELERATOR_RUNTIME;
#endif
  auto si = _staticInfo();
  AcceleratorRuntimeInitialisationInfo& init_info = si->m_accelerator_init_info;
  if (!init_info.isUsingAcceleratorRuntime())
    return 0;
  return Accelerator::Impl::RuntimeLoader::loadRuntime(init_info, default_runtime_name, si->m_arcane_lib_path, has_accelerator);
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
ArcaneMain(const ApplicationInfo& app_info, IMainFactory* factory)
: m_p(new Impl(app_info))
, m_main_factory(factory)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneMain::
ArcaneMain(const ApplicationInfo& app_info, IMainFactory* factory,
           const ApplicationBuildInfo& app_build_info,
           const DotNetRuntimeInitialisationInfo& dotnet_info,
           const AcceleratorRuntimeInitialisationInfo& accelerator_info)
: m_p(new Impl(app_info, app_build_info, dotnet_info, accelerator_info))
, m_main_factory(factory)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneMain::
~ArcaneMain()
{
  // Ensures that observers associated with TheadBindingMng are removed
  // before finalization to avoid wasting threads when
  // it is no longer useful.
  m_p->m_thread_binding_mng.finalize();
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
  // Calls the registered visitors.
  {
    auto& x = _staticInfo()->m_application_build_info_visitors;
    for (IApplicationBuildInfoVisitor* v : x) {
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
  if (errcode != 0) {
    // Only the master process writes the file unless it is a fatal error because
    // in this case, any PE can do it.
    if (ArcaneMain::m_is_master_io || errcode == 4) {
      String errname = "fatal_" + String::fromNumber(errcode);
      std::ofstream ofile(errname.localstr());
      ofile.close();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ArcaneMain::
parseArgs(StringList args)
{
  // If true, display internal information
  bool arcane_internal = false;
  // If true, display internal information for each time loop
  bool arcane_all_internal = false;
  // If true, generates a file containing internal code information.
  bool arcane_database = false;
  // If true, display information about present modules and services and a brief help
  bool arcane_help = false;

  String us_arcane_opt("-arcane_opt");
  String us_help("help");
  String us_arcane_internal("arcane_internal");
  String us_arcane_all_internal("arcane_all_internal");
  String us_arcane_database("arcane_database");

  StringList unknown_args;
  for (Integer i = 0, s = args.count(); i < s; ++i) {
    if (args[i].startsWith("-A")) {
      continue;
    }
    if (args[i] != us_arcane_opt) {
      unknown_args.add(args[i]);
      continue;
    }
    bool is_valid_opt = false;
    ++i;
    String str;
    if (i < s)
      str = args[i];
    if (str == us_arcane_internal) {
      arcane_internal = true;
      is_valid_opt = true;
    }
    if (str == us_arcane_all_internal) {
      arcane_all_internal = true;
      is_valid_opt = true;
    }
    if (str == us_arcane_database) {
      arcane_database = true;
      is_valid_opt = true;
    }
    if (str == us_help) {
      arcane_help = true;
      is_valid_opt = true;
    }
    if (!is_valid_opt) {
      // If the option is not valid, add it to the list of
      // unprocessed options
      unknown_args.add(us_arcane_opt);
      if (!str.null())
        unknown_args.add(str);
      //trace->fatal() << "Unknown arcane option <" << str << ">\n";
    }
  }

  bool do_stop = false;
  if (arcane_database) {
    InternalInfosDumper dumper(application());
    dumper.dumpArcaneDatabase();
    do_stop = true;
  }
  if (arcane_internal) {
    InternalInfosDumper dumper(application());
    dumper.dumpInternalInfos();
    do_stop = true;
  }
  if (arcane_all_internal) {
    InternalInfosDumper dumper(application());
    dumper.dumpInternalAllInfos();
    do_stop = true;
  }
  if (arcane_help) {
    _dumpHelp();
    do_stop = true;
  }

  args.clear();
  for (StringList::Enumerator i(unknown_args); ++i;)
    args.add(*i);

  return do_stop;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneMain::
_dumpHelp()
{
  // We want to list by IServiceInfo. Since it is possible that a service has multiple
  // factories, we filter using done_set.
  typedef std::multimap<String, IServiceInfo*> ServiceList;
  ServiceList service_list;
  Integer max_name_size = 0;

  {
    // On veut lister par IServiceInfo. Comme il est possible qu'un service ait plusieurs
    // fabriques, on filtre grâce à done_set.
    std::set<IServiceInfo*> done_set;
    for (ServiceFactory2Collection::Enumerator j(application()->serviceFactories2()); ++j;) {
      IServiceInfo* si = (*j)->serviceInfo();
      if (done_set.find(si) != done_set.end()) {
        continue;
      }
      done_set.insert(si);
      const String& name = si->localName();
      max_name_size = math::max(max_name_size, CheckedConvert::toInteger(name.length()));
      service_list.insert(std::make_pair(name, si));
    }
  }

  UniqueArray<String> module_names;
  for (EnumeratorT<IModuleFactoryInfo*> e = application()->moduleFactoryInfos(); ++e;) {
    IModuleFactoryInfo* mfi = (*e);
    const String& name = mfi->moduleName();
    max_name_size = math::max(max_name_size, CheckedConvert::toInteger(name.length()));
    module_names.add(name);
  }

  ITraceMng* trace = application()->traceMng();
  trace->info() << " ";
  trace->info() << std::setw(max_name_size) << "Module List";
  trace->info() << std::setw(max_name_size) << "-------------"
                << "--";
  for (int i = 0, n = module_names.size(); i < n; ++i) {
    trace->info() << std::setw(max_name_size) << module_names[i];
  }

  trace->info() << " ";
  trace->info() << std::setw(max_name_size) << "Service List";
  trace->info() << std::setw(max_name_size) << "--------------"
                << "--";
  for (ServiceList::const_iterator i = service_list.begin(); i != service_list.end(); ++i) {
    IServiceInfo* si = i->second;
    OStringStream oss;
    oss() << std::setw(max_name_size) << i->first;
    StringCollection interfaces = si->implementedInterfaces();
    if (!interfaces.empty())
      oss() << " Implements : ";
    for (EnumeratorT<String> e(interfaces.enumerator()); ++e;) {
      oss() << e.current() << "  ";
    }
    trace->info() << oss.str();
  }

  const Integer option_size = 20;
  trace->info() << " ";
  trace->info() << std::setw(max_name_size) << "Usage";
  trace->info() << std::setw(max_name_size) << "-------"
                << "--";
  trace->info() << application()->applicationName() << ".exe [-arcane_opt OPTION] dataset_file.arc";
  trace->info() << "Where OPTION is";
  trace->info() << std::setw(option_size) << "help"
                << " : this help page and abort";
  trace->info() << std::setw(option_size) << "arcane_internal"
                << " : save into a file internal Arcane informations and abort execution";
  trace->info() << std::setw(option_size) << "arcane_all_internal"
                << " : save into a file timeloop informations and abort execution";
  trace->info() << std::setw(option_size) << "arcane_database"
                << " : save internal database infos in file 'arcane_database.json'";
  trace->info() << std::setw(option_size) << "init_only"
                << " : only run initialization step";
  trace->info() << std::setw(option_size) << "continue"
                << " : continue an interrupted run";
  trace->info() << std::setw(option_size) << "max_iteration"
                << " : define maximum iteration number";
  trace->info() << std::setw(option_size) << "casename"
                << " : define case name";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Brief function called when the program is interrupted by the 'val' signal.
 *
 * Performs an emergency procedure before exiting.
 */
extern "C" void
arcaneSignalHandler(int val)
{
  const char* signal_str = "Unknown";
  bool is_alarm = false;
  int written_signal_number = val;

  switch (val) {
  case SIGSEGV:
    signal_str = "Segmentation Fault";
    break;
  case SIGFPE:
    signal_str = "Floating exception";
    break;
#ifdef SIGBUS
  case SIGBUS:
    signal_str = "Bus Error";
    break;
#endif
#ifdef SIGSYS
  case SIGSYS:
    signal_str = "System signal";
    break;
#endif
#ifdef SIGPIPE
  case SIGPIPE:
    signal_str = "Broken pipe";
    break;
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
    written_signal_number = SIGALRM; //! Uses the same identifier as SIGALRM
    is_alarm = true;
    break;
#endif
  }

  cerr << "Signal Caught !!! number=" << val << " name=" << signal_str << ".\n";
#ifdef ARCANE_DEBUG
  //arcaneDebugPause("SIGNAL");
#endif

#ifndef ARCANE_OS_WIN32
  // To prevent all PEs from writing the same file for SIGALRM,
  // only the master process does it. In the case of other signals, everyone
  // does it.
  bool create_file = ArcaneMain::isMasterIO() || (!is_alarm);
  if (create_file) {
    // Creates the 'signal_*' file to indicate in parallel that a
    // signal has been sent
    mode_t mode = S_IRUSR | S_IWUSR;
    char path[256];
    sprintf(path, "signal_%d", written_signal_number);
    path[255] = '\0';
    int fd = ::open(path, O_WRONLY | O_CREAT | O_TRUNC, mode);
    if (fd != (-1))
      ::close(fd);
  }
#endif

  // Repositions the signals for next time, if the signal is
  // one that can be received multiple times.
  arcaneRedirectSignals(arcaneSignalHandler);

  Arcane::arcaneCallDefaultSignal(val);
  //::exit(val);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
