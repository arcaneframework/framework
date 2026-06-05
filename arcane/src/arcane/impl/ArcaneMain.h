// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneMain.h                                                (C) 2000-2025 */
/*                                                                           */
/* Class managing execution.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_ARCANEMAIN_H
#define ARCANE_IMPL_ARCANEMAIN_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/IFunctor.h"
#include "arcane/core/IArcaneMain.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ApplicationInfo;
class IMainFactory;
class IApplication;
class ICodeService;
class ServiceFactoryInfo;
class ArcaneMainExecInfo;
class DotNetRuntimeInitialisationInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_IMPL_EXPORT ArcaneMainExecutionOverrideFunctor
{
  friend class ArcaneMain;
  friend class ArcaneMainExecInfo;

 public:

  explicit ArcaneMainExecutionOverrideFunctor(IFunctor* functor)
  : m_functor(functor)
  , m_application(nullptr)
  {}
  IFunctor* functor() { return m_functor; }
  IApplication* application() { return m_application; }

 private:

  IFunctor* m_functor;
  IApplication* m_application;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_IMPL_EXPORT IApplicationBuildInfoVisitor
{
 public:

  virtual ~IApplicationBuildInfoVisitor() {}

 public:

  virtual void visit(ApplicationBuildInfo& app_build_info) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Execution management class.
 *
 * This class is internal to %Arcane and should not be used
 * directly. To initialize and execute the code, use the
 * ArcaneLauncher class.
 */
class ARCANE_IMPL_EXPORT ArcaneMain
: public IArcaneMain
{
  friend class ArcaneMainExecInfo;
  friend class ArcaneLauncher;
  friend class ArcaneMainAutoDetectRuntimeHelper;
  class Impl;

 public:

  // TODO: to be removed.
  ArcaneMain(const ApplicationInfo& infos, IMainFactory* factory);

 public:

  ArcaneMain(const ApplicationInfo& app_info, IMainFactory* factory,
             const ApplicationBuildInfo& app_build_info,
             const DotNetRuntimeInitialisationInfo& dotnet_init_info,
             const AcceleratorRuntimeInitialisationInfo& accelerator_init_info);
  ~ArcaneMain() override;

 public:

  /*!
   * \brief Entry point of the executable in Arcane.
   *
   * \note This method should not be called directly. It is
   * preferable to use the ArcaneLauncher class to manage the
   * execution launch.
   *
   * This method performs the following calls:
   *
   *  - creation of an instance <tt>a</tt> of IArcaneMain by calling
   *    createArcaneMain().
   *  - construction of <tt>a</tt> by the IArcaneMain::build() method
   *  - initialization of <tt>a</tt> by the IArcaneMain::initialize() method
   *  - launching the execution by the IArcaneMain::execute() method.
   *
   * \param app_info information about the executable.
   * \param factory builds architecture managers. If null,
   * uses the factory specified by setDefaultMainFactory() otherwise
   * a default factory is used.
   *
   * The call to this method must be preceded by Initialize();
   *
   * \retval 0 if the execution proceeded without error
   * \retval 1 in case of unknown error.
   * \retval 2 in case of standard exception (std::exception)
   * \retval 3 in case of architecture exception (IArcaneException)
   * \retval 4 in case of fatal error in Arcane.
   *
   */
  static int arcaneMain(const ApplicationInfo& app_info, IMainFactory* factory = nullptr);

  /*!
   * \brief Entry point of the executable in Arcane.
   *
   * This method calls arcaneMain(const ApplicationInfo&,IMainFactory*)
   * using the values of defaultApplicationInfo() and the factory specified
   * during calls to setDefaultMainFactory().
   */
  static int run();

  /*!
   * \brief Initializes Arcane.
   *
   * This method must be called before any use of an Arcane object. It can be called
   * multiple times, in which case the method
   * arcaneFinalize() must be called an equivalent number of times.
   *
   * The call to run() triggers the initialization. Therefore, it is generally
   * not necessary to call this method.
   */
  static void arcaneInitialize();

  /*!
   * \brief Terminates Arcane usage.
   *
   * This method must be called at the end of the execution. Once
   * called, Arcane objects must no longer be used.
   *
   * The call to run() manages the initialization and the call to this method.
   * Therefore, it is generally not necessary to call this method directly.
   *
   * \sa arcaneInitialize();
   */
  static void arcaneFinalize();

  /*!
   * \brief Indicates that certain objects are managed by a garbage collector.
   *
   * This property can only be set at the start of the calculation,
   * before calling arcaneInitialize().
   */
  static void setHasGarbageCollector();

  /*!
   * \brief Indicates that we are running in the .NET runtime.
   *
   * This property can only be set at the start of the calculation,
   * before calling arcaneInitialize().
   */
  static void setHasDotNETRuntime();

  /*!
   * \brief Sets the default factory.
   *
   * This method sets the default factory used if none
   * is specified in the call to arcaneMain().
   *
   * This method must be called before arcaneMain().
   */
  static void setDefaultMainFactory(IMainFactory* mf);

  /*!
   * \brief Default application info
   *
   * This method allows retrieving the instance of `ApplicationInfo`
   * which will be used when calling arcaneMain() without arguments.
   *
   * Therefore, this method should generally be called
   * before calling run().
   */
  static ApplicationInfo& defaultApplicationInfo();

  /*!
   * \brief Information for .Net runtime initialization.
   *
   * To be taken into account, this information must be modified
   * before calling run().
   */
  static DotNetRuntimeInitialisationInfo& defaultDotNetRuntimeInitialisationInfo();

  /*!
   * \brief Information for accelerator initialization.
   *
   * To be taken into account, this information must be modified
   * before calling run() or rundDirect().
   */
  static AcceleratorRuntimeInitialisationInfo& defaultAcceleratorRuntimeInitialisationInfo();

  /*!
   * \brief Information for accelerator initialization.
   *
   * To be taken into account, this information must be modified
   * before calling run() or rundDirect().
   */
  static ApplicationBuildInfo& defaultApplicationBuildInfo();

  /*!
   * \brief Calls the functor \a functor while catching possible
   * exceptions.
   *
   * The return value \a clean_abort is true if the code stops cleanly,
   * meaning that all processes and threads execute the same code in parallel.
   * This is the case, for example, if all processes
   * detect the same error and launch, for example, a ParallelFatalErrorException.
   * In this case, \a is_print indicates whether this process or thread displays
   * the error messages. If \a is_print is true, the error message is
   * displayed, otherwise it is not.
   *
   * If \ a clean_abort is false, it means that one of the processes or
   * thread stops without the others knowing, which generally
   * results in MPI_Abort in parallel.
   */

  static int callFunctorWithCatchedException(IFunctor* functor, IArcaneMain* amain,
                                             bool* clean_abort,
                                             bool is_print = true);

  /*!
   * brief Execution functor.
   *
   * This optional method allows setting a functor that will be called
   * instead of execute(). This functor is called once the application
   * is initialized. 
   *
   * Since the call to this functor replaces normal execution,
   * only one IApplication instance is available.
   * There is no subdomain, session, or mesh available.
   *
   */
  static void setExecuteOverrideFunctor(ArcaneMainExecutionOverrideFunctor* functor);

  //! Indicates if a '.Net' assembly is being executed from a C++ `main`.
  static bool hasDotNetWrapper();

  /*!
   * \brief Returns the time (in seconds) for the initialization
   * of accelerator runtimes for this process.
   *
   * Returns 0.0 if no accelerator runtime has been initialized.
   */
  static Real initializationTimeForAccelerator();

 public:

  /*!
   * \brief Adds a service factory.
   *
   * This method must be called before arcaneMain()
   */
  static void addServiceFactoryInfo(IServiceFactoryInfo* factory);

  /*!
   * \brief Adds a module factory
   *
   * This method must be called before arcaneMain()
   */
  static void addModuleFactoryInfo(IModuleFactoryInfo* factory);

 public:

  /*!
   * \brief Adds a visitor to fill ApplicationBuildInfo.
   *
   * The pointer passed as an argument must remain valid until the call to arcaneMain();
   * The registered visitors are called just before the application is created.
   */
  static void addApplicationBuildInfoVisitor(IApplicationBuildInfoVisitor* visitor);

 public:

  static void redirectSignals();
  static bool isMasterIO() { return m_is_master_io; }
  static void setUseTestLogger(bool v);

 public:

  void build() override;
  void initialize() override;
  bool parseArgs(StringList args) override;
  int execute() override;
  void doAbort() override;
  void setErrorCode(int errcode) override;
  int errorCode() const override { return m_error_code; }
  void finalize() override {}

 public:

  const ApplicationInfo& applicationInfo() const override;
  const ApplicationBuildInfo& applicationBuildInfo() const override;
  const DotNetRuntimeInitialisationInfo& dotnetRuntimeInitialisationInfo() const override;
  const AcceleratorRuntimeInitialisationInfo& acceleratorRuntimeInitialisationInfo() const override;
  IMainFactory* mainFactory() const override { return m_main_factory; }
  IApplication* application() const override { return m_application; }
  ServiceFactoryInfoCollection registeredServiceFactoryInfos() override;
  ModuleFactoryInfoCollection registeredModuleFactoryInfos() override;
  bool hasGarbageCollector() const override { return m_has_garbage_collector; }
  void setDirectExecuteFunctor(IDirectSubDomainExecuteFunctor* f) override { m_direct_sub_domain_execute_functor = f; }
  IDirectSubDomainExecuteFunctor* _directExecuteFunctor() const { return m_direct_sub_domain_execute_functor; }

 protected:

  IApplication* _application() { return m_application; }
  ApplicationBuildInfo& _applicationBuildInfo();
  static int _internalRun(IDirectSubDomainExecuteFunctor* func);

 private:

  Impl* m_p;
  IMainFactory* m_main_factory = nullptr;
  IApplication* m_application = nullptr;
  int m_error_code = 0;
  IDirectSubDomainExecuteFunctor* m_direct_sub_domain_execute_functor = nullptr;
  static bool m_has_garbage_collector;
  static bool m_is_master_io;
  static bool m_is_use_test_logger;
  static IMainFactory* m_default_main_factory;
  static ArcaneMainExecutionOverrideFunctor* m_exec_override_functor;

 private:

  static int _arcaneMain(const ApplicationInfo&, IMainFactory*);
  void _dumpHelp();
  void _parseApplicationBuildInfoArgs();
  //! Number of times arcaneInitialize() has been called
  static std::atomic<Int32> m_nb_arcane_init;
  //! 1 if init finished, 0 otherwise
  static std::atomic<Int32> m_is_init_done;
  static void _launchMissingInitException();
  static void _checkHasInit();
  static int _runDotNet();
  static void _checkAutoDetectMPI();
  static int _checkAutoDetectAccelerator(bool& has_accelerator);
  static void _setArcaneLibraryPath();
  static int _initRuntimes();
  static int _checkTestLoggerResult();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
