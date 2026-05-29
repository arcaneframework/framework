// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneLauncher.h                                            (C) 2000-2025 */
/*                                                                           */
/* Class managing execution.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_LAUNCHER_ARCANELAUNCHER_H
#define ARCANE_LAUNCHER_ARCANELAUNCHER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/launcher/LauncherGlobal.h"

// The following files are not directly used in this '.h'
// but are added so that user code only needs to include
// 'ArcaneLauncher.h'.
#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/CommandLineArguments.h"

#include "arcane/core/ApplicationBuildInfo.h"
#include "arcane/core/DotNetRuntimeInitialisationInfo.h"
#include "arcane/core/AcceleratorRuntimeInitialisationInfo.h"

#include "arcane/launcher/DirectExecutionContext.h"
#include "arcane/launcher/DirectSubDomainExecutionContext.h"
#include "arcane/launcher/IDirectExecutionContext.h"
#include "arcane/launcher/StandaloneAcceleratorMng.h"
#include "arcane/launcher/StandaloneSubDomain.h"

#include <functional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IMainFactory;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Execution management class.
 *
 * There are two modes of using %Arcane: classic mode and standalone mode.
 *
 * Regardless of the mode chosen, the first thing to do is initialize %Arcane by
 * setting the arguments via the init() method, because certain command-line
 * parameters are used to populate the properties of applicationInfo() and
 * dotNetRuntimeInitialisationInfo().
 *
 * The page \ref arcanedoc_execution_launcher provides usage examples.
 *
 * The two execution modes are:
 * - classic mode, which uses a time loop, and thus the entire execution
 *   will be managed by %Arcane. In this mode, you simply call the run() method
 *   without arguments.
 * - standalone mode, which allows %Arcane to be used as a library.
 *   For this mode, you must use the createStandaloneSubDomain()
 *   or createStandaloneAcceleratorMng() method. The page
 *   \ref arcanedoc_execution_direct_execution describes how to use this mechanism.
 *
 * The classic usage is as follows:
 *
 * \code
 * int main(int* argc,char* argv[])
 * {
 *   ArcaneLauncher::init(CommandLineArguments(&argc,&argv));
 *   auto& app_info = ArcaneLauncher::applicationInfo();
 *   app_info.setCodeName("MyCode");
 *   app_info.setCodeVersion(VersionInfo(1,0,0));
 *   return ArcaneLauncher::run();
 * }
 * \endcode
 */
class ARCANE_LAUNCHER_EXPORT ArcaneLauncher
{
  friend StandaloneSubDomain;

 public:

  /*!
   * \brief Positions information from command-line arguments and initializes
   * the launcher.
   *
   * This method fills the uninitialized values of applicationInfo() and
   * dotNetRuntimeInitialisationInfo() with the parameters specified in \a args.
   *
   * This method must only be called once. Additional calls generate a
   * FatalErrorException.
   */
  static void init(const CommandLineArguments& args);

  /*!
   * \brief Indicates if init() has already been called.
   */
  static bool isInitialized();

  /*!
   * \brief Entry point of the executable in Arcane.
   *
   * This method initializes the application, reads the dataset, and executes
   * the code according to the time loop specified in the dataset.
   *
   * \retval 0 upon success
   * \return a value different from 0 in case of error.
   */
  static int run();

  /*!
   * \brief Direct execution.
   *
   * Initializes the application and calls the function \a func after
   * initialization.
   * This method must only be called in sequential execution.
   */
  static int run(std::function<int(DirectExecutionContext&)> func);

  /*!
   * \brief Direct execution with subdomain creation.
   *
   * Initializes the application and creates the subdomain(s) and calls
   * the function \a func afterward.
   * This method allows executing code without going through the time loop
   * mechanisms.
   * This method automatically manages the creation of subdomains
   * based on launch parameters (MPI parallel execution, multithreading, ...).
   */
  static int run(std::function<int(DirectSubDomainExecutionContext&)> func);

  /*!
   * \brief Positions the default factory for creating the different managers
   *
   * This method must be called before run(). The instance passed as an argument
   * must remain valid during the execution of run(). The caller remains the owner
   * of the instance.
   */
  static void setDefaultMainFactory(IMainFactory* mf);

  /*!
   * \brief Application information.
   *
   * This method allows retrieving the `ApplicationInfo` instance
   * that will be used when calling run().
   *
   * To be taken into account, this information must be modified
   * before calling run() or runDirect().
   */
  static ApplicationInfo& applicationInfo();

  /*!
   * \brief Application execution parameter information.
   *
   * This method allows retrieving the `ApplicationBuildInfo` instance
   * that will be used when calling run().
   *
   * To be taken into account, this information must be modified
   * before calling run() or runDirect().
   */
  static ApplicationBuildInfo& applicationBuildInfo();

  /*!
   * \brief Information for '.Net' runtime initialization.
   *
   * To be taken into account, this information must be modified
   * before calling run() or rundDirect().
   */
  static DotNetRuntimeInitialisationInfo& dotNetRuntimeInitialisationInfo();

  /*!
   * \brief Information for accelerator initialization.
   *
   * To be taken into account, this information must be modified
   * before calling run() or rundDirect().
   */
  static AcceleratorRuntimeInitialisationInfo& acceleratorRuntimeInitialisationInfo();

  //! Full name of the directory where the executable is located
  static String getExeDirectory();

  /*!
   * \brief Creates a standalone implementation to manage accelerators.
   *
   * You must call init() before calling this method. The choice of
   * runtime (Arcane::Accelerator::eExecutionPolicy) is determined
   * by the arguments used when calling init() or specified via
   * acceleratorRuntimeInitialisationInfo() (see
   * \ref arcanedoc_parallel_accelerator_exec for more information)
   */
  static StandaloneAcceleratorMng createStandaloneAcceleratorMng();

  /*!
   * \brief Creates a standalone implementation to manage a subdomain.
   *
   * Only one instance of StandaloneSubDomain is allowed. Calling this
   * method more than once generates an exception.
   *
   * You must call init() before calling this method.
   *
   * If this method is called, you must not call other ArcaneLauncher execution
   * methods (for example ArcaneLauncher::run()).
   *
   * \a case_file_name is the name of the file containing the dataset. If null,
   * there is no dataset.
   */
  static StandaloneSubDomain createStandaloneSubDomain(const String& case_file_name);

  /*!
   * \brief Requests help with the "--help" or "-h" option.
   *
   * Method allowing to know if the user requested help with the "--help" or "-h" option.
   *
   * \return true if help was requested.
   */
  static bool needHelp();

  /*!
   * \brief Display of generic Arcane help.
   *
   * Method allowing to display generic Arcane help if the user requested it with
   * the "--help" or "-h" option.
   *
   * \return true if help was requested.
   */
  static bool printHelp();

 public:

  /*!
   * \deprecated
   */
  ARCCORE_DEPRECATED_2020("Use run(func) instead")
  static int runDirect(std::function<int(IDirectExecutionContext*)> func);

  /*!
   * \deprecated
   */
  ARCCORE_DEPRECATED_2020("Use init(args) instead")
  static void setCommandLineArguments(const CommandLineArguments& args)
  {
    init(args);
  }

 private:

  static void _initStandalone();
  static void _notifyRemoveStandaloneSubDomain();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
