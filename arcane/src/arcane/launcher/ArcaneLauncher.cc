// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneLauncher.cc                                           (C) 2000-2025 */
/*                                                                           */
/* Class managing the execution launch.                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/launcher/ArcaneLauncher.h"

#include "arcane/launcher/IDirectExecutionContext.h"
#include "arcane/launcher/DirectSubDomainExecutionContext.h"
#include "arcane/launcher/GeneralHelp.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/JSONReader.h"
#include "arcane/utils/Exception.h"
#include "arcane/utils/ParameterList.h"
#include "arcane/utils/Ref.h"
#include "arcane/utils/ConcurrencyUtils.h"
#include "arcane/utils/internal/ParallelLoopOptionsProperties.h"
#include "arcane/utils/internal/ApplicationInfoProperties.h"
#include "arcane/core/internal/DotNetRuntimeInitialisationInfoProperties.h"
#include "arcane/accelerator/core/internal/AcceleratorRuntimeInitialisationInfoProperties.h"

#include "arccore/common/internal/Property.h"
#include "arccore/common/internal/ParameterListPropertyReader.h"
#include "arccore/common/internal/JSONPropertyReader.h"

#include "arcane/impl/ArcaneMain.h"
#include "arcane/impl/ArcaneSimpleExecutor.h"

#include "arcane/IDirectSubDomainExecuteFunctor.h"

#include <iomanip>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

namespace
{
bool global_has_init_done = false;
bool _checkInitCalled()
{
  if (!global_has_init_done){
    std::cerr << "ArcaneLauncher::init() has to be called before";
    return true;
  }
  return false;
}

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DirectExecutionContextImpl
: public IDirectExecutionContext
{
 public:
  explicit DirectExecutionContextImpl(ArcaneSimpleExecutor* simple_exec)
  : m_simple_exec(simple_exec) {}

 public:
  ISubDomain* createSequentialSubDomain() override
  {
    return createSequentialSubDomain(String());
  }
  ISubDomain* createSequentialSubDomain(const String& case_file_name) override
  {
    return m_simple_exec->createSubDomain(case_file_name);
  }
  ISubDomain* subDomain() const { return nullptr; }
 private:
  ArcaneSimpleExecutor* m_simple_exec;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
_checkReadConfigFile(StringView config_file_name)
{
  // TODO: in parallel, only one PE should perform the reading.
  if (config_file_name.empty())
    return;
  std::cout << "TRY_READING_CONFIG " << config_file_name << "\n";
  if (!platform::isFileReadable(config_file_name))
    return;
  UniqueArray<std::byte> bytes;
  bool is_bad = platform::readAllFile(config_file_name,false,bytes);
  if (is_bad)
    return;
  ApplicationInfo& app_info(ArcaneLauncher::applicationInfo());
  app_info.setRuntimeConfigFileContent(bytes);
  JSONDocument jdoc;
  jdoc.parse(bytes);
  JSONValue config = jdoc.root().child("configuration");
  if (config.null())
    return;
  std::cout << "READING CONFIG\n";
  properties::readFromJSON<ApplicationInfo,ApplicationInfoProperties>(config,app_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int ArcaneLauncher::
run()
{
  return ArcaneMain::run();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ApplicationInfo& ArcaneLauncher::
applicationInfo()
{
  return ArcaneMain::defaultApplicationInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DotNetRuntimeInitialisationInfo& ArcaneLauncher::
dotNetRuntimeInitialisationInfo()
{
  return ArcaneMain::defaultDotNetRuntimeInitialisationInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorRuntimeInitialisationInfo& ArcaneLauncher::
acceleratorRuntimeInitialisationInfo()
{
  return ArcaneMain::defaultAcceleratorRuntimeInitialisationInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ApplicationBuildInfo& ArcaneLauncher::
applicationBuildInfo()
{
  return ArcaneMain::defaultApplicationBuildInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String ArcaneLauncher::
getExeDirectory()
{
  String exe_full_path = platform::getExeFullPath();
  String exe_dir = platform::getFileDirName(exe_full_path);
  return exe_dir;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DirectCodeFunctor
: public IFunctor
{
 public:
  typedef std::function<int(IDirectExecutionContext*)> OldFunctorType;
  typedef std::function<int(DirectExecutionContext&)> FunctorType;
 public:
  DirectCodeFunctor(ArcaneSimpleExecutor* x,FunctorType* ft)
  : m_simple_executor(x), m_functor(ft){}
  DirectCodeFunctor(ArcaneSimpleExecutor* x,OldFunctorType* ft)
  : m_simple_executor(x), m_old_functor(ft){}
  void executeFunctor() override
  {
    DirectExecutionContextImpl direct_context_impl(m_simple_executor);
    if (m_functor){
      DirectExecutionContext direct_context(&direct_context_impl);
      m_return_value = (*m_functor)(direct_context);
    }
    else if (m_old_functor)
      m_return_value = (*m_old_functor)(&direct_context_impl);
  }
  int returnValue() const { return m_return_value; }
 public:
  ArcaneSimpleExecutor* m_simple_executor = nullptr;
  OldFunctorType* m_old_functor = nullptr;
  FunctorType* m_functor = nullptr;
  int m_return_value = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Obsolete
int ArcaneLauncher::
runDirect(std::function<int(IDirectExecutionContext* c)> func)
{
  if (_checkInitCalled())
    return (-1);
  int final_return = 0;
  {
    ArcaneSimpleExecutor simple_exec;
    int r = simple_exec.initialize();
    if (r!=0)
      return r;
    // Encapsulates the code in a functor that will handle exceptions. Without
    // this, in case of an exception and if the calling code does nothing, we
    // will have a call to std::terminate
    DirectCodeFunctor direct_functor(&simple_exec,&func);
    simple_exec.runCode(&direct_functor);
    final_return = direct_functor.returnValue();
  }
  return final_return;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int ArcaneLauncher::
run(std::function<int(DirectExecutionContext&)> func)
{
  if (_checkInitCalled())
    return (-1);

  int final_return = 0;
  {
    ArcaneSimpleExecutor simple_exec;
    int r = simple_exec.initialize();
    if (r!=0)
      return r;
    // Encapsulates the code in a functor that will handle exceptions. Without
    // this, in case of an exception and if the calling code does nothing, we
    // will have a call to std::terminate
    DirectCodeFunctor direct_functor(&simple_exec,&func);
    simple_exec.runCode(&direct_functor);
    final_return = direct_functor.returnValue();
  }
  return final_return;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ArcaneLauncherDirectExecuteFunctor
: public IDirectSubDomainExecuteFunctor
{
 public:
  ArcaneLauncherDirectExecuteFunctor(std::function<int(DirectSubDomainExecutionContext&)> func)
  : m_function(func){}
 public:
  int execute() override
  {
    if (!m_sub_domain)
      ARCANE_FATAL("Can not execute 'IDirectSubDomainExecuteFunctor' without sub domain");
    DirectSubDomainExecutionContext direct_context(m_sub_domain);
    return m_function(direct_context);
  }
  void setSubDomain(ISubDomain* sd) override { m_sub_domain = sd; }
 private:
  std::function<int(DirectSubDomainExecutionContext&)> m_function;
 public:
  ISubDomain* m_sub_domain = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int ArcaneLauncher::
run(std::function<int(DirectSubDomainExecutionContext&)> func)
{
  if (_checkInitCalled())
    return (-1);

  ArcaneLauncherDirectExecuteFunctor direct_exec(func);
  // In direct execution, by default there is no configuration file for the
  // code. If the user has not provided a configuration file, we set it to
  // the null string.
  String config_file = applicationBuildInfo().configFileName();
  // The default is the empty string. The null string explicitly indicates
  // that no configuration file is desired
  if (config_file.empty())
    applicationBuildInfo().setConfigFileName(String());
  int r = ArcaneMain::_internalRun(&direct_exec);
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
class MyVisitor
: public properties::IPropertyVisitor
{
 public:
  void visit(const properties::IPropertySetting* s) override
  {
    if (!s->commandLineArgument().null()){
      std::cout << "ARG:" << std::setw(30) << s->commandLineArgument()
                 << "  " << s->description() << "\n";
    }
  }
};

void
_listPropertySettings()
{
  using namespace Arcane::properties;
  MyVisitor my_visitor;
  visitAllRegisteredProperties(&my_visitor);
}
}

void ArcaneLauncher::
init(const CommandLineArguments& args)
{
  try{
    if (global_has_init_done)
      ARCANE_FATAL("ArcaneLauncher::init() has already been called");
    global_has_init_done = true;
    auto& application_info = applicationInfo();
    application_info.setCommandLineArguments(args);
    bool do_list = false;
    if (do_list)
      _listPropertySettings();
    const CommandLineArguments& cargs = applicationInfo().commandLineArguments();
    String runtime_config_file_name = cargs.getParameter("RuntimeConfigFile");
    if (!runtime_config_file_name.empty())
      _checkReadConfigFile(runtime_config_file_name);
    properties::readFromParameterList<ApplicationInfo,ApplicationInfoProperties>(args.parameters(),application_info);
    auto& dotnet_info = ArcaneLauncher::dotNetRuntimeInitialisationInfo();
    properties::readFromParameterList< DotNetRuntimeInitialisationInfo, DotNetRuntimeInitialisationInfoProperties>(args.parameters(),dotnet_info);
    auto& accelerator_info = ArcaneLauncher::acceleratorRuntimeInitialisationInfo();
    properties::readFromParameterList<AcceleratorRuntimeInitialisationInfo, Accelerator::AcceleratorRuntimeInitialisationInfoProperties>(args.parameters(), accelerator_info);
    ParallelLoopOptions loop_options;
    properties::readFromParameterList<ParallelLoopOptions,ParallelLoopOptionsProperties>(args.parameters(),loop_options);
    TaskFactory::setDefaultParallelLoopOptions(loop_options);
  }
  catch(const Exception& ex){
    cerr << ex << '\n';
    cerr << "** (ArcaneLauncher) Can't continue with the execution.\n";
    throw;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ArcaneLauncher::
isInitialized()
{
  return global_has_init_done;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneLauncher::
setDefaultMainFactory(IMainFactory* mf)
{
  ArcaneMain::setDefaultMainFactory(mf);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneLauncher::
_initStandalone()
{
  if (!global_has_init_done)
    ARCANE_FATAL("ArcaneLauncher::init() has to be called before");
  // This is necessary to potentially dynamically load the runtime associated
  // with accelerators
  ArcaneMain::_initRuntimes();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StandaloneAcceleratorMng ArcaneLauncher::
createStandaloneAcceleratorMng()
{
  _initStandalone();
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StandaloneSubDomain ArcaneLauncher::
createStandaloneSubDomain(const String& case_file_name)
{
  _initStandalone();
  StandaloneSubDomain s;
  s._initUniqueInstance(case_file_name);
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneLauncher::
_notifyRemoveStandaloneSubDomain()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ArcaneLauncher::
needHelp()
{
  return applicationInfo().commandLineArguments().needHelp();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ArcaneLauncher::
printHelp()
{
  if (applicationInfo().commandLineArguments().needHelp()) {
    GeneralHelp::printHelp();
    return true;
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
