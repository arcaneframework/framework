// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include "arcane/utils/PlatformUtils.h"

#include "arcane/launcher/ArcaneLauncher.h"

#include "arcane/impl/ArcaneMain.h"

// Pour détecter si Hdf5 est disponible.
#include "arcane_packages.h"

// Pour forcer enregistrement de 'arcane_std'
#include "arcane/std/ArcaneStdRegisterer.h"

#ifdef ARCANE_HAS_PACKAGE_HDF5
#include "arcane/hdf5/ArcaneHdf5Registerer.h"
#endif

// La fonction doit être 'C' pour pouvoir être appelée depuis le C#.
extern "C" ARCANE_EXPORT void
arcaneTestSetApplicationInfo()
{
  using namespace Arcane;

  // Force le chargement de 'arcane_std'
  ArcaneStdRegisterer::registerLibrary();
#ifdef ARCANE_HAS_PACKAGE_HDF5
  ArcaneHdf5Registerer::registerLibrary();
#endif

  ApplicationBuildInfo& app_build_info = ArcaneLauncher::applicationBuildInfo();
  app_build_info.setApplicationName("ArcaneTest");
  app_build_info.setCodeName("ArcaneTest");
  app_build_info.setCodeVersion(VersionInfo(1,0,0));

  app_build_info.addDynamicLibrary("arcane_driverlib");
  app_build_info.addDynamicLibrary("arcane_geometry");
#ifdef ARCANE_OS_WIN32
  app_build_info.addDynamicLibrary("PerfectGas");
  app_build_info.addDynamicLibrary("StiffenedGas");
  app_build_info.addDynamicLibrary("arcane_cea_tests");
  app_build_info.addDynamicLibrary("arcane_aleph_tests");
  app_build_info.addDynamicLibrary("arcane_aleph_hypre");
  app_build_info.addDynamicLibrary("arcane_aleph_kappa");
#endif

  // Modifie le répertoire de sortie pour prendren en compte le nom du
  // test s'il est défini.
  String test_name = platform::getEnvironmentVariable("ARCANE_TEST_NAME");
  if (!test_name.empty())
    app_build_info.setOutputDirectory(String("test_output_")+test_name);
  ArcaneMain::setUseTestLogger(true);
}

extern "C++" ARCANE_EXPORT void
arcaneTestSetDotNetInitInfo()
{
  using namespace Arcane;

  auto& dotnet_info = ArcaneLauncher::dotNetRuntimeInitialisationInfo();

  String dotnet_assembly = platform::getEnvironmentVariable("ARCANE_DOTNET_ASSEMBLY");
  if (!dotnet_assembly.null())
    dotnet_info.setMainAssemblyName(dotnet_assembly);

  String dotnet_class = platform::getEnvironmentVariable("ARCANE_DOTNET_CLASS");
  if (!dotnet_class.null())
    dotnet_info.setExecuteClassName(dotnet_class);

  String dotnet_method = platform::getEnvironmentVariable("ARCANE_DOTNET_METHOD");
  if (!dotnet_method.null())
    dotnet_info.setExecuteMethodName(dotnet_method);

  String dotnet_runtime = platform::getEnvironmentVariable("ARCANE_DOTNET_RUNTIME");
  if (!dotnet_runtime.null())
    dotnet_info.setEmbeddedRuntime(dotnet_runtime);
}
