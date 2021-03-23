//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using Arcane;
using Arcane.Launcher;

namespace Arcane
{
  public class ArcaneLauncher
  {
    public static int Run()
    {
      return ArcaneMain.Run();
    }
    public static ApplicationInfo ApplicationInfo
    {
      get { return ArcaneLauncher_INTERNAL.ApplicationInfo(); }
    }
    public static DotNetRuntimeInitialisationInfo DotNetRuntimeInitialisationInfo
    {
      get { return ArcaneLauncher_INTERNAL.DotNetRuntimeInitialisationInfo(); }
    }
    [Obsolete("Use Init(args) instead")]
    public static void SetCommandLineArguments(CommandLineArguments args)
    {
      ArcaneLauncher_INTERNAL.Init(args);
    }
    public static void Init(CommandLineArguments args)
    {
      ArcaneLauncher_INTERNAL.Init(args);
    }
  }
}
