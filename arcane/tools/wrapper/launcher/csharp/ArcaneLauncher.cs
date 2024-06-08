//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using Arcane;

namespace Arcane
{
  class DirectSubDomainFunctor : IDirectSubDomainExecutionFunctor
  {
    public DirectSubDomainFunctor(ArcaneLauncher.DirectSubDomainExecutionContextDelegate d)
    {
      m_functor = d;
    }

    public override int Execute(DirectSubDomainExecutionContext ctx)
    {
      if (m_functor!=null)
        m_functor(ctx);
      return 0;
    }

    readonly ArcaneLauncher.DirectSubDomainExecutionContextDelegate m_functor;
  }

  class DirectFunctor : IDirectExecutionFunctor
  {
    public DirectFunctor(ArcaneLauncher.DirectExecutionContextDelegate d)
    {
      m_functor = d;
    }
    public override int Execute(DirectExecutionContext ctx)
    {
      if (m_functor!=null)
        m_functor(ctx);
      return 0;
    }

    readonly ArcaneLauncher.DirectExecutionContextDelegate m_functor;
  }

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
    public static ApplicationBuildInfo ApplicationBuildInfo
    {
      get { return ArcaneLauncher_INTERNAL.ApplicationBuildInfo(); }
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
#if ARCANE_HAS_DOTNET_PYTHON
      Arcane.Python.MainInit.Init();
#endif
    }

    public delegate int DirectSubDomainExecutionContextDelegate(DirectSubDomainExecutionContext ctx);

    public static int Run(DirectSubDomainExecutionContextDelegate d)
    {
      var x = new DirectSubDomainFunctor(d);
      return DirectExecutionWrapper.Run(x);
    }

    public delegate int DirectExecutionContextDelegate(DirectExecutionContext ctx);

    public static int Run(DirectExecutionContextDelegate d)
    {
      var x = new DirectFunctor(d);
      return DirectExecutionWrapper.Run(x);
    }
  }
}
