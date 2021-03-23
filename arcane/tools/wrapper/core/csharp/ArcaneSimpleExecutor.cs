//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;

namespace Arcane
{
  public class ArcaneSimpleExecutor : IDisposable
  {
    public delegate int ExecFunctor(ArcaneSimpleExecutor executor);

    public static int Run(ExecFunctor exec_functor)
    {
      return Run(null,exec_functor);
    }

    public static int Run(ExecFunctor init_functor,ExecFunctor exec_functor)
    {
      Console.WriteLine("ArcaneTest.Launcher.ExecDirect");
      ArcaneMain.Initialize();

      int r = 0;
      using(var simple_exec = new ArcaneSimpleExecutor()){
        if (init_functor!=null){
          r = init_functor.Invoke(simple_exec);
          if (r!=0)
            return r;
        }
        r = simple_exec.Initialize();
        if (r!=0){
          Console.Error.WriteLine("Error during ArcaneSimpleExecutor.Initialize() R={0}",r);
          return r;
        }
        if (exec_functor!=null)
          r = exec_functor.Invoke(simple_exec);
      }
      ArcaneMain.Cleanup();
      return r;
    }

    ArcaneSimpleExecutor_INTERNAL m_internal_executor = new ArcaneSimpleExecutor_INTERNAL();

    public void Dispose()
    {
      if (m_internal_executor!=null){
        m_internal_executor.Dispose();
        m_internal_executor = null;
      }
    }

    public int Initialize()
    {
      return m_internal_executor.Initialize();
    }

    public ISubDomain CreateSubDomain()
    {
      return m_internal_executor.CreateSubDomain(null);
    }
  }
}
