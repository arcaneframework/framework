//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;

namespace Arcane.ExecDrivers.Launcher
{
  public class Launcher
  {
    public static int MainExec(string[] args)
    {
      Arcane.ExecDrivers.Common.Utils.Init();

      int nb_arg = args.Length;
      if (nb_arg==0){
        Console.WriteLine("Usage: arcane_run [options] program [program_args]");
        return 0;
      }
      int r = 0;

      Arcane.ExecDrivers.Common.ExecDriver exec_driver = new Arcane.ExecDrivers.Common.ExecDriver();
      exec_driver.ParseArgs(args,null);
      r = exec_driver.Execute();

      return r;
    }
  }
}

