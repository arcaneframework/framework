//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Collections;
using System.Reflection;

namespace Arcane.ExecDrivers.TestDriver
{
  public class TestDriverMain
  {
    static readonly bool g_print_var = false;
    public static int MainExec(string[] args)
    {
      Arcane.ExecDrivers.Common.Utils.Init();

      int nb_arg = args.Length;
      if (nb_arg == 0) {
        Console.WriteLine("Usage: program (launch|compare) [options]");
        return 0;
      }
      if (g_print_var){
        Console.WriteLine();
        Console.WriteLine("GetEnvironmentVariables: ");
        foreach (DictionaryEntry de in Environment.GetEnvironmentVariables())
          Console.WriteLine("  {0} = {1}", de.Key, de.Value);
      }

      List<string> l_remaining_args = new List<string>();
      for (int i = 1; i < nb_arg; ++i)
        l_remaining_args.Add(args[i]);
      int r = 0;
      string[] remaining_args = l_remaining_args.ToArray();
      switch (args[0]) {
        case "launch":
          var test_driver = new Arcane.ExecDrivers.Common.ExecDriver();
          test_driver.ParseArgs(remaining_args, null);
          r = test_driver.Execute();
          if (r==0)
            test_driver.Cleanup();
          break;
        case "compare":
          var compare_driver = new VariableComparerDriver();
          compare_driver.ParseArgs(remaining_args);
          r = compare_driver.Execute();
          break;
        case "script":
          var script_driver = new ScriptDriver();
          script_driver.ParseArgs(remaining_args);
          r = script_driver.Execute();
          break;
        default:
          throw new ApplicationException($"Bad option '{args[0]}' for driver. Valid values are 'launch', 'compare' or 'script'");
      }
      return r;
    }
  }
}
