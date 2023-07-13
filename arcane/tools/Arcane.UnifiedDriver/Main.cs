//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections.Generic;

namespace Arcane.UnifiedDriver
{
  class MainClass
  {
    static int Main(string[] args)
    {
      int nb_arg = args.Length;
      bool do_help = false;
      if (nb_arg>=1){
        string command_name = args[0];
        if (command_name=="-h" || command_name=="help" || command_name=="--help" || command_name=="-?")
          do_help = true;
      }
      if (nb_arg==0 || do_help){
        Console.WriteLine("Usage: program command [options]");
        return (-1);
      }
      List<string> l_remaining_args = new List<string>();
      // Le premier argument indique le nom de la commande
      for( int i=1; i<nb_arg; ++i )
        l_remaining_args.Add(args[i]);
      var remaining_args = l_remaining_args.ToArray();

      switch (args[0]) {
        case "dotnet_compile":
          return Arcane.DotNetCompile.MainClass.MainExec(remaining_args);
        case "template":
          return Arcane.Templates.MainClass.MainExec(remaining_args);
        case "mesh_utils":
          return Arcane.ExecDrivers.MeshUtilsDriver.MeshUtilsDriver.MainExec(remaining_args);
        case "launcher":
          return Arcane.ExecDrivers.Launcher.Launcher.MainExec(remaining_args);
        case "curve_utils":
          return Arcane.ExecDrivers.CurveUtilsDriver.CurveUtilsDriver.MainExec(remaining_args);
        case "test_driver":
          return Arcane.ExecDrivers.TestDriver.TestDriverMain.MainExec(remaining_args);
        default:
          throw new ApplicationException(String.Format("Bad option '{0}' for driver", args[0]));
      }
    }
  }
}

