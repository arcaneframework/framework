//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using Arcane.ExecDrivers.Common;

namespace Arcane.DotNetCompile
{
  public class MainClass
  {
    static public int MainExec(string[] args)
    {
      Utils.Init();

      int nb_arg = args.Length;
      bool do_help = false;

      if (nb_arg>=1){
        string command_name = args[0];
        if (command_name=="-h" || command_name=="help" || command_name=="--help" || command_name=="-?")
          do_help = true;
      }

      if (nb_arg==0 || do_help){
        Console.WriteLine("Usage: arcane_dotnet_compile [csfiles]");
        return (-1);
      }

      var e = new Compile();
      int ret_value = e.Execute(args);

      return ret_value;
    }
  }
}

