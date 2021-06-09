//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections.Generic;

namespace Arcane.Templates
{
  public class MainClass
  {
    static public int MainExec(string[] args)
    {
      int nb_arg = args.Length;
      bool do_help = false;
      if (nb_arg>=1){
        string command_name = args[0];
        if (command_name=="-h" || command_name=="help" || command_name=="--help" || command_name=="-?")
          do_help = true;
      }
      if (nb_arg==0 || do_help){
        Console.WriteLine("Usage: arcane_templates generate-application [options]");
        return (-1);
      }
      List<string> l_remaining_args = new List<string>();

      for( int i=1; i<nb_arg; ++i )
        l_remaining_args.Add(args[i]);
      int r = 0;
      switch(args[0]){
      case "generate-application":
        var e = new GenerateApplication();
        r = e.Execute(l_remaining_args.ToArray());
        break;
      default:
        throw new ApplicationException(String.Format("Bad option '{0}' for driver. Valid value is 'generate-application'",args[0]));
      }
      return r;
    }
  }
}

