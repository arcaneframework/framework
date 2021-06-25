//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Xml;

namespace Axlstar.Driver
{
  //! Copie un fichier axl en intégrant les inclusions.
  public class MainClass
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
        Console.WriteLine("Usage: program axl2cc|axl2ccT4|axldoc|axlcopy [options]");
        return (-1);
      }
      List<string> l_remaining_args = new List<string>();
      // Le premier argument indique le nom de la commande
      for( int i=1; i<nb_arg; ++i )
        l_remaining_args.Add(args[i]);
      var remaining_args = l_remaining_args.ToArray();

      switch (args[0]) {
        case "axl2cc":
          return Axlstar.Axl.MainAXL2CC.MainExec(remaining_args);
        case "axl2ccT4":
          return Axlstar.Axl2ccT4.MainAxl2ccT4.MainExec(remaining_args);
        case "axldoc":
          return Arcane.AxlDoc.MainAXLDOC.MainExec(remaining_args);
        case "axlcopy":
          return Arcane.Axl.MainAXLCOPY.MainExec(remaining_args);
        default:
          throw new ApplicationException(String.Format("Bad option '{0}' for driver", args[0]));
      }
    }
  }

}
