//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.IO;
using System.Xml.Linq;
using Arcane.ExecDrivers.Common;

namespace Arcane.ExecDrivers.TestDriver
{
  public class ScriptDriver
  {
    string m_script_path;
    public void ParseArgs(string [] args)
    {
      int nb_arg = args.Length;
      if (nb_arg < 1) {
        Console.WriteLine("Bad number of args.");
        Console.WriteLine("Usage: [exe] script test_file");
      }
      m_script_path = args [0];
      Console.WriteLine("SCRIPT_PATH='{0}'", m_script_path);
    }
    void _ExecuteTestDriver(string test_driver_path,string args)
    {
      if (Utils.IsWin32){
        Utils.ExecCommand(test_driver_path+".bat", args, null);
      }
      else{
        string cmd = test_driver_path + " " + args;
        Utils.ExecShellCommand(cmd, null);
      }
    }

    public int Execute()
    {
      // Supprime l'éventuelle variable d'environnement qui nettoie
      // le répertoire de sortie car en général les exécutions successives
      // ont besoin des résultats de celles d'avant (par exemple les protections)
      Environment.SetEnvironmentVariable("ARCANE_TEST_CLEANUP_AFTER_RUN",null);

      XDocument doc = XDocument.Load(m_script_path);
      string test_driver_path = Path.Combine(Utils.CodeBinPath, "arcane_test_driver");
      foreach (XElement command_elem in doc.Document.Root.Elements()) {
        XName command_name = command_elem.Name;
        string local_name = command_name.LocalName;
        Console.WriteLine("NAME = {0}", local_name);
        if (local_name == "test") {
          string args = command_elem.Value;
          string cmd_args = " launch " + args;
          _ExecuteTestDriver(test_driver_path,cmd_args);
        }
        if (local_name == "driver") {
          string args = command_elem.Value;
          string cmd_args = args;
          _ExecuteTestDriver(test_driver_path,cmd_args);
        }
      }
      return 0;
    }
  }
}
