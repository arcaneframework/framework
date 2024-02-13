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
    int _ExecuteTestDriver(string test_driver_path,string args)
    {
      if (Utils.IsWin32){
        return Utils.ExecCommandNoException(test_driver_path+".bat", args, null);
      }
      string cmd = test_driver_path + " " + args;
      return Utils.ExecShellCommandNoException(cmd, null);
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
        XAttribute expected_return_attr = command_elem.Attribute("expected-return-value");
        int expected_return_value = 0;
        if (expected_return_attr!=null)
          expected_return_value = int.Parse(expected_return_attr.Value);

        XName command_name = command_elem.Name;
        string local_name = command_name.LocalName;
        Console.WriteLine($"NAME = {local_name} expected_return={expected_return_value}");
        int return_value = 0;
        if (local_name == "test") {
          string args = command_elem.Value;
          string cmd_args = " launch " + args;
          return_value = _ExecuteTestDriver(test_driver_path,cmd_args);
        }
        else if (local_name == "driver") {
          string args = command_elem.Value;
          string cmd_args = args;
          return_value = _ExecuteTestDriver(test_driver_path,cmd_args);
        }
        else
          throw new ApplicationException($"Bad element {local_name}. Valid values are 'test' or 'driver'");

        if (return_value!=expected_return_value)
          throw new ApplicationException($"Bad return value '{return_value}'. Expected value is '{expected_return_value}'");
      }
      return 0;
    }
  }
}
