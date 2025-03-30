//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.IO;
using System.Collections.Generic;
using Arcane.ExecDrivers.Common;

namespace Arcane.ExecDrivers.MeshUtilsDriver
{
  public class MeshUtilsDriver
  {
    internal class CasePartitionDriver
    {
      void _ErrorArg(string msg)
      {
        Console.WriteLine("ERROR: {0}", msg);
        Environment.Exit(1);
      }
      public int Execute(List<string> remaining_args)
      {
        // Il faut mettre les arguments additionels au debut pour etre sur qu'il
        // se trouve avant le fichier de maillage a partitionner (qui doit etre le dernier argument)
        string test_path = Arcane.ExecDrivers.Common.Utils.GetTestPath();
        string exec_name = Path.Combine(test_path, "arcane_driver");
        remaining_args.InsertRange(0,new string[]{ "--exec-name", exec_name});
        ExecDriver exec_driver = new ExecDriver();
        Mono.Options.OptionSet opt_set = new Mono.Options.OptionSet();
        int nb_part = 0;
        int nb_ghost = 0;
        bool generate_correspondance_file = false;
        string partitioner_name = "DefaultPartitioner";
        string mesh_writer_name = Utils.ReadConfig("DefaultMeshWriter");
        string output_file_pattern = null;
        List<string> constrained_groups = new List<string>();
        opt_set.Add("p|parties|nb-part=", "nombre de parties a decouper", (int v) => nb_part = v);
        opt_set.Add("f|fantomes|nb-ghost-layer=", "nombre de couches de mailles fantomes", (int v) => nb_ghost = v);
        opt_set.Add("correspondance", "genere le fichier de correspondance", (string v) => generate_correspondance_file = (v != null));
        opt_set.Add("I|indivisible=", "groupe d'entites indivisibles", (string v) => constrained_groups.Add(v));
        opt_set.Add("A|algorithme=", "nom du partitionneur a utiliser (Metis, Zoltan ou PTScotch)", (string v) => partitioner_name = v);
        opt_set.Add("writer|ecrivain=", "nom du service pour l'ecriture des maillages decoupes", (string v) => mesh_writer_name = v);
        opt_set.Add("output-file-pattern=", "file pattern for output file (default to CPU%05d)", (string v) => output_file_pattern = v);
        exec_driver.OnAddAdditionalArgs += delegate (ExecDriver d)
        {
          // Si le nombre de parties n'est pas specifie, il est egal au nombre de processeurs
          if (nb_part == 0)
            nb_part = d.NbProc;
          if (d.NbProc < 2)
          {
            _ErrorArg(String.Format("le nombre de processeurs (option -n) doit etre superieur a 1 (actuellement {0})", d.NbProc));
          }
          d.AdditionalArgs.AddRange(new string[] { "-arcane_opt", "direct_exec", "ArcaneCasePartitioner" });
          _AddArg(d, "nb-ghost-layer", nb_ghost.ToString());
          _AddArg(d, "create-correspondances", generate_correspondance_file ? "1" : "0");
          _AddArg(d, "library", partitioner_name);
          _AddArg(d, "nb-cut-part", nb_part.ToString());
          if (!String.IsNullOrEmpty(output_file_pattern))
          {
            _AddArg(d, "mesh-file-name-pattern", output_file_pattern);
          }
          foreach (string s in constrained_groups)
          {
            d.AdditionalArgs.AddRange(new string[] { "-arcane_opt", "tool_arg", "constraints", s });
          }
          if (!String.IsNullOrEmpty(mesh_writer_name))
          {
            _AddArg(d, "writer-service-name", mesh_writer_name);
          }
        };
        exec_driver.ParseArgs(remaining_args.ToArray(), opt_set);
        return exec_driver.Execute();
      }

      void _AddArg(ExecDriver d, string parameter, string param_value)
      {
        d.AdditionalArgs.AddRange(new string[] { "-arcane_opt", "tool_arg", parameter, param_value });
      }
    }

    internal class MeshConvertDriver
    {
      void _ErrorArg(string msg)
      {
        Console.WriteLine("ERROR: {0}", msg);
        Console.WriteLine("Usage: program --file output_file --writer write_service input_file");
        Environment.Exit(1);
      }
      public int Execute(List<string> remaining_args)
      {
        // Il faut mettre les arguments additionels au debut pour etre sur qu'ils
        // se trouvent avant le fichier de maillage à convertir (qui doit etre le dernier argument)
        string test_path = Arcane.ExecDrivers.Common.Utils.GetTestPath();
        string exec_name = Path.Combine(test_path, "arcane_driver");
        remaining_args.InsertRange(0,new string[]{ "--exec-name", exec_name});
        ExecDriver exec_driver = new ExecDriver();
        Mono.Options.OptionSet opt_set = new Mono.Options.OptionSet();

        string mesh_writer_name = Utils.ReadConfig("DefaultMeshWriter");
        string output_file_name = null;
        opt_set.Add("f|fichier|file=", "nom du fichier de sortie", (string v) => output_file_name = v);
        opt_set.Add("ecrivain|writer=", "nom du service pour l'écriture du maillage", (string v) => mesh_writer_name = v);
        exec_driver.OnAddAdditionalArgs += delegate (ExecDriver d)
        {
          d.AdditionalArgs.AddRange(new string[] { "-arcane_opt", "direct_exec", "ArcaneMeshConverter" });
          if (!String.IsNullOrEmpty(mesh_writer_name))
            _AddArg(d, "writer-service-name", mesh_writer_name);
          if (String.IsNullOrEmpty(output_file_name))
            _ErrorArg(String.Format("Name of output file (-f|--file) is not specified"));
          _AddArg(d, "file-name", output_file_name);
          if (exec_driver.RemainingArgs.Length == 0)
            _ErrorArg(String.Format("Name of input file is not specified"));
        };
        exec_driver.ParseArgs(remaining_args.ToArray(), opt_set);
        return exec_driver.Execute();
      }

      void _AddArg(ExecDriver d, string parameter, string param_value)
      {
        d.AdditionalArgs.AddRange(new string[] { "-arcane_opt", "tool_arg", parameter, param_value });
      }
    }

    public static int MainExec(string[] args)
    {
      Arcane.ExecDrivers.Common.Utils.Init();

      int nb_arg = args.Length;
      if (nb_arg == 0)
      {
        Console.WriteLine("Usage: program partition|convert [options]");
        return (-1);
      }
      List<string> l_remaining_args = new List<string>();

      for (int i = 1; i < nb_arg; ++i)
        l_remaining_args.Add(args[i]);
      int r = 0;
      switch (args[0])
      {
        case "partition":
          CasePartitionDriver e = new CasePartitionDriver();
          r = e.Execute(l_remaining_args);
          break;
        case "convert":
          MeshConvertDriver e2 = new MeshConvertDriver();
          r = e2.Execute(l_remaining_args);
          break;
        default:
          throw new ApplicationException(String.Format("Bad option '{0}' for driver. Valid values are 'partition' or 'convert'", args[0]));
      }
      return r;
    }
  }
}
