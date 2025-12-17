//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections.Generic;

namespace Arcane.ExecDrivers.Common
{
  //! Lecteur des arguments de la ligne de commande
  public class CommandArgs
  {
    public int NbProc;
    public int NbThreadPerProcess;
    public int NbTaskPerProcess;
    public int NbReplication;
    public int MaxIteration;
    public int NbContinue;

    public bool UseDotNet;
    public string DotNetRuntime = "coreclr";
    public string DotNetAssembly;
    public string DotNetUserCompile;
    public string DotNetUserOutputDll;
    public string[] ParallelArgs;

    public string DebugTool;

    public string[] RemainingArguments;
    public string ExecName;
    public string DirectExecMethod;
    public CommandArgs()
    {
      RemainingArguments = new string[0];
    }

    public void ParseArgs(string[] args, Mono.Options.OptionSet additional_options)
    {
      List<string> parallel_args = new List<string>();
      foreach (string s in args) {
        Console.WriteLine("PARSING ARG '{0}'", s);
      }
      List<string> additional_args = new List<string>();
      Mono.Options.OptionSet opt_set = new Mono.Options.OptionSet();
      bool show_help = false;
      NbTaskPerProcess = -1;
      opt_set.Add("h|help|?", "ce message d'aide", (v) => show_help = v != null);
      opt_set.Add("n|nb-proc=", "nombre de processeurs", (int v) => NbProc = v);
      opt_set.Add("T|threads=", "nombre de sous-domaines geres par les threads par processus", (int v) => NbThreadPerProcess = v);
      opt_set.Add("K|tasks=", "nombre de thread par processus pour gerer les taches", (int v) => NbTaskPerProcess = v);
      opt_set.Add("m|max-iteration=", "nombre d'itérations à effectuer", (int v) => MaxIteration = v);
      opt_set.Add("c|nb-continue=", "nombre de reprises", (int v) => NbContinue = v);
      opt_set.Add("C|build-config=", "configuration de build à utiliser (Debug, Release, ...)", (string v) => Utils.OutDir = v);
      opt_set.Add("Z|dotnet", "indique un cas .NET", (v) => UseDotNet = true);
      opt_set.Add("dotnet-runtime=", "nom du runtime '.Net' à utiliser (mono|coreclr)", (string v) => DotNetRuntime = v);
      opt_set.Add("dotnet-assembly=", "nom complet de l'assembly '.Net' à charger au démarrage", (string v) => DotNetAssembly = v);
      opt_set.Add("dotnet-compile=", "liste de fichiers C# à compiler", (string v) => DotNetUserCompile = v);
      opt_set.Add("dotnet-output-dll=", "chemin et nom du fichier .dll généré", (string v) => DotNetUserOutputDll = v);
      opt_set.Add("P|parallelarg=", "option passee au gestionnaire parallele", (string v) => parallel_args.Add(v));
      opt_set.Add("d|debugtool=", "outil de debug (tv,gdb,memcheck,...)", (string v) => DebugTool = v);
      opt_set.Add("R|replication=", "nombre de replication de sous-domaines", (int v) => NbReplication = v);
      opt_set.Add("D|direct-method=", "méthode à lancer en exécution directe",(string v) => DirectExecMethod = v);
      opt_set.Add("E|exec-name=", "nom de l'exécutable de test (sans extension ni chemin)", (string v) => ExecName = v);
      opt_set.Add("W=", "paramètres spécifiques:\n-We,VARIABLE,VALUE pour spécifier" +
                  "la variable d'environnement VARIABLE avec la valeur VALUE.\n" +
                  "-Wp,ARG1,ARG2,... pour spécifier des paramètres pour le lanceur parallèle", (string v) => additional_args.Add(v));
      if (additional_options != null) {
        foreach (Mono.Options.Option opt in additional_options)
          opt_set.Add(opt);
      }
      List<string> remaining_args = opt_set.Parse(args);
      foreach (string s in remaining_args) {
        Console.WriteLine("REMAINING ARG '{0}'", s);
      }

      RemainingArguments = remaining_args.ToArray();

      if (show_help) {
        Console.WriteLine("Options:");
        opt_set.WriteOptionDescriptions(Console.Out);
        Environment.Exit(0);
      }
      string env_parallel_args = Utils.GetEnvironmentVariable("ARCANE_PARALLEL_ARGS");
      if (!String.IsNullOrEmpty(env_parallel_args)) {
        parallel_args.AddRange(env_parallel_args.Split(new char[] { ' ' }));
      }
      foreach (string s in additional_args) {
        Console.WriteLine("ADDITIONAL_ARG = '{0}'", s);
        string[] vals = s.Split(',');
        if (vals.Length <= 1) {
          Console.WriteLine("Invalid value for argument '-W' : '{0}'", s);
          continue;
        }
        string val0 = vals[0];
        if (val0.Length != 1) {
          Console.WriteLine("Invalid value '{0}' in '{1}'", val0, s);
        }
        char t = val0[0];
        if (t == 'e') {
          if (vals.Length != 3) {
            Console.WriteLine("Invalid option '{0}' : valid form is e,VARIABLE,VALUE", s);
          }
          else {
            Utils.SetEnvironmentVariable(vals[1], vals[2]);
          }
        }
        else if (t == 'p') {
          for (int z = 1; z < vals.Length; ++z)
            parallel_args.Add(vals[z]);
        }
      }

      ParallelArgs = parallel_args.ToArray();
      foreach (string s in remaining_args) {
        Console.WriteLine("REMAINING ARG (2) '{0}'", s);
      }
    }
  }
}
