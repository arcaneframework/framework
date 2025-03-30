//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text;

namespace Arcane.ExecDrivers.Common
{
  public interface ICustomExecDriver
  {
    string Name { get; }
    /// <summary>
    /// Tente de traiter l'execution avec le lanceur MPI. Retourne true si traitement effectue, false sinon.
    /// L'appelé a le droit de modifier dans l'argument p les champs 'MpiLauncherArgs' et 'MpiLauncher' ainsi
    /// que 'UseTotalview' et 'UseDdt'.
    /// </summary>
    /// <param name="p">
    /// A <see cref="ExecDriverProperties"/>
    /// </param>
    /// <returns>
    /// A <see cref="System.Boolean"/>
    /// </returns>
    bool HandleMpiLauncher(ExecDriverProperties p, string mpi_launcher);
  }

  public class ExecDriver
  {
    public int NbProc { get { return m_properties.NbProc; } }

    ExecDriverProperties m_properties;
    public ExecDriverProperties Properties { get { return m_properties; } }

    string[] m_remaining_args;
    public string[] RemainingArgs { get { return m_remaining_args; } }

    bool m_use_dotnet;
    string m_dotnet_runtime;
    string m_dotnet_assembly;
    string m_dotnet_compile;
    string m_dotnet_output_dll;
    string[] m_parallel_args;
    CommandArgs m_command_args;
    List<Assembly> m_additional_assemblies;
    ICustomExecDriver m_custom_driver;

    public delegate void AddAdditionalArgsCallback(ExecDriver driver);

    //! Arguments additionnels. Ils doivent être positionnés lors de l'évènement OnAddAdditionalArgs.
    public List<string> AdditionalArgs;

    public event AddAdditionalArgsCallback OnAddAdditionalArgs;

    public ExecDriver()
    {
      m_properties = new ExecDriverProperties();
      m_additional_assemblies = new List<Assembly>();
      m_custom_driver = new CustomMpiDriver();
      Console.WriteLine("Custome driver ={0}", m_custom_driver);
    }

    public void ParseArgs(string[] args, Mono.Options.OptionSet additional_options)
    {
      AdditionalArgs = new List<string>();
      CommandArgs command_args = new CommandArgs();
      m_command_args = command_args;
      command_args.ParseArgs(args, additional_options);

      m_properties.NbProc = command_args.NbProc;
      m_properties.NbIteration = command_args.MaxIteration;
      m_properties.NbContinue = command_args.NbContinue;
      m_properties.NbReplication = command_args.NbReplication;
      m_properties.DirectExecMethod = command_args.DirectExecMethod;

      m_use_dotnet = command_args.UseDotNet;
      m_dotnet_runtime = command_args.DotNetRuntime;
      m_dotnet_assembly = command_args.DotNetAssembly;
      m_dotnet_compile = command_args.DotNetUserCompile;
      m_dotnet_output_dll = command_args.DotNetUserOutputDll;

      m_remaining_args = command_args.RemainingArguments;
      if (!String.IsNullOrEmpty(command_args.ExecName))
        m_properties.ExecName = command_args.ExecName;
      if (String.IsNullOrEmpty(m_properties.ExecName)) {
        Console.WriteLine("No exec file given on command line");
        Environment.Exit(1);
      }
      m_parallel_args = command_args.ParallelArgs;

      m_properties.NbSharedMemorySubDomain = command_args.NbThreadPerProcess;
      m_properties.NbTaskPerProcess = command_args.NbTaskPerProcess;
    }
    void _AddArcaneArg(List<string> args,string name, object value)
    {
      args.Add(name + "=" + value.ToString());
    }
    /// <summary>
    /// Lance l'execution. Suivant les options specifiees en ligne de commande,
    /// l'executable est lance en parallele ou sequentiel, avec ou sans debug.
    /// </summary>
    public int Execute()
    {
      const string ARG_NB_TASK = "T";
      const string ARG_NB_REPLICATION = "R";
      const string ARG_NB_SHAREDMEMORY_SUBDOMAIN = "S";
      List<string> arcane_args = new List<string>();
      string exe_file_name = m_properties.ExecName;
      Console.WriteLine("ExecName={0}", exe_file_name);
      Console.WriteLine("ProcessInfo: Is64Bit={0}", Environment.Is64BitProcess);
      if (Utils.IsWin32) {
        // Ajoute extension '.exe' si besoin
        string extension = Path.GetExtension(exe_file_name);
        if (extension != ".exe")
          exe_file_name = exe_file_name + ".exe";
      }
      // Positionne le nom complet de l'exécutable avec le chemin.
      string test_path = Arcane.ExecDrivers.Common.Utils.GetTestPath();
      exe_file_name = Path.Combine(test_path, exe_file_name);
      string outdir = Utils.OutDir;

      Console.WriteLine("OutDir={0}", outdir);
      string lib_path = Utils.CodeLibPath;
      string share_path = Utils.CodeSharePath;
      string exe_name = exe_file_name;
      Console.WriteLine("EXEC: {0}", exe_name);

      // Compile les fichiers C# spécifiés
      if (!String.IsNullOrEmpty(m_dotnet_compile)){
        var x = new Arcane.ExecDrivers.DotNetCompile.Compile();
        if (!String.IsNullOrEmpty(m_dotnet_output_dll)){
          FileInfo fif = new FileInfo(m_dotnet_output_dll);
          Console.WriteLine("OUTPUT DLL: {0}", fif.FullName);
          DirectoryInfo di = new DirectoryInfo(fif.Directory.FullName);
          if (!di.Exists){
            di.Create();
          }
          x.Execute(new string[]{"/out:" + fif.FullName, m_dotnet_compile});
        }
        else{
          x.Execute(new string[]{m_dotnet_compile});
        }
      }
      if (!String.IsNullOrEmpty(m_properties.DirectExecMethod))
        _AddArcaneArg(arcane_args, "DirectExecutionMethod", m_properties.DirectExecMethod);
      if (m_properties.NbTaskPerProcess >= 0) {
        _AddArcaneArg(arcane_args, ARG_NB_TASK, m_properties.NbTaskPerProcess);
      }
      if (m_properties.NbSharedMemorySubDomain != 0) {
        Console.WriteLine("Using shared_memory_sub_domain n={0}", m_properties.NbSharedMemorySubDomain);
        _AddArcaneArg(arcane_args, ARG_NB_SHAREDMEMORY_SUBDOMAIN, m_properties.NbSharedMemorySubDomain);
      }
      else if (m_properties.NbProc != 0) {
      }
      if (m_properties.NbReplication != 0) {
        _AddArcaneArg(arcane_args, ARG_NB_REPLICATION, m_properties.NbReplication);
      }
      if (Utils.IsWin32) {
        string path = Utils.GetEnvironmentVariable("PATH");
        // Ajoute le repertoire lib/${OutDir}
        string lib_dir = lib_path;
        if (!String.IsNullOrEmpty(outdir))
          lib_dir = Path.Combine(lib_path, outdir);
        //Utils.SetEnvironmentVariable("STDENV_PATH_LIB", lib_dir);
        path = lib_dir + ";" + path;
        lib_dir = Path.Combine(lib_path, "sys_dll");
        path = lib_dir + ";" + path;
        Utils.SetEnvironmentVariable("PATH", path);
      }
      else {
        // Ne positionne pas de variable d'environnement sous unix.
      }
      bool do_add_ld_library_path = false;
      if (do_add_ld_library_path) {
        string ld_library_path = Utils.GetEnvironmentVariable("LD_LIBRARY_PATH");
        if (String.IsNullOrEmpty(ld_library_path)) {
          ld_library_path = lib_path;
        }
        else {
          // Il vaut mieux laisser les libs systemes en premier
          // et ajouter nos libs a la fin du LD_LIBRARY_PATH
          ld_library_path = ld_library_path + ":" + lib_path;
        }
        Console.WriteLine("SET LD_LIBRARY_PATH to '{0}'", ld_library_path);
        Utils.SetEnvironmentVariable("LD_LIBRARY_PATH", ld_library_path);
      }
      //bool use_totalview = false;
      bool use_gdb = false;
      bool use_memcheck = false;
      bool is_parallel = false;
      bool use_helgrind = false;
      bool use_massif = false;
      bool use_devenv = false;
      bool use_vtune = false;
      string debugger = Utils.GetEnvironmentVariable("ARCANE_DEBUGGER");
      if (String.IsNullOrEmpty(debugger))
        debugger = m_command_args.DebugTool;

      if (debugger == "tv" || debugger == "totalview")
        m_properties.UseTotalview = true;

      if (debugger == "gdb") {
        Console.WriteLine("Using 'gdb'");
        use_gdb = true;
      }
      if (debugger == "ddt") {
        Console.WriteLine("Using ddt");
        m_properties.UseDdt = true;
      }
      if (debugger == "memcheck") {
        Console.WriteLine("Using 'memcheck'");
        use_memcheck = true;
      }
      if (debugger == "vtune") {
        Console.WriteLine("Using 'vtune'");
        use_vtune = true;
      }
      if (debugger == "helgrind") {
        Console.WriteLine("Using 'helgrind'");
        use_helgrind = true;
      }
      if (debugger == "massif") {
        Console.WriteLine("Using 'massif'");
        use_massif = true;
      }
      if (debugger == "vs") {
        Console.WriteLine("Using visual studio debugger");
        use_devenv = true;
      }

      if (m_properties.NbProc != 0)
        is_parallel = true;
      // Regarde si on force l'utilisation du lanceur MPI (en général mpixec) même pour les jobs séquentiels
      // Cela peut être nécessaire sur certaines plateformes
      // On le fait si demandé via une variable d'environnement ou si
      // on utilise un driver spécifique.
      bool force_use_mpi_driver = !String.IsNullOrEmpty(Utils.CustomMpiDriver);
      {
        string str = Utils.GetEnvironmentVariable("ARCANE_ALWAYS_USE_MPI_DRIVER");
        if (str=="1" || str=="TRUE")
          force_use_mpi_driver = true;
        if (str=="0" || str=="FALSE")
          force_use_mpi_driver = false;
      }
      if (m_properties.NbProc==0 && force_use_mpi_driver){
        Console.WriteLine("Force using mpi driver to launch sequential test");
        m_properties.NbProc = 1;
      }
      if (!String.IsNullOrEmpty(m_dotnet_assembly)){
        m_use_dotnet = true;
        Utils.SetEnvironmentVariable("ARCANE_DOTNET_ASSEMBLY", m_dotnet_assembly);
      }
      bool want_dotnet_shared = true;
      string shared_dotnet_str = Utils.GetEnvironmentVariable("ARCANE_USE_DOTNET_WRAPPER");
      if (shared_dotnet_str == "1" || shared_dotnet_str == "TRUE") {
        Console.WriteLine($"Using '.Net' wrapper '{m_dotnet_runtime}' for launching C# code");
        Utils.SetEnvironmentVariable("ARCANE_DOTNET_RUNTIME",m_dotnet_runtime);
        want_dotnet_shared = false;
      }
      string orig_exe_name = exe_name;
      for (int current_exec = 0; current_exec < (m_properties.NbContinue + 1); ++current_exec) {
        string command = "";
        //string args = "";
        List<string> args = new List<string>();
        List<string> sub_args = new List<string>();
        if (use_memcheck) {
          string[] valgrind_args = new string[]{ "--tool=memcheck",
                                                 "-v",
                                                 "--leak-check=full",
                                                 "--track-origins=yes",
                                                 "--num-callers=15",
                                                 "--show-reachable=yes"
          };

          sub_args.AddRange(valgrind_args);
          if (is_parallel)
            sub_args.Add("--log-file=valgrind_out.%p");
          sub_args.Add(orig_exe_name);
          exe_name = Utils.ValgrindExecName; //"valgrind";
        }
        if (use_helgrind) {
          string[] valgrind_args = new string[]{ "--tool=helgrind",
                                                 "-v",
                                                 "--num-callers=15"
         };

          sub_args.AddRange(valgrind_args);
          if (is_parallel)
            sub_args.Add("--log-file=valgrind_out.%p");
          sub_args.Add(exe_name);
          exe_name = Utils.ValgrindExecName;
        }
        if (use_vtune) {
          string[] vtune_args = new string[]{
            "-collect",
            "hotspots",
            "-follow-child",
            "-mrte-mode=auto",
            "-target-duration-type=short",
            "-no-allow-multiple-runs",
            "-no-analyze-system",
            "-data-limit=100",
            "-slow-frames-threshold=40",
            "-fast-frames-threshold=100"
          };

          sub_args.AddRange(vtune_args);
          sub_args.Add("--");
          sub_args.Add(exe_name);
          exe_name = "amplxe-cl";
        }

        if (use_massif) {
          string[] valgrind_args = new string[]{ "--tool=massif",
                                                 "-v",
                                                 "--num-callers=15"
          };

          sub_args.AddRange(valgrind_args);
          if (is_parallel)
            sub_args.Add("--log-file=valgrind_out.%p");
          sub_args.Add(exe_name);
          exe_name = Utils.ValgrindExecName;
        }

        if (m_properties.NbProc == 0) {
          command = exe_name;
          _HandleMpiLauncher();
          if (sub_args != null)
            args.AddRange(sub_args);
        }
        else {
          _HandleMpiLauncher();
          string mpi_exec_name = m_properties.MpiLauncher;
          if (string.IsNullOrEmpty(mpi_exec_name))
            mpi_exec_name = Utils.MpiExecName;
          command = mpi_exec_name;
          foreach (string s in m_properties.MpiLauncherArgs) {
            args.Add(s);
          }
          if (m_parallel_args != null) {
            foreach (string s in m_parallel_args) {
              args.Add(s);
            }
          }
          if (m_use_dotnet && want_dotnet_shared) {
            string test_exe_dll_path = Path.Combine(lib_path, "ArcaneTestExe.dll");
            if (m_dotnet_runtime=="coreclr"){
              string coreclr_bin = Utils.DotnetCoreClrPath;
              exe_name = $"\"{coreclr_bin}\" {test_exe_dll_path}";
            }
            else
             throw new ArgumentException("Invalid value for option 'dotnet-runtime'. Valid value is 'coreclr'");
          }

          args.Add(exe_name);
          if (sub_args != null)
            args.AddRange(sub_args);
        }
        if (arcane_args.Count != 0) {
          string x = "-A," + String.Join(",", arcane_args.ToArray());
          args.Add(x);
          Console.WriteLine("ArgsForArcane: {0}", x);
        }
        if (m_properties.NbIteration >= 1) {
          args.Add("-A,MaxIteration="+m_properties.NbIteration.ToString());
        }
        if (current_exec != 0) {
          args.Add("-arcane_opt");
          args.Add("continue");
        }
        // En séquentiel pure, ne charge pas l'environnement MPI.
        // Cela permet de gagner du temps lors de l'initialisation et de permettre
        // de lancer directement l'exécutable sans passer par 'mpiexec'.
        if (m_properties.NbProc==0 && m_properties.NbSharedMemorySubDomain==0){
          string env_parallel_service = Utils.GetEnvironmentVariable("ARCANE_PARALLEL_SERVICE");
          if (String.IsNullOrEmpty(env_parallel_service))
            args.Add("-A,MessagePassingService=Sequential");
        }

        if (OnAddAdditionalArgs != null)
          OnAddAdditionalArgs(this);
        args.AddRange(AdditionalArgs);
        args.AddRange(m_remaining_args);
        string args_str = String.Join(" ", args.ToArray());

        int r = 0;
        if (m_use_dotnet && want_dotnet_shared && m_properties.NbProc == 0) {
          Console.WriteLine("Launching '.Net' sequential test");
          string test_exe_dll_path = Path.Combine(lib_path, "ArcaneTestExe.dll");
          string all_args = $"{test_exe_dll_path} {args_str}";
          if (m_dotnet_runtime == "coreclr") {
            string coreclr_bin = Utils.DotnetCoreClrPath;
            r = Utils.ExecCommandNoException(coreclr_bin, all_args);
          }
          else
            throw new ArgumentException("Invalid value for option 'dotnet-runtime'. Valid value is 'coreclr'");
        }
        else {
          if (m_properties.UseTotalview) {
            r = Utils.ExecCommandNoException("totalview", "--args " + command + " " + args_str);
          }
          else if (use_gdb) {
            r = Utils.ExecCommandNoException("gdb", "--args " + command + " " + args_str);
          }
          else if (m_properties.UseDdt) {
            string ddt_args = command + " " + args_str;
            if (command.EndsWith("mpiexec") || command.EndsWith("mpirun"))
              ddt_args = args_str;
            r = Utils.ExecCommandNoException("ddt", ddt_args);
          }
          else if (use_devenv) {
            r = Utils.ExecCommandNoException("c:/vs90/Common7/IDE/devenv.exe", "/debugexe " + command + " " + args_str);
          }
          //else if (use_memcheck){
          //  r = Utils.ExecCommandNoException("valgrind",val_args+ " " + command+" " + args_str);
          //}
          else {
            if (!File.Exists(command)) {
              StringBuilder sb = new StringBuilder();
              sb.AppendFormat("ERROR: Can not execute command '{0}' because the file is missing\n", command);
              sb.AppendLine("ERROR: Check if the file exists. On Win32, check if you need to specifiy build config (Debug, Release, ...)");
              Console.WriteLine(sb.ToString());
              return (-1);
            }
            r = Utils.ExecCommandNoException(command, args_str);
          }
        }
        if (r != 0)
          return r;
      }
      return 0;
    }

    // Supprime le répertoire de sortie si demandé
    // Cela permet d'économise de la place sur le disque pour le CI
    public void Cleanup()
    {
      bool do_cleanup = (Utils.GetEnvironmentVariable("ARCANE_TEST_CLEANUP_AFTER_RUN")=="1");
      if (!do_cleanup)
        return;
      string test_name = Utils.GetEnvironmentVariable("ARCANE_TEST_NAME");
      if (String.IsNullOrEmpty(test_name))
        return;
      string test_output_path = Path.Combine(Directory.GetCurrentDirectory(),$"test_output_{test_name}");
      Console.WriteLine($"TEST_NAME={test_name} path={test_output_path}");
      DirectoryInfo output_dir = new DirectoryInfo(test_output_path);
      // On pourrait directement faire output_dir.Delete() mais pour éviter
      // des erreurs de manipulation, on ne le fait que sur les sous-répertoires.
      if (output_dir.Exists){
        foreach (DirectoryInfo dir in output_dir.GetDirectories()){
          Console.WriteLine($"Removing directory {dir.FullName}");
          dir.Delete(true);
        }
      }
    }

    void _HandleMpiLauncher()
    {
      // Comme cette méthode peut être appelée plusieurs fois,
      // supprime les arguments positionnés par les anciens appels.
      m_properties.MpiLauncherArgs.Clear();
      string mpi_exec_name = Utils.MpiExecName;
      string custom_mpi_driver = Utils.CustomMpiDriver;
      if (!String.IsNullOrEmpty(custom_mpi_driver))
        mpi_exec_name = custom_mpi_driver;
      m_properties.MpiLauncher = mpi_exec_name;
      if (m_custom_driver != null) {
        if (m_custom_driver.HandleMpiLauncher(m_properties, mpi_exec_name))
          return;
      }

      m_properties.MpiLauncherArgs.Add("-n");
      m_properties.MpiLauncherArgs.Add(m_properties.NbProc.ToString());

      // A partir de OpenMPI 2.1, on ne peut pas utiliser par défaut plus de processus
      // MPI que de coeurs disponibles sur la machine. Pour éviter cela, il faut
      // ajouter l'option '--oversubscribe'. Sans cela, certains tests peuvent planter
      // sur les machines avec peu de coeurs
      if (Utils.ConfigMpiVendorName == "openmpi") {
        m_properties.MpiLauncherArgs.Add("--oversubscribe");
      }
    }
  }
}
