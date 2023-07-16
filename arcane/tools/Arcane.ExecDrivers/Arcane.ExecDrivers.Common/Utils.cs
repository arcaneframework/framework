//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.IO;
using System.Text;
using System.Diagnostics;
using System.Reflection;
using System.Collections.Generic;

namespace Arcane.ExecDrivers.Common
{
  /*!
   * \brief Diverses fonctions utilitaires
   */
  static public class Utils
  {
    public interface IAssemblyLoader
    {
      Assembly Load(string full_name);
    }

    //! Chargeur spécifique d'assembly (utile avec netcoreapp3.0)
    public static IAssemblyLoader AssemblyLoader { get; set; }

    static Encoding m_default_encoding = Encoding.Default;
    public static Encoding DefaultEncoding
    {
      get { return m_default_encoding; }
      set { m_default_encoding = value; }
    }

    static string m_code_bin_path;
    public static string CodeBinPath { get { return m_code_bin_path; } }

    static string m_code_lib_path;
    public static string CodeLibPath { get { return m_code_lib_path; } }

    static string m_code_share_path;
    public static string CodeSharePath { get { return m_code_share_path; } }

    static string m_mpi_exec_name;
    public static string MpiExecName { get { return m_mpi_exec_name; } }

    static string m_valgrind_exec_name;
    public static string ValgrindExecName { get { return m_valgrind_exec_name; } }

    static string m_mono_exec_path;
    public static string MonoExecPath { get { return m_mono_exec_path; } }
    public static string DotnetCoreClrPath { get; private set; }
    public static string DotnetCoreClrFullVersion { get; private set; }
    public static string DotnetCoreClrVersion { get; private set; }
    public static string DotnetCoreClrSdkPath { get; private set; }

    static string m_external_libraries;
    public static string ExternalLibraries { get { return m_external_libraries; } }

    static bool m_is_init = false;
    static bool m_is_win32 = false;
    public static bool IsWin32 { get { return m_is_win32; } }

    static string m_outdir;
    public static string OutDir { get { return m_outdir; } set { m_outdir = value; } }

    public static string ConfigMpiexec { get; set; }
    public static string ConfigMpiexecNumprocFlag { get; set; }
    public static string ConfigMpiexecPreflags { get; set; }
    public static string ConfigMpiexecPostflags { get; set; }
    public static string ConfigMpiVendorName { get; set; }
    public static string CustomMpiDriver { get; set; }

    static Dictionary<string,string> m_settings;

    /*!
     * \brief Initialise l'application.
     *
     * - Lit les valeurs du fichier de configuration.
     */
    public static void Init()
    {
      if (m_is_init)
        return;
      // Récupère l'info de la Plateforme. Doit se fait au début car cela
      // est utilisé par certaines fonctions comme _NormalizePath().
      OperatingSystem os = Environment.OSVersion;
      PlatformID pid = os.Platform;
      m_is_win32 = pid == PlatformID.Win32NT;

      Encoding encoding = Encoding.GetEncoding("utf-8");
      Utils.DefaultEncoding = encoding;

      if (AssemblyLoader==null)
        AssemblyLoader = NetCoreIndirectAssemblyLoader.CheckCreateLoader();

      Assembly a = Assembly.GetExecutingAssembly();
      string path = a.Location;

      // Regarde si la version 'install' existe et si c'est le cas prend ce fichier de configuration
      string config_path = path + ".install.config.json";
      if (!File.Exists(config_path))
        config_path = path + ".config.json";
      var settings =_ReadConfigJSON(config_path);
      m_settings = settings;

      m_code_bin_path = NormalizePath(_ReadConfig(settings, "CodeBinPath"));
      m_code_lib_path = NormalizePath(_ReadConfig(settings, "CodeLibPath"));
      m_code_share_path = NormalizePath(_ReadConfig(settings, "CodeSharePath"));
      m_mpi_exec_name = NormalizePath(_ReadConfig(settings, "MpiBinary"));
      m_valgrind_exec_name = NormalizePath(_ReadConfig(settings, "ValgrindBinary"));
      m_mono_exec_path = NormalizePath(_ReadConfig(settings, "MonoExecPath"));
      DotnetCoreClrPath = NormalizePath(_ReadConfig(settings, "DotnetCoreClrPath"));
      DotnetCoreClrFullVersion = NormalizePath(_ReadConfig(settings, "DotnetCoreClrFullVersion"));
      DotnetCoreClrVersion = NormalizePath(_ReadConfig(settings, "DotnetCoreClrVersion"));
      DotnetCoreClrSdkPath = NormalizePath(_ReadConfig(settings, "DotnetCoreClrSdkPath"));
      m_external_libraries = _ReadConfig(settings, "ExternalLibraries");
      CustomMpiDriver = _ReadConfig(settings, "CustomMpiDriver");

      ConfigMpiexec = NormalizePath(_ReadConfig(settings, "Config_MPIEXEC"));
      ConfigMpiexecNumprocFlag = NormalizePath(_ReadConfig(settings, "Config_MPIEXEC_NUMPROC_FLAG"));
      ConfigMpiexecPreflags = NormalizePath(_ReadConfig(settings, "Config_MPIEXEC_PREFLAGS"));
      ConfigMpiexecPostflags = NormalizePath(_ReadConfig(settings, "Config_MPIEXEC_POSTFLAGS"));
      ConfigMpiVendorName = NormalizePath(_ReadConfig(settings, "Config_MPI_VENDOR_NAME"));


      // $OutDir est utilisé sous Win32 pous spécifier le type de build,
      // à savoir 'Debug', 'Release', ...
      // Depuis Visual Studio 10, cette variable n'existe plus. Elle
      // est remplacée par $Configuration.
      // Pour éviter tous ces problèmes de compatibilité,  CMake fournit
      // une variable d'environnemet _CONFIG_TYPE qui contient
      // toujours la bonne valeurs. Il est aussi possible avec ctest
      // de passer le type de build avec l'option '-C':
      //   ctest -I 1,1,1 -C Debug
      if (m_is_win32){
        m_outdir = GetEnvironmentVariable("OutDir");
        if (String.IsNullOrEmpty(m_outdir)) {
          m_outdir = GetEnvironmentVariable("CMAKE_CONFIG_TYPE");
        }
      }
      m_is_init = true;
    }

    public static string ReadConfig(string name)
    {
      return _ReadConfig(m_settings, name);
    }

    static Dictionary<string,string> _ReadConfigJSON(string config_path)
    {
      Dictionary<string,string> dict = null;
      // Lit le fichier de configuration 'ArcaneCea.config' au format JSON
      // qui doit se trouver dans le même répertoire que cette assembly.
      Console.WriteLine("JSON: Try read config path={0}",config_path);
      Newtonsoft.Json.JsonSerializer ser = new Newtonsoft.Json.JsonSerializer();
      using (StreamReader sr = new StreamReader(config_path)) {
        Newtonsoft.Json.JsonTextReader r = new Newtonsoft.Json.JsonTextReader(sr);
        dict = ser.Deserialize<Dictionary<string, string>>(r);
      }
      return dict;
    }

    public static string NormalizePath(string path)
    {
      // S'assure que les chemins sont au bon format suivant le système:
      // - sous Unix, les chemins doivent uniquement comporter des '/'.
      // - sous Windows, on peut tolérer les '\' et les '/' mais cela peut
      // poser problème d'avoir les deux dans un même nom. Comme Path.Combine()
      // utilise le '\', on fait de même pour être cohérent.
      if (Utils.IsWin32)
        return path.Replace("/", "\\");
      return path.Replace("\\", "/");
    }

    private static string _ReadConfig(Dictionary<string,string> settings, string name)
    {
      string value = null;
      bool is_ok = settings.TryGetValue(name,out value);
      if (!is_ok) {
        Console.WriteLine("ConfigPath {0}", Assembly.GetExecutingAssembly().Location);
        throw new ArgumentNullException("Valeur de configuration '{0}' absente. Vérifier l'installation", name);
      }
      Console.WriteLine("Configuration {0} = {1}", name, value);
      return value;
    }

    // Lance une commande shell.
    // Lève une exception en cas d'erreur.
    public static void ExecShellCommand(string cmd, string work_dir)
    {
      int ret = ExecShellCommandNoException(cmd, work_dir);
      if (ret != 0)
        throw new Exception("shell command failed");
    }

    // Lance une commande
    // Lève une exception en cas d'erreur.
    public static void ExecCommand(string cmd, string args,string work_dir)
    {
      int ret = ExecCommandNoException(cmd, args, work_dir);
      if (ret != 0)
        throw new Exception("shell command failed");
    }

    // Lance une commande shell.
    public static int ExecShellCommandNoException(string cmd, string work_dir)
    {
      if (work_dir != null)
        Console.WriteLine("DATE: {0} ExecShellCommand (in dir '{1}'): {2}", DateTime.Now, work_dir, cmd);
      else
        Console.WriteLine("DATE: {0} ExecShellCommand: {1}", DateTime.Now, cmd);
      //int retval = 0;
      using (Process process = new Process()) {

        if (Utils.IsWin32){
          throw new NotImplementedException("ExecShellCommand on Windows platform");
        }
        else{
          process.StartInfo.FileName = "/bin/sh";
          process.StartInfo.Arguments = "-c \"" + cmd + "\"";
          process.StartInfo.UseShellExecute = false;
        }
        if (work_dir != null)
          process.StartInfo.WorkingDirectory = work_dir;
        process.Start();
        process.WaitForExit();
        Console.WriteLine("DATE: {0} ExecShellCommand: Finished: {1}", DateTime.Now, cmd);
        return process.ExitCode;
      }
    }
    // Lance une commande shell avec timeout (en miliseconde)
    // Si au bout de \a timeout milisecond, le process n'est pas terminé,
    // le tue et le relance. Au bout de \a nb_try essai, s'arrête
    public static int ExecShellCommandNoExceptionMaxTime(string cmd, int timeout, int nb_try)
    {
      Console.WriteLine("DATE: {0} ExecShellCommandMaxTime: {1}", DateTime.Now, cmd);
      //int retval = 0;
      using (Process process = new Process()) {

        process.StartInfo.FileName = "/bin/sh";
        process.StartInfo.Arguments = "-c \"" + cmd + "\"";
        process.StartInfo.UseShellExecute = false;

        for (int i = 0; i < nb_try; ++i) {
          process.Start();
          bool has_exited = process.WaitForExit(timeout);
          if (has_exited)
            break;
          Console.WriteLine("DATE: {0} ExecShellCommand: Kill process: {1}", DateTime.Now, cmd);
          bool is_killed = false;
          if (!is_killed)
            process.CloseMainWindow();
          process.WaitForExit(50000);
        }
        Console.WriteLine("DATE: {0} ExecShellCommand: Finished: {1}", DateTime.Now, cmd);
        return process.ExitCode;
      }
    }

    // Lance une commande shell.
    public static int ExecCommandNoException(string file, string args, string work_dir=null)
    {
      Console.WriteLine($"DATE: {DateTime.Now} ExecCommand: {file} {args}");
      //int retval = 0;
      using (Process process = new Process()) {

        process.StartInfo.FileName = file;
        process.StartInfo.Arguments = args;
        process.StartInfo.UseShellExecute = false;
        if (work_dir != null)
          process.StartInfo.WorkingDirectory = work_dir;
        process.Start();
        process.WaitForExit();
        return process.ExitCode;
      }
    }

    // Lance une commande shell et retourne la sortie de cette commande
    public static string GetShellOutput(string cmd, string args)
    {
      using (Process process = new Process()) {
        process.StartInfo.FileName = cmd;
        process.StartInfo.Arguments = args;
        process.StartInfo.UseShellExecute = false;
        process.StartInfo.RedirectStandardOutput = true;
        process.Start();
        process.WaitForExit();
        StreamReader sr = process.StandardOutput;
        string s = sr.ReadToEnd();
        return s;
      }
    }

    public static void SetEnvironmentVariable(string name, string value)
    {
      string s = name + "=" + value;
      Console.WriteLine("PUTENV: '{0}'", s);
      Environment.SetEnvironmentVariable(name, value);
    }

    public static string GetEnvironmentVariable(string name)
    {
      return Environment.GetEnvironmentVariable(name);
    }

    public static string GetTestPath()
    {
      string test_path = Utils.CodeLibPath;
      string outdir = Utils.OutDir;
      //if (!String.IsNullOrEmpty(outdir))
        //test_path = Path.Combine(test_path, outdir);
      return test_path;
    }

    public static Assembly LoadAssembly(string full_path)
    {
      Assembly b = null;
      if (AssemblyLoader!=null)
        b = AssemblyLoader.Load(full_path);
      if (b==null)
        b = Assembly.LoadFile(full_path);
      return b;
    }
  }
}
