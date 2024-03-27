using System;
using Arcane;
using System.Reflection;
using System.IO;

public class Launcher
{
  [System.Runtime.InteropServices.DllImport("arcane_tests_lib", EntryPoint="arcaneTestSetApplicationInfo")]
  public static extern void InitCommonApplicationInfo();

  public static int Exec(string[] args)
  {
    var cmd_line_args = CommandLineArguments.Create(args);
    ApplicationInfo app_info = ArcaneLauncher.ApplicationInfo;

    // Positionne le chemin contenant les infos de configuration
    // Il s'agit du répertoire contenant cette assembly.
    Assembly this_assembly = Assembly.GetAssembly(typeof(Launcher));
    string this_assembly_path = Path.GetDirectoryName(this_assembly.Location);
    app_info.SetDataDir(this_assembly_path);
    app_info.SetDataOsDir(this_assembly_path);

    ArcaneLauncher.SetCommandLineArguments(cmd_line_args);
    InitCommonApplicationInfo();
    var dotnet_info = ArcaneLauncher.DotNetRuntimeInitialisationInfo;

    string s3 = Environment.GetEnvironmentVariable("ARCANE_DOTNET_ASSEMBLY");
    if (!String.IsNullOrEmpty(s3))
      dotnet_info.SetMainAssemblyName(s3);

    string s = Environment.GetEnvironmentVariable("ARCANE_SIMPLE_EXECUTOR");
    if (!String.IsNullOrEmpty(s))
      return ExecDirect();

    return _ExecStandard();
  }

  public static int _ExecStandard()
  {
    Console.WriteLine("ArcaneTest.Launcher.Exec (V2)");
#if ARCANE_HAS_DOTNET_PYTHON
    Arcane.Python.MainInit.Init();
#endif
    return ArcaneMain.Run();
  }

  public static int ExecDirect()
  {
    ArcaneSimpleExecutor.ExecFunctor exec_func = (ArcaneSimpleExecutor executor) =>
    {
      return TestClass.Test1(executor);
    };

    return ArcaneSimpleExecutor.Run(exec_func);
  }
}

public class TestClass
{
  public static int Test1(ArcaneSimpleExecutor executor)
  {
    ISubDomain sd = executor.CreateSubDomain();
    var mrm = new MeshReaderMng(sd);
    string mesh_file_name = "sod.vtk";
    IMesh mesh = mrm.ReadMesh("Mesh1",mesh_file_name);
    Console.WriteLine("MESH_NB_CELL4={0}",mesh.NbCell());
    return 0;
  }

  public static void Test2()
  {
    Console.WriteLine("Calling specific init method");
  }

  public static void BadSignatureTest3(int v)
  {
  }

}
