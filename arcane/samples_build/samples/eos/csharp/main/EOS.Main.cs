using System;
using Arcane;

public class EOSMain
{
  public static int Main(string[] args)
  {
    Console.WriteLine("THIS IS Eos.Main C# EXE");
    var cmd_line_args = CommandLineArguments.Create(args);
    ArcaneLauncher.Init(cmd_line_args);
    ApplicationInfo app_info = ArcaneLauncher.ApplicationInfo;
    app_info.SetCodeName("EOS");
    app_info.SetCodeVersion(new VersionInfo(1,0,0));
    app_info.AddDynamicLibrary("EOSLib");
    int r = ArcaneMain.Run();
    return r;
  }
}
