using System;
using System.Reflection;
using Arcane;

public class ArcaneMainExec
{
  public static int Main(string[] args)
  {
    Debug.Write("Entering '.Net' Main()");
    if (!ArcaneMain.HasDotNetWrapper) {
      var cmd_line_args = CommandLineArguments.Create(args);
      //ApplicationInfo app_info = ArcaneLauncher.ApplicationInfo;
      ArcaneLauncher.Init(cmd_line_args);
    }
    return ArcaneLauncher.Run();
  }
}
