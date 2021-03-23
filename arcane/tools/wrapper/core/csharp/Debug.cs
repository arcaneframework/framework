using System;

namespace Arcane
{
  public static class Debug
  {
    public static void Write(string str)
    {
      Write(1,str);
    }

    public static void Write(string format,params object[] objs)
    {
      Write(1,format,objs);
    }

    public static void Write(int level,string str)
    {
      if (ArcaneMain.VerboseLevel>=level)
        Console.WriteLine($"[C#] {str}");
    }

    public static void Write(int level,string format,params object[] objs)
    {
      if (ArcaneMain.VerboseLevel>=level)
        Console.WriteLine($"[C#] {format}",objs);
    }
  }
}
