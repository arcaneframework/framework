using Arcane;
using Python.Runtime;

namespace Arcane.Python
{
  public static class MainInit
  {
    static public void Init()
    {
      // Only useful for loading this assembly
      Debug.Write("Loading '.Net' python wrapping assembly");
    }
    static public void Shutdown()
    {
      Debug.Write("Shutdown '.Net' python wrapping assembly");
      PythonEngine.Shutdown();
    }
  }
}
