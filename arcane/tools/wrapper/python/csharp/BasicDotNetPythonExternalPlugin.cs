using Arcane;
using System;
using Python.Runtime;
using System.Threading.Tasks;
using Numpy;
using System.IO;

namespace Arcane.Python
{
  [Arcane.Service("BasicDotNetPythonExternalPlugin",typeof(Arcane.IExternalPlugin))]
  public class BasicDotNetPythonExternalPlugin : Arcane.IExternalPlugin_WrapperService
  {

    public BasicDotNetPythonExternalPlugin(ServiceBuildInfo bi) : base(bi)
    {
    }

    public override void LoadFile(string s)
    {
      PythonEngine.Initialize();
      Console.WriteLine("EXTERNAL LOAD!");
      string text = File.ReadAllText(s);
      Console.WriteLine("TEXT={0}",text);
      Console.WriteLine("-- -- -- Executing Python");
      using (Py.GIL())
      using (var scope = Py.CreateScope())
      {
        scope.Exec(text);
      }
      Console.WriteLine("** -- End setup python");
    }
  }
}
