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
    ~BasicDotNetPythonExternalPlugin()
    {
      if (m_python_scope!=null){
        m_python_scope.Dispose();
        m_python_scope = null;
      }
      //PythonEngine.Shutdown();
    }

    public override void LoadFile(string s)
    {
      _checkInit();
      Console.WriteLine("EXTERNAL LOAD!");
      string text = File.ReadAllText(s);
      Console.WriteLine("TEXT={0}",text);
      Console.WriteLine("-- -- -- Executing Python");
      using (Py.GIL())
      //using (var scope = Py.CreateScope())
      {
        m_python_scope = Py.CreateScope();
        m_python_scope.Exec(text);
      }
      Console.WriteLine("** -- End setup python");
    }

    public override void ExecuteFunction(string function_name)
    {
      if (m_python_scope==null)
        throw new ApplicationException("Null python scope. You need to call 'LoadFile()' before");
      using (Py.GIL())
      //using (var scope = Py.CreateScope())
      {
        m_python_scope.Exec(function_name+"()");
      }
    }

    void _checkInit()
    {
      if (m_has_init)
        return;
      PythonEngine.Initialize();
      m_has_init = true;
    }

    bool m_has_init;
    PyModule m_python_scope;
  }
}
