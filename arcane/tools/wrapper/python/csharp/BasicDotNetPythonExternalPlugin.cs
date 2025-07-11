using System;
using Python.Runtime;
using System.IO;

namespace Arcane.Python
{
  [Arcane.Service("BasicDotNetPythonExternalPlugin",typeof(Arcane.IExternalPlugin))]
  public class BasicDotNetPythonExternalPlugin : Arcane.IExternalPlugin_WrapperService
  {

    public BasicDotNetPythonExternalPlugin(ServiceBuildInfo bi) : base(bi)
    {
      m_sub_domain_context = new SubDomainContext(bi.SubDomain(), null);
    }

    ~BasicDotNetPythonExternalPlugin()
    {
      m_py_sub_domain_context = null;
      if (m_python_scope!=null){
        m_python_scope.Dispose();
        m_python_scope = null;
      }
    }

    public override void LoadFile(string s)
    {
      _checkInit();
      Console.WriteLine("EXTERNAL LOAD!");
      string text = File.ReadAllText(s);
      Console.WriteLine("TEXT={0}",text);
      Console.WriteLine("-- -- -- Executing Python");
      using (Py.GIL())
      {
        m_python_scope = Py.CreateScope();
        m_python_scope.Exec(text);
        if (m_py_sub_domain_context==null)
          m_py_sub_domain_context = m_sub_domain_context.ToPython();
        if (m_numpy_module==null){
          m_numpy_module = Py.Import("numpy");
          m_sub_domain_context.SetNumpyModule(m_numpy_module);
        }
      }
      Console.WriteLine("** -- End setup python");
    }

    public override void ExecuteFunction(string function_name)
    {
      _checkScope();
      using (Py.GIL())
      {
        m_python_scope.InvokeMethod(function_name);
      }
    }

    public override void ExecuteContextFunction(string function_name)
    {
      _checkScope();
      using (Py.GIL())
      {
        m_python_scope.InvokeMethod(function_name,m_py_sub_domain_context);
      }
    }

    void _checkScope()
    {
      if (m_python_scope==null)
        throw new ApplicationException("Null python scope. You need to call 'LoadFile()' before");
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
    readonly SubDomainContext m_sub_domain_context;
    PyObject m_py_sub_domain_context;
    PyObject m_numpy_module;
  }
}
