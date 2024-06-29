using System;
using System.Collections.Generic;
using Arcane;
using Numpy;
using Python.Runtime;

namespace Arcane.Python
{
  public class SubDomainContext
  {
    public SubDomainContext(ISubDomain sd)
    {
      m_sub_domain = sd;
      m_default_mesh = sd.DefaultMesh();
    }
    internal void SetNumpyModule(PyObject py_module)
    {
      m_numpy_module = py_module;
    }
    public string Name() { return "SubDomain"; }
    public PyObject GetNDArray(string var_name)
    {
      if (m_numpy_module == null)
        throw new ApplicationException("Null numpy module");
      IVariable v = m_sub_domain.VariableMng().FindMeshVariable(m_default_mesh, var_name);
      if (v == null)
        return null;
      IData var_data = v.Data();
      if (var_data.DataType() != Arcane.eDataType.DT_Real)
        return null;
      Arcane.IDataInternal xd = VariableUtilsInternal.GetDataInternal(v);
      Arcane.INumericDataInternal ndi = xd.NumericData();
      Int32 nb_item = ndi.Extent0();
      Console.WriteLine("Doint conversion");
      dynamic np2 = m_numpy_module;
      using (Py.GIL())
      {
        Arcane.RealArray x = new Arcane.RealArray(nb_item);
        Arcane.VariableUtilsInternal.FillFloat64Array(v, x.View);
        dynamic px = np2.array(x.View.ToArray());
        return px;
      }
    }
    readonly ISubDomain m_sub_domain;
    readonly IMesh m_default_mesh;
    PyObject m_numpy_module;
  }
}
