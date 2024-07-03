//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using Python.Runtime;

namespace Arcane.Python
{
  public class VariableWrapper
  {
    internal VariableWrapper(SubDomainContext sdc, IVariable var)
    {
      m_sd_context = sdc;
      m_variable = var;
    }
    public PyObject GetNDArray()
    {
      return m_sd_context.GetNDArray(this);
    }
    public void SetNDArray(PyObject nd_array)
    {
      m_sd_context.SetNDArray(this, nd_array);
    }

    internal readonly SubDomainContext m_sd_context;
    internal readonly IVariable m_variable;
  }

  public class SubDomainContext
  {
    public SubDomainContext(ISubDomain sd)
    {
      m_sub_domain = sd;
      m_default_mesh = sd.DefaultMesh();
    }
    public string Name() { return "SubDomain"; }
    public VariableWrapper GetVariable(IMesh mesh, string var_name)
    {
      IVariable v = m_sub_domain.VariableMng().FindMeshVariable(m_default_mesh, var_name);
      if (v == null)
        return null;
      return new VariableWrapper(this, v);
    }

    internal PyObject GetNDArray(VariableWrapper var_wrapper)
    {
      _checkNumpyImport();
      IVariable v = var_wrapper.m_variable;
      IData var_data = v.Data();
      if (var_data.DataType() != Arcane.eDataType.DT_Real)
        return null;
      Arcane.IDataInternal xd = VariableUtilsInternal.GetDataInternal(v);
      Arcane.INumericDataInternal ndi = xd.NumericData();
      Int32 nb_item = ndi.Extent0();
      Console.WriteLine("Doint conversion");
      Arcane.MutableMemoryView mem_view = ndi.MemoryView();
      Console.WriteLine("MemView size={0} nb_item={1}", mem_view.ByteSize, mem_view.NbElement);
      if (nb_item != mem_view.NbElement)
        throw new ApplicationException("Bad number of items");
      dynamic np = m_numpy_module;
      Arcane.RealArray x = new Arcane.RealArray(nb_item);
      Arcane.VariableUtilsInternal.FillFloat64Array(v, x.View);
      using (Py.GIL())
      {
        dynamic px = np.array(x.View.ToArray(), dtype: np.float64);
        return px;
      }
    }
    internal void SetNDArray(VariableWrapper var_wrapper, PyObject nd_array)
    {
      _checkNumpyImport();
      double[] xvalues = null;
      using (Py.GIL())
      {
        // TODO Supprimer le passage par un double[]
        xvalues = nd_array.As<double[]>();
      }
      Console.WriteLine("X=" + xvalues.Length);
      using (var rview_wrapper = new RealArrayView.Wrapper(xvalues))
      {
        RealConstArrayView v = rview_wrapper.ConstView;
        Arcane.VariableUtilsInternal.SetFromFloat64Array(var_wrapper.m_variable, v);
      }
    }
    public IMesh DefaultMesh { get { return m_default_mesh; } }
    internal void SetNumpyModule(PyObject py_module)
    {
      m_numpy_module = py_module;
    }
    void _checkNumpyImport()
    {
      if (m_numpy_module == null)
        throw new ApplicationException("Null numpy module");
    }
    readonly ISubDomain m_sub_domain;
    readonly IMesh m_default_mesh;
    PyObject m_numpy_module;
  }
}
