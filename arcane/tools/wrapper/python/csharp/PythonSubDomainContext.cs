//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using Python.Runtime;

namespace Arcane.Python
{
  public class BadNDArrayException : Exception
  {
    public BadNDArrayException(string message)
    {
      Message = message;
    }
    public override string Message { get; }
  }
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
      using (Py.GIL()){
        m_numpy_module = Py.Import("numpy");
      }
    }
    internal SubDomainContext(ISubDomain sd, PyObject numpy_module)
    {
      m_sub_domain = sd;
      m_default_mesh = sd.DefaultMesh();
      m_numpy_module = numpy_module;
    }
    public string Name() { return "SubDomain"; }
    public VariableWrapper GetVariable(IMesh mesh, string var_name)
    {
      IVariable v = m_sub_domain.VariableMng().FindMeshVariable(mesh, var_name);
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
      ConstMemoryView memory_view;
      using (Py.GIL())
      {
        dynamic nd = nd_array;
        PyObject p = nd.ctypes.data;
        PyObject dtype = nd_array.GetAttr("dtype");
        dynamic dyn_dtype = dtype;
        dynamic np = m_numpy_module;
        PyObject dtype_f64 = np.float64;
        bool is_same = dtype == dtype_f64;
        //string dt2 = dtype.As<string>();
        //int dtype = nd.dtype.As<int>();
        int item_size = nd.itemsize;
        int nb_item = nd.size;
        UInt64 nd_address = p.As<UInt64>();
        bool is_c_contiguous = nd.flags.c_contiguous;
        if (!is_c_contiguous)
          throw new BadNDArrayException("NDArray is not 'c_contiguous'");
        PyTuple nd_shape = nd.shape;
        long nd_shape_len = nd_shape.Length();
        // TODO: Vérifier que le tableau est layout 'C' et contigu
        Console.WriteLine("ND address={0} is_c?={1} shape={2} shape_len={3}", nd_address, is_c_contiguous, nd_shape, nd_shape_len);
        Console.WriteLine("ND dtype={0} item_size={1} nb_item={2} is_float64={3} x={4}", dtype, item_size, nb_item, is_same, dyn_dtype.str);
        // TODO Supprimer le passage par un double[]
        xvalues = nd_array.As<double[]>();
        memory_view = new ConstMemoryView((IntPtr)nd_address, nb_item * item_size, nb_item, item_size);
      }
      Console.WriteLine("X=" + xvalues.Length);
      bool use_memory_view = true;
      if (use_memory_view) {
        //Console.WriteLine("DIRECT_COPY FROM NUMPY ARRAY");
        Arcane.VariableUtilsInternal.SetFromMemoryBuffer(var_wrapper.m_variable, memory_view);
      }
      else {
        using (var rview_wrapper = new RealArrayView.Wrapper(xvalues)) {
          RealConstArrayView v = rview_wrapper.ConstView;
          ConstArrayView<double> cv = v;
          ConstMemoryView mv = ConstMemoryView.FromView(cv);
          //Arcane.VariableUtilsInternal.SetFromMemoryBuffer(var_wrapper.m_variable, mv);
          //Console.WriteLine("DIRECT_COPY FROM NUMPY ARRAY");
          Arcane.VariableUtilsInternal.SetFromMemoryBuffer(var_wrapper.m_variable, mv);
        }
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
