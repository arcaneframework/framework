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
    // Cette méthode peut-être appelée depuis le python et donc peut utiliser le GIL.
    public SubDomainContext(ISubDomain sd) : this(sd, true)
    {
    }
    internal SubDomainContext(ISubDomain sd, bool do_init_modules)
    {
      m_sub_domain = sd;
      m_default_mesh = sd.DefaultMesh();
      if (do_init_modules){
        using (Py.GIL()){
          m_numpy_module = Py.Import("numpy");
          m_ctypes_module = Py.Import("ctypes");
          m_numpy_ctypeslib_module = Py.Import("numpy.ctypeslib");
        }
      }
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
      Arcane.eDataType var_datatype = var_data.DataType();
      bool is_real3 = var_datatype == Arcane.eDataType.DT_Real3;
      if (var_datatype != Arcane.eDataType.DT_Real && !is_real3)
        throw new BadNDArrayException("conversion to NDArray is only valid for variables of type 'Real' or 'Real3'");
      Arcane.IDataInternal xd = VariableUtilsInternal.GetDataInternal(v);
      Arcane.INumericDataInternal ndi = xd.NumericData();
      Int32 nb_item = ndi.Extent0();
      //Console.WriteLine("Doint conversion");
      Arcane.MutableMemoryView mem_view = ndi.MemoryView();
      Console.WriteLine("MemView size={0} nb_element={1} datatype_size={2} nb_item={3}",
                        mem_view.ByteSize, mem_view.NbElement, mem_view.DatatypeSize, nb_item);
      if (nb_item != mem_view.NbElement)
        throw new BadNDArrayException("Bad number of items");
      int dim2_size = 1;
      int nb_dim = 1;
      if (is_real3){
        dim2_size = 3;
        nb_dim = 2;
      }
      dynamic np = m_numpy_module;
      using (Py.GIL())
      {
        dynamic py_numpy_ctypeslib = m_numpy_ctypeslib_module;
        dynamic py_ctypes = m_ctypes_module;
        PyObject[] array_shape = new PyObject[]{ new PyInt(nb_item) };
        if (nb_dim==2)
          array_shape = new PyObject[]{ new PyInt(nb_item), new PyInt(dim2_size) };
        var ob_shape = new PyTuple(array_shape);
        //dynamic nd_pointer = py_numpy_ctypeslib.ndpointer(dtype: np.float64, ndim : nb_dim,
        //                                                  shape : ob_shape, flags: "C_CONTIGUOUS");

        //dynamic ob0 = py_ctypes.cast((long)mem_view.Pointer,nd_pointer);
        //dynamic px = np.frombuffer(ob0);

        dynamic ob0 = py_ctypes.cast((long)mem_view.Pointer,py_ctypes.POINTER(py_ctypes.c_double));
        dynamic px = py_numpy_ctypeslib.as_array(ob0,shape : ob_shape);
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
      using (Py.GIL()){
        m_numpy_ctypeslib_module = Py.Import("numpy.ctypeslib");
        m_ctypes_module = Py.Import("ctypes");
      }
    }
    void _checkNumpyImport()
    {
      if (m_numpy_module == null)
        throw new ApplicationException("Null numpy module");
    }
    readonly ISubDomain m_sub_domain;
    readonly IMesh m_default_mesh;
    PyObject m_numpy_module;
    PyObject m_numpy_ctypeslib_module;
    PyObject m_ctypes_module;
  }
}
