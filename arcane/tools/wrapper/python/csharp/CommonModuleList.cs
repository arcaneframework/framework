//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using Python.Runtime;
using System.IO;

namespace Arcane.Python
{
  /*!
   * \brief Liste des modules utilisés par le wrapper.
   */
  class CommonModuleList
  {
    public void ImportModules()
    {
      using (Py.GIL()){
        _ImportModule(ref m_numpy_module,"numpy");
        _ImportModule(ref m_ctypes_module,"ctypes");
        _ImportModule(ref m_numpy_ctypeslib_module,"numpy.ctypeslib");
        _ImportModule(ref m_arcane_python_module,"_ArcanePython");
      }
    }

    void _ImportModule(ref PyObject py_module, string module_name)
    {
      if (py_module == null){
        py_module = Py.Import(module_name);
        if (py_module == null)
          throw new ApplicationException($"Can not import python module '{module_name}'");
      }
    }

    public PyObject Numpy { get { return m_numpy_module; } }
    public PyObject NumpyCtypeslib { get { return m_numpy_ctypeslib_module; } }
    public PyObject Ctypes { get { return m_ctypes_module; } }
    public PyObject ArcanePython { get { return m_arcane_python_module; } }

    PyObject m_numpy_module;
    PyObject m_numpy_ctypeslib_module;
    PyObject m_ctypes_module;
    PyObject m_arcane_python_module;
  }
}
