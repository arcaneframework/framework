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
   * \brief Service de couplage python via pythonnet.
   *
   * Il faut appeler LoadFile() avant d'utiliser cette instance.
   * Il y a deux manières de l'utiliser:
   * 1. en passant fichier python. Dans ce cas on exécute ce fichier
   *    dans un 'python scope' créé via Py.CreateScope().
   * 2. sans passer par un fichier python. Dans ce cas on considère que
   *    l'initialisation python a déjà eu lieu.
   *
   * Lors de l'appel à ExecuteFunction() ou ExecuteContextFunction(), on
   * considère dans le cas (1) que la fonction est dans le fichier lu et qu'elle
   * est dans le namespace global dans le cas (2).
   */
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
    /*!
     * \brief Charge un fichier python.
     *
     * Charge le fichier python de nom \a file_name si \a file_name n'est
     * pas nul ou pas vide. Sinon, récupère l'environnement global python
     * en supposant que l'environnement python est déjà existant (par exemple
     * car on le code est lancé via un script python).
     */
    public override void LoadFile(string file_name)
    {
      _checkInit();
      bool has_file = !String.IsNullOrEmpty(file_name);
      Console.WriteLine("EXTERNAL LOAD file_name='{0}'",file_name);
      string text = String.Empty;
      if (has_file){
        text = File.ReadAllText(file_name);
        Console.WriteLine("TEXT={0}",text);
      }
      Console.WriteLine("-- -- -- Executing Python");
      using (Py.GIL())
      {
        if (has_file){
          m_python_scope = Py.CreateScope();
          m_python_scope.Exec(text);
        }
        else{
          m_python_global_dictionary = PythonEngine.Eval("globals()");
        }
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
      _checkIsLoadFileCalled();
      using (Py.GIL())
      {
        if (m_python_global_dictionary!=null){
          PyObject func = _GetMethodFromGlobalScope(function_name);
          func.Invoke();
        }
        else
          m_python_scope.InvokeMethod(function_name);
      }
    }

    public override void ExecuteContextFunction(string function_name)
    {
      _checkIsLoadFileCalled();
      using (Py.GIL())
      {
        if (m_python_global_dictionary!=null){
          PyObject func = _GetMethodFromGlobalScope(function_name);
          func.Invoke(m_py_sub_domain_context);
        }
        else
          m_python_scope.InvokeMethod(function_name,m_py_sub_domain_context);
      }
    }

    //NOTE: Cette méthode doit être appelée avec le GIL actif.
    PyObject _GetMethodFromGlobalScope(string function_name)
    {
      // TODO: Vérifier que l'object est une fonction.
      PyObject py_func = m_python_global_dictionary.GetItem(function_name);
      if (py_func==null)
        throw new ApplicationException($"No function named '{function_name}' in the global scope");
      return py_func;
    }

    void _checkIsLoadFileCalled()
    {
      if (m_python_scope==null && m_python_global_dictionary==null)
        throw new ApplicationException("Initialization error: LoadFile()' has not beed called.");
    }

    void _checkInit()
    {
      lock(m_init_lock){
        if (m_has_init)
          return;
        PythonEngine.Initialize();
        m_has_init = true;
      }
    }

    bool m_has_init;
    object m_init_lock = new object();
    PyModule m_python_scope;
    PyObject m_python_global_dictionary;
    readonly SubDomainContext m_sub_domain_context;
    PyObject m_py_sub_domain_context;
    PyObject m_numpy_module;
  }
}
