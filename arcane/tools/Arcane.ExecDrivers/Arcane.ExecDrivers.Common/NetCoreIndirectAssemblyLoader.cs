//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Reflection;

namespace Arcane.ExecDrivers.Common
{
  public class NetCoreIndirectAssemblyLoader : Utils.IAssemblyLoader
  {
    MethodInfo m_get_default_method;
    MethodInfo m_load_assembly_from_path_method;
    public NetCoreIndirectAssemblyLoader(MethodInfo method1,MethodInfo method2)
    {
      m_get_default_method = method1;
      m_load_assembly_from_path_method = method2;
    }
    public Assembly Load(string name)
    {
      Console.WriteLine($"Loading assembly [.NetCore3 Indirect] '{name}'");
      object default_context =  m_get_default_method.Invoke(null, new object [0]);
      if (default_context==null)
        return null;
      object a = m_load_assembly_from_path_method.Invoke(default_context,new object[] { name });
      if (a != null)
        return a as Assembly;
      return null;
    }

    /*!
     * \brief Détecte si on peut utiliser le chargement via 'AssemblyLoadContext'.
     *
     * Pour cela, détecte si la méthode
     * 'System.Runtime.Loader.AssemblyLoadContext.Defaut.LoadFromAssemblyPath' pour
     * charger les assembly est disponible. Cela est nécessaire avec '.NetCore 3' car la
     * méthode 'Assembly.LoadPath' ne fonctionne pas car elle est spécifique au framework.
     */
    public static Utils.IAssemblyLoader CheckCreateLoader()
    {
      Type type = Type.GetType("System.Runtime.Loader.AssemblyLoadContext");
      if (type==null)
        return null;

      Console.WriteLine($"Has 'AssemblyLoadContext' type={type}");
      BindingFlags common_flags = BindingFlags.Public | BindingFlags.InvokeMethod | BindingFlags.FlattenHierarchy;
      string method_name = "get_Default";
      MethodInfo method = type.GetMethod(method_name, common_flags | BindingFlags.Static);
      if (method == null){
        Console.WriteLine($"Can not find static method named '{method_name}' in type '{type}'");
        return null;
      }
      string method2_name = "LoadFromAssemblyPath";
      MethodInfo method2 = type.GetMethod(method2_name,common_flags | BindingFlags.Instance);
      if (method2 == null){
        Console.WriteLine($"Can not find static method named '{method2_name}' in type '{type}'");
        return null;
      }

      Console.WriteLine($"Found valid method '{method2_name}' in type '{type}'");
      return new NetCoreIndirectAssemblyLoader(method,method2);
    }
  }
}
