//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Reflection;
using System.Runtime.Loader;

namespace Arcane.ExecDrivers.Common
{
  public class NetCoreIndirectAssemblyLoader : Utils.IAssemblyLoader
  {
    public Assembly Load(string name)
    {
      Console.WriteLine($"Loading assembly [.NetCore3 direct with AssemblyLoadContext] '{name}'");
      return AssemblyLoadContext.Default.LoadFromAssemblyPath(name);
    }

    //! Créé une instance pour charger une assembly.
    public static Utils.IAssemblyLoader CheckCreateLoader()
    {
      return new NetCoreIndirectAssemblyLoader();
    }
  }
}
