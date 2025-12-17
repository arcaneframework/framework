//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

using Arcane;
using Python.Runtime;

namespace Arcane.Python
{
  public static class MainInit
  {
    static public void Init()
    {
      // Only useful for loading this assembly
      Debug.Write("Loading '.Net' python wrapping assembly");
    }
    static public void Shutdown()
    {
      Debug.Write("Shutdown '.Net' python wrapping assembly");
      PythonEngine.Shutdown();
    }
  }
}
