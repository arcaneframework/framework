//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Runtime.InteropServices;

namespace Arcane
{
  public partial class ExternalRef
  {
    static ExternalRef()
    {
      //Console.WriteLine("[C#] Set ExternalRef DESTROY Functor!!!");
      DestroyDelegate d = _DestroyObject;
      var h = GCHandle.Alloc(d,GCHandleType.Normal);
      _ArcaneWrapperCoreSetExternalRefDestroyFunctor(d);
    }
    static void _DestroyObject(IntPtr handle)
    {
      // ATTENTION: Cette méthode peut être appelée depuis un finalizer
      // de fin d'exécution et donc il est possible qu'il soit interdit
      // d'écrire sur la console.
      // Console.WriteLine("[C#] Destroy object");
      var h = GCHandle.FromIntPtr(handle);
      if (h!=null)
        h.Free();
    }
    static public ExternalRef Create(object o)
    {
      var h = GCHandle.Alloc(o,GCHandleType.Normal);
      Debug.Write("[C#] Create handle '{0}'",(IntPtr)h);
      return new ExternalRef((IntPtr)h);
    }

    public object GetReference()
    {
      IntPtr handle = _internalHandle();
      if (handle==IntPtr.Zero)
        return null;
      GCHandle gchandle = GCHandle.FromIntPtr(handle);
      if (gchandle==null)
        return null;
      return gchandle.Target;
    }
  }
}
