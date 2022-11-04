//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using Arcane;
#if ARCANE_64BIT
using Integer = System.Int64;
#else
using Integer = System.Int32;
#endif

namespace Tests
{
  class MainClass
  {
    public static void Main(string[] args)
    {
      Console.WriteLine("Hello World!");
      Int32Array a = new Int32Array();
      a.Add(5);
      a.Add(23);
      a.Add(7);
      a.Add(32);
      a.Add(56);
      if (a.Size!=5)
        throw new ApplicationException();
      Console.WriteLine("CAPACITY = {0}",a.Capacity);
      Int32ConstArrayView view = a.ConstView;

      Console.WriteLine("Values: {0} {1} {2}",view[0],view[1],view[2]);
      for( Integer i=0, n=a.Size; i<n; ++i )
        if (a[i] != view[i])
          throw new ApplicationException();
      a.Dispose();
      _Test1();
    }

    private static void _Test1()
    {
      for( int i=0; i<10000; ++i ){
        Int32Array a = new Int32Array();
        a.Resize(100000);
        a.Dispose();
      }
      for( int i=0; i<1000; ++i ){
        Int32Array a = new Int32Array();
        a.Resize(100000);
        //a.Dispose();
      }
      Console.WriteLine("Memory Used {0}",GC.GetTotalMemory(false));
      GC.Collect(2);
      GC.WaitForPendingFinalizers();
      GC.Collect(2);
      Console.WriteLine("End of allocation test");
    }
  }
}
