//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using Arcane;
using Integer = System.Int32;
using Real = System.Double;

namespace Tests
{
  class MainClass
  {
    public static void Main(string[] args)
    {
      Console.WriteLine("Hello World!");
      using Int32Array a = new Int32Array();
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
      for( int i=0; i<1000; ++i ){
        var a = new Array<Int32>();
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


namespace ArcaneTest
{
  public class Tester1
  {
    public void Test()
    {
      Array<Real> real = new Array<Real>();
      ArrayView<Real> real_view = real.View;
      Array<Real2> m_real2 = new Array<Real2>();
      Array<Real3> m_real3 = new Array<Real3>();
      Array<Real2x2> m_real2x2 = new Array<Real2x2>();
      Array<Real3x3> m_real3x3 = new Array<Real3x3>();

      RealArray e_real_array = new RealArray();
      RealArrayView e_real_array_view = e_real_array.View;
      ArrayView<Real> real_array_view = e_real_array.View;
      
      RealConstArrayView e_const_real_array_view = e_real_array.ConstView;
      ConstArrayView<Real> const_real_array_view1 = e_real_array.ConstView;
      ConstArrayView<Real> const_real_array_view2 = e_real_array.View;
    }
  }
}
