
using System;

namespace Arcane.Curves
{
  public static class CurveFactory
  {
    public static ICurve Create(string name,double[] x,double[] y)
    {
      RealArray rx = new RealArray(x.Length);
      for( int i=0; i<x.Length; ++i )
        rx[i] = x[i];
      RealArray ry = new RealArray(y.Length);
      for( int i=0; i<y.Length; ++i )
        ry[i] = y[i];
      return new BasicCurve(name,rx,ry);
    }
  }
}
