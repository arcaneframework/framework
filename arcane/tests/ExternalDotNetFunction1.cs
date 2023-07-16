using System;
using Arcane;
using Real = System.Double;

namespace ArcaneTest
{
  public class TestCaseFunction1
  {
    public Real DynamicExternalFuncTimeMultiply2(Real x)
    {
      Console.WriteLine("DynamicExternalFuncTimeMultiply2 x={0}",x);
      return x * 2.0;
    }
    public int DynamicExternalFuncIterMultiply3(int x)
    {
      Console.WriteLine("DynamicExternalFuncIterMultiply3 x={0}",x);
      return x * 3;
    }
    public Real DynamicExternalFuncStandardRealReal3NormL2(Real x,Real3 position)
    {
      Console.WriteLine("DynamicExternalFuncStandardNormL2 x={0} position={1}",x,position);
      return x * System.Math.Sqrt(position.x*position.x + position.y*position.y + position.z*position.z);
    }
  }
}
