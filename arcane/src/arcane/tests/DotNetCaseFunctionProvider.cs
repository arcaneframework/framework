using System;
using Arcane;
using Real = System.Double;

#if ARCANE_64BIT
using Integer = System.Int64;
using IntegerConstArrayView = Arcane.Int64ConstArrayView;
using IntegerArrayView = Arcane.Int64ArrayView;
#else
using Integer = System.Int32;
using IntegerConstArrayView = Arcane.Int32ConstArrayView;
using IntegerArrayView = Arcane.Int32ArrayView;
#endif

namespace ArcaneTest
{
  [Arcane.Service("DotNetCaseFunctionProvider",typeof(Arcane.ICaseFunctionProvider))]
  public class DotNetCaseFunctionProvider : Arcane.ICaseFunctionProvider_WrapperService
  {
    public DotNetCaseFunctionProvider(ServiceBuildInfo bi) : base(bi)
    {
    }

    public override void RegisterCaseFunctions(ICaseMng cm)
    {
      Console.WriteLine("REGISTER C# CASE_FUNCTION_PROVIDER");
      Arcane.CaseFunctionLoader.LoadCaseFunction(cm,typeof(MyTestCaseFunction));
    }
  }

  public class MyTestCaseFunction
  {
    public Real FuncTimeMultiply2(Real x)
    {
      Console.WriteLine("FuncTimeMultiply2 x={0}",x);
      return x * 2.0;
    }
    public int FuncIterMultiply3(int x)
    {
      Console.WriteLine("FuncIterMultiply3 x={0}",x);
      return x * 3;
    }
  }
}
