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

[Arcane.Service("DotNetCaseFunctionProvider",typeof(Arcane.ICaseFunctionProvider))]
public class DotNetCaseFunctionProvider : Arcane.ICaseFunctionProvider_WrapperService
{
  public DotNetCaseFunctionProvider(ServiceBuildInfo bi) : base(bi)
  {
  }

  public override void RegisterCaseFunctions(ICaseMng cm)
  {
    Console.WriteLine("REGISTER C# CASE_FUNCTION_PROVIDER");
  }
}
