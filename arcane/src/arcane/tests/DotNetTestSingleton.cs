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

using System.Collections.Generic;

[Arcane.Service("DotNetTestSingletonCS",typeof(Arcane.IUnitTest))]
public class DotNetTestSingleton : Arcane.IUnitTest_WrapperService
{
  public DotNetTestSingleton(ServiceBuildInfo bi) : base(bi)
  {
  }
  public override void InitializeTest(){}
  public override void ExecuteTest(){}
  public override void FinalizeTest(){}
}
