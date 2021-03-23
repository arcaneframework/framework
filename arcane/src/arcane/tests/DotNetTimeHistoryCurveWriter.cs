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

[Arcane.Service("DotNetCurveWriter2",typeof(Arcane.ITimeHistoryCurveWriter2))]
public class DotNetTimeHistoryCurveWriter2 : Arcane.ITimeHistoryCurveWriter2_WrapperService
{
  string m_output_path;

  public DotNetTimeHistoryCurveWriter2(ServiceBuildInfo bi) : base(bi)
  {
  }

  public override void Build()
  {
    Console.WriteLine("C# CurveWriter: build");
  }

  public override void BeginWrite(TimeHistoryCurveWriterInfo writer_info)
  {
    Console.WriteLine("C# CurveWriter: begin write");
  }

  public override void WriteCurve(TimeHistoryCurveInfo curve_info)
  {
    Console.WriteLine("C# CurveWriter: write curve name={0} nb_value={1}",
                      curve_info.Name(),curve_info.Values().Size);
  }

  public override void EndWrite()
  {
    Console.WriteLine("C# CurveWriter: end write");
  }

  public override string Name()
  {
    return "dotnetcurve2";
  }

  public override void SetOutputPath(string path)
  {
    m_output_path = path;
  }

  public override string OutputPath()
  {
    return m_output_path;
  }
}
