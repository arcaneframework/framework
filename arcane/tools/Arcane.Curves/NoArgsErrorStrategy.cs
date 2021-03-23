using System;
namespace Arcane.Curves
{
  public abstract class NoArgsErrorStrategy : IErrorStrategy
  {
    public abstract ErrorInfo Compute (ICurve ref_curve, ICurve target_curve, string name, ErrorStrategyArguments args);
    public ErrorInfo Compute (ICurve ref_curve, ICurve target_curve, string name)
    {
      return this.Compute(ref_curve, target_curve, name, new ErrorStrategyArguments());
    }
  }
}
