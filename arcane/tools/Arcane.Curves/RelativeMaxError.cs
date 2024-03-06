//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;

namespace Arcane.Curves
{
  /// <summary>
  /// Calcul de l'erreur maximale entre deux courbes
  /// </summary>
  public class RelativeMaxError : NoArgsErrorStrategy
  {
    public override ErrorInfo Compute(ICurve ref_curve, ICurve target_curve,string name, ErrorStrategyArguments args)
    {
      RealArray pgr = null;

      ICurve pcrb1 = null;
      ICurve pcrb2 = null;

      Utils.ComputeProjection(ref_curve,target_curve,args.MinX,args.MaxX,out pcrb1,out pcrb2,out pgr);
      int nbp = pgr.Length;

      if (nbp==0)
        return new ErrorInfo(null,0.0);

      RealArray curve3x = pgr;
      RealArray curve3y = new RealArray(nbp);
      RealConstArrayView pcrb1y = pcrb1.Y;
      RealConstArrayView pcrb2y = pcrb2.Y;

      RealConstArrayView ref_curve_y = ref_curve.Y;

      double max_curve = 0.0;
      //Console.WriteLine("NB_P={0} {1}",nbp,ref_curve_y.Length);
      for (int i = 0; i < ref_curve_y.Length; ++i){
        max_curve = System.Math.Max(max_curve, System.Math.Abs(ref_curve_y[i]));
      }

      for (int i = 0; i < nbp; ++i){
        double delta = (pcrb2y[i] - pcrb1y[i]);
        if (max_curve!=0.0)
          curve3y[i] = delta / max_curve;
        else
          curve3y[i] = delta;
      }

      ICurve curve3 = new BasicCurve(name,curve3x,curve3y);
      double error_value = Utils.NormeInf(curve3);
      return new ErrorInfo(curve3,error_value);
    }
  }
}
