//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
namespace Arcane.Curves
{
  public class DrbarError : NoArgsErrorStrategy
  {
    public override ErrorInfo Compute(ICurve ref_curve, ICurve target_curve,string name,ErrorStrategyArguments args)
    {
      RealArray pgr = null;

      ICurve pcrb1 = null;
      ICurve pcrb2 = null;

      Utils.ComputeProjection(ref_curve,target_curve,args.MinX,args.MaxX, out pcrb1,out pcrb2,out pgr);
      int nbp = pgr.Length;
      RealArray curve3x = pgr;
      RealArray curve3y = new RealArray(nbp);
      RealConstArrayView pcrb1y = pcrb1.Y;
      RealConstArrayView pcrb2y = pcrb2.Y;

      double max_ref = pcrb1y[0];
      double max_target = pcrb2y[0];

      for (int i = 0; i < nbp; ++i){
        max_ref = Math.Max(max_ref, pcrb1y[i]);
        max_target = Math.Max(max_target, pcrb2y[i]);
        curve3y[i] = max_target - max_ref;
      }
      
      ICurve curve3 = new BasicCurve(name,curve3x,curve3y);
      int nb_p = curve3.NbPoint;
      double error_value = 0.0;
      if (nb_p > 1)
        error_value = Math.Abs(curve3y[nb_p - 1]); // valeur absolue du dernier element
      return new ErrorInfo(curve3,error_value);
    }
  }
}
