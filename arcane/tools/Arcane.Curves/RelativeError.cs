//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
﻿
namespace Arcane.Curves
{
  public class RelativeError : NoArgsErrorStrategy
  {
    public override ErrorInfo Compute(ICurve ref_curve, ICurve target_curve,string name, ErrorStrategyArguments args)
    {
      RealArray pgr = null;

      ICurve pcrb1 = null;
      ICurve pcrb2 = null;

      Utils.ComputeProjection(ref_curve,target_curve,args.MinX, args.MaxX, out pcrb1,out pcrb2,out pgr);
      int nbp = pgr.Length;
      RealArray curve3x = pgr;
      RealArray curve3y = new RealArray(nbp);
      RealConstArrayView pcrb1y = pcrb1.Y;
      RealConstArrayView pcrb2y = pcrb2.Y;

      for( int i=0; i<nbp; ++i){
        double val = pcrb1y[i];
        if (val == 0.0)
          val = 1.0;
        curve3y[i] = (pcrb2y[i] - pcrb1y[i]) / val;
      }
      
      ICurve curve3 = new BasicCurve(name,curve3x,curve3y);
      double error_value = Utils.NormeInf(curve3);
      return new ErrorInfo(curve3,error_value);
    }
  }
}

