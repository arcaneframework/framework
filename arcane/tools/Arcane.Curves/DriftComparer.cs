//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

using System;

namespace Arcane.Curves
{
  public class DriftError : NoArgsErrorStrategy
  {
    public override ErrorInfo Compute(ICurve ref_curve, ICurve target_curve,string name, ErrorStrategyArguments args)
    {
      RealArray pgr = null;

      ICurve pcrb1 = null;
      ICurve pcrb2 = null;

      Utils.ComputeProjection(ref_curve,target_curve,args.MinX,args.MaxX,out pcrb1,out pcrb2,out pgr);
      int nbp = pgr.Length;
      RealArray curve3x = pgr;
      RealArray curve3y = new RealArray(nbp);
      RealConstArrayView pcrb1y = pcrb1.Y;
      RealConstArrayView pcrb2y = pcrb2.Y;
      //Console.WriteLine("NB_P={0} {1}",nbp,pcrb1y.Length);
      if (nbp > 0){
        double Pm = pgr[0]; //curve3.point(0);

        double Tm = Pm;
        double dt = Tm - 0.0;
        if (pcrb1y[0] == 0 || Tm == 0)
          curve3y[0] = 0;
        else
          curve3y[0] = (dt * (pcrb2y[0] - pcrb1y[0]) / pcrb1y[0])/Tm;

        for (int i = 0; i < nbp - 1; ++i){
          double denom = pcrb1y[i + 1];
          double Pp = pgr[i + 1];
          double Tp = Pp;
          dt = Tp - Tm;
          if (denom != 0)
            curve3y[i + 1] = (Tm * curve3y[i] + dt * (pcrb2y[i + 1] - pcrb1y[i + 1]) / denom) / Tp;
          else
            curve3y[i + 1] = Tm * curve3y[i] / Tp;

          Pm = pgr[i + 1];
          Tm = Pm;
        }

        for (int i = 0; i < nbp; ++i)
        {
          curve3y[i] = Math.Abs(curve3y[i]);
        }
      }
      ICurve curve3 = new BasicCurve(name,curve3x,curve3y);
      int nb_p = curve3.NbPoint;
      double error_value = 0.0;
      if (nb_p > 1)
        error_value = Math.Abs(curve3y[nb_p - 1]); // valeur absolue du dernier element
      else
        error_value = 0;
      return new ErrorInfo(curve3,error_value);
    }
  }
}
