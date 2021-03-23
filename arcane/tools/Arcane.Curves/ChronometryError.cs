//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
﻿using System;

namespace Arcane.Curves
{
  /*!
   * \brief Calcule une erreur de chronométrie.
   *
   * Pour le calcul, on suppose que l'absisse est un temps physique. Pour chaque courbe
   * on cherche à quelle absisse correspond la valeur maximale (en valeur absolue) de la courbe. L'erreur de chronométrie
   * est la différence entre ces deux absices.
   */
  public class ChronometryError : NoArgsErrorStrategy
  {
    public override ErrorInfo Compute (ICurve ref_curve, ICurve target_curve, string name, ErrorStrategyArguments args)
    {
      RealArray pgr = null;

      ICurve pcrb1 = null;
      ICurve pcrb2 = null;

      Utils.ComputeProjection (ref_curve, target_curve, args.MinX, args.MaxX, out pcrb1, out pcrb2, out pgr);
      int nbp = pgr.Length;
      RealArray curve3x = pgr;
      RealArray curve3y = new RealArray (nbp);
      RealConstArrayView ref_y = pcrb1.Y;
      RealConstArrayView target_y = pcrb2.Y;

      RealConstArrayView ref_x = pcrb1.X;
      RealConstArrayView target_x = pcrb2.X;

      double ref_x_for_min = ref_x[0];
      double target_x_for_min = Math.Abs(target_x[0]);
      double max_ref= ref_y[0];
      double max_target = Math.Abs(target_y[0]);

      for (int i = 0; i < nbp; ++i) {
        double abs_ref_y = Math.Abs(ref_y[i]);
        double abs_target_y = Math.Abs(target_y[i]);
        if (abs_ref_y>max_ref){
          max_ref = abs_ref_y;
          ref_x_for_min = ref_x[i];
        }
        if (abs_target_y>max_target){
          max_target = abs_target_y;
          target_x_for_min = target_x[i];
        }
        curve3y [i] = target_x_for_min - ref_x_for_min;
      }

      ICurve curve3 = new BasicCurve (name, curve3x, curve3y);
      double error_value = Math.Abs(target_x_for_min - ref_x_for_min);
      return new ErrorInfo (curve3, error_value);
    }
  }
}
