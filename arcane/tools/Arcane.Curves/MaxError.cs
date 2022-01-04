//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
﻿using System;
namespace Arcane.Curves
{
  /*!
   * \brief Calcule une erreur sur le maximum de la courbe.
   *
   * Cette stratégie retourne la différence entre le maximum de la courbe cible et
   * le maximum de la courbe de référence.
   */
  public class MaxError : NoArgsErrorStrategy
  {
    public override ErrorInfo Compute (ICurve ref_curve, ICurve target_curve, string name,ErrorStrategyArguments args)
    {
      RealArray pgr = null;

      ICurve pcrb1 = null;
      ICurve pcrb2 = null;

      Utils.ComputeProjection(ref_curve, target_curve, args.MinX, args.MaxX, out pcrb1, out pcrb2, out pgr);
      int nbp = pgr.Length;
      RealArray curve3x = pgr;
      RealArray curve3y = new RealArray(nbp);
      RealConstArrayView ref_y = pcrb1.Y;
      RealConstArrayView target_y = pcrb2.Y;

      double max_ref = Math.Abs(ref_y[0]);
      double max_target = Math.Abs(target_y[0]);
      // Calcul comme la valeur absolue du maximum
      for (int i = 0; i < nbp; ++i) {
        max_ref = System.Math.Max(max_ref, Math.Abs(ref_y[i]));
        max_target = System.Math.Max(max_target, Math.Abs(target_y[i]));
        curve3y[i] = max_target - max_ref;
      }

      // Divise par le maximum de la référence (si non nul)
      if (max_ref != 0.0) {
        for (int i = 0; i < nbp; ++i) {
          curve3y[i] = curve3y[i] / max_ref;
        }
      }

      ICurve curve3 = new BasicCurve(name, curve3x, curve3y);
      double error_value = 0.0;
      if (nbp > 1)
        error_value = Math.Abs(curve3y[nbp - 1]); // valeur absolue du dernier element
      return new ErrorInfo(curve3, error_value);
    }
  }
}