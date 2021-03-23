//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

using System;
using System.IO;

namespace Arcane.Curves
{


  public class CaseSmoother
  {
    int m_nb_point;
    string m_case_path;
    public CaseSmoother (string case_path,int nb_point)
    {
      m_nb_point = nb_point;
      m_case_path = case_path;
    }

    public void Smooth()
    {
      CaseCurves curves = CaseCurves.ReadCase(m_case_path);
      Smoother smoother = new Smoother(m_nb_point);
      string dir_name = "smooth";
      Directory.CreateDirectory(dir_name);
      GnuplotCurveWriter gnuplot_writer = new GnuplotCurveWriter(dir_name);
      foreach(ICaseCurve ccv in curves.Curves){
        ICurve cv = ccv.Read();
        ICurve new_curve = smoother.Apply(cv);
        gnuplot_writer.Write(new_curve);
      }
    }
  }
}
