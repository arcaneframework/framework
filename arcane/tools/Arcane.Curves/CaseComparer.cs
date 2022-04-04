//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

using System;
using System.Collections.Generic;

namespace Arcane.Curves
{
  public class CaseComparer
  {
    CaseCurves m_ref_curves;
    CaseCurves m_target_curves;

    public CaseComparer (string ref_file,string target_file)
    {
      Console.WriteLine("Compare case .NET={0}",Environment.Version);
      m_ref_curves = _ReadCase(ref_file);
      m_target_curves = _ReadCase(target_file);
    }

    CaseCurves _ReadCase(string file)
    {
      return CaseCurves.ReadCase(file);
    }

    public void Compare()
    {
      DriftError dr = new DriftError();
      RelativeMaxError rme = new RelativeMaxError();
      MaxError err_max = new MaxError();

      Dictionary<string,ICurve> ref_curves = new Dictionary<string, ICurve>();
      foreach(ICaseCurve curve in m_ref_curves.Curves)
        ref_curves.Add(curve.Name,curve.Read());

      Console.WriteLine("  {0,-60} {1,-11}  {2,-11}  {3,-11}", "Courbe", "Drift", "RelativeMax", "Max");
      foreach(ICaseCurve case_target_curve in m_target_curves.Curves){
        ICurve target_curve = case_target_curve.Read();
        string name = target_curve.Name;
        ICurve ref_curve = null;
        bool is_found = ref_curves.TryGetValue(target_curve.Name,out ref_curve);
        if (!is_found){
          Console.WriteLine("Curve '{0}' not found in reference",name);
          continue;
        }
        try{
          ErrorInfo err_info = dr.Compute(ref_curve,target_curve,"Compare");
          double drift_error = err_info.ErrorValue;
          err_info = rme.Compute(ref_curve,target_curve,"Compare");
          double rme_error = err_info.ErrorValue;
          err_info = err_max.Compute(ref_curve, target_curve, "Compare");
          double max_error = err_info.ErrorValue;
          Console.WriteLine("  {0,-60} {1}  {2}  {3}",name,drift_error.ToString("E4"),rme_error.ToString("E4"),max_error.ToString("E4"));
        }
        catch(Exception ex){
          Console.WriteLine("Exception catch during comparaison of curve {0} ex={1} stack={2}",name,ex.Message,ex.StackTrace);
        }
      }
    }
  }
}
