//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;

namespace Arcane.Curves
{
  public class ErrorStrategy
  {
    private DrbarError m_drbar;
    private DriftError m_drift;
    private RelativeError m_relative;
    private RelativeMaxError m_relative_max;
    ChronometryError m_chronometry;
    MaxError m_max;

    public ErrorStrategy()
    {
      m_drbar = new DrbarError();
      m_drift = new DriftError();
      m_relative = new RelativeError();
      m_relative_max = new RelativeMaxError();
      m_chronometry = new ChronometryError();
      m_max = new MaxError();
    }

    /// <summary>
    /// Compare la courbe de reference \a ref_curve avec la courbe cible \a target_curve
    /// et retourne la courbe de comparaison ainsi que la valeur de l'erreur.
    /// La courbe de comparaison aura pour nom \a name.
    /// La comparaison sera faite avec le comparateur de nom \a error_type.
    /// </summary>
    /// <param name="ref_curve">
    /// A <see cref="ICurve"/>
    /// </param>
    /// <param name="target_curve">
    /// A <see cref="ICurve"/>
    /// </param>
    /// <param name="name">
    /// A <see cref="System.String"/>
    /// </param>
    /// <returns>
    /// A <see cref="ErrorInfo"/>
    /// </returns>
    public ErrorInfo Compute(string error_type, ICurve ref_curve, ICurve target_curve, string curve_name)
    {
      return Compute(error_type, ref_curve, target_curve, curve_name, new ErrorStrategyArguments());
    }

    /// <summary>
    /// Compare la courbe de reference \a ref_curve avec la courbe cible \a target_curve
    /// et retourne la courbe de comparaison ainsi que la valeur de l'erreur.
    /// La courbe de comparaison aura pour nom \a name.
    /// La comparaison sera faite avec le comparateur de nom \a error_type.
    /// Le paramètre \a args contient d'éventuels arguments nécessaires suivant le type de comparaison
    /// </summary>
    /// <param name="ref_curve">
    /// A <see cref="ICurve"/>
    /// </param>
    /// <param name="target_curve">
    /// A <see cref="ICurve"/>
    /// </param>
    /// <param name="name">
    /// A <see cref="System.String"/>
    /// </param>
    /// <returns>
    /// A <see cref="ErrorInfo"/>
    /// </returns>
    public ErrorInfo Compute(string error_type, ICurve ref_curve, ICurve target_curve, string curve_name, ErrorStrategyArguments args)
    {
      IErrorStrategy s = _FindFrom(error_type);
      return s.Compute(ref_curve, target_curve, curve_name, args);
    }
    IErrorStrategy _FindFrom(string error_type)
    {
      if (String.IsNullOrEmpty(error_type))
        return m_relative_max;
      if (error_type == "RelativeError")
        return m_relative;
      if (error_type == "DrbarError")
        return m_drbar;
      if (error_type == "DriftError")
        return m_drift;
      if (error_type == "ChronometryError")
        return m_chronometry;
      if (error_type == "RelativeMaxError")
        return m_relative_max;
      if (error_type == "MaxError")
        return m_max;
      throw new ArgumentException(String.Format ("Bad value '{0}' for error_type",error_type));
    }
  }
}
