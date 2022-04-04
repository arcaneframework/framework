//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
namespace Arcane.Curves
{
  public class ErrorInfo
  {
    public ErrorInfo (ICurve curve, double err_value)
    {
      m_curve = curve;
      m_error_value = err_value;
    }

    private ICurve m_curve;
    public ICurve Curve { get { return m_curve; } }

    private double m_error_value;
    public double ErrorValue { get { return m_error_value; } }
  }

  public interface IErrorStrategy
  {
    /// <summary>
    /// Compare la courbe de reference \a ref_curve avec la courbe cible \a target_curve
    /// et retourne la courbe de comparaison ainsi que la valeur de l'erreur.
    /// La courbe de comparaison aura pour nom \a name
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
    ErrorInfo Compute (ICurve ref_curve, ICurve target_curve, string name);

    ErrorInfo Compute (ICurve ref_curve, ICurve target_curve, string name, ErrorStrategyArguments args);
  }
}
