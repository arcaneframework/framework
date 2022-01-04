//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using Arcane;

namespace Arcane.Curves
{
  /// <summary>
  /// Interface d'une courbe
  /// </summary>
  public interface ICurve
  {
    /// <summary>
    /// Nom de la courbe
    /// </summary>
    string Name { get; }
    /// <summary>
    /// Tableau des abscisses
    /// </summary>
    RealConstArrayView X { get; }
    /// <summary>
    /// Tableau des ordonnees
    /// </summary>
    RealConstArrayView Y { get; }
    /// <summary>
    /// Nombre de points de la courbe
    /// </summary>
    int NbPoint { get; }
  }
}
