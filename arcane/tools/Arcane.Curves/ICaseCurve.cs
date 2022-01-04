//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
namespace Arcane.Curves
{
  /// <summary>
  ///  Courbe d'un cas.
  /// Les valeurs de la courbe ne sont pas lues tant que Read() n'est pas appele.
  /// </summary>
  public interface ICaseCurve
  {
    /// <summary>
    /// Nom de la courbe
    /// </summary>
    string Name { get; }
    
    /// <summary>
    /// Indique si la courbe est deja lue.
    /// </summary>
    bool IsRead { get; }
    
    /// <summary>
    /// Lit les valeurs de la courbe. Si la courbe est deja lue, rien ne se fait.
    /// </summary>
    /// <returns>
    /// A <see cref="ICurve"/>
    /// </returns>
    ICurve Read();
  }
}

