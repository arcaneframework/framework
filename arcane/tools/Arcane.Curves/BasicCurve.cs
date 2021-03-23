//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

using System;

namespace Arcane.Curves
{
  /// <summary>
  /// Classe de base d'une courbe. Lors de la construction, la courbe devient proprietaire des tableaux X et Y.
  /// </summary>
  internal class BasicCurve : ICurve
  {
    RealArray m_x;
    RealArray m_y;
    string m_name;

    public string Name { get { return m_name; } }
    public RealConstArrayView X { get { return m_x.ConstView; } }
    public RealConstArrayView Y { get { return m_y.ConstView; } }
    public int NbPoint { get { return m_x.Size; } }
    public BasicCurve(string name,RealArray x,RealArray y)
		{
      m_x = x;
      m_y = y;
      m_name = name;
		}
	}
}
