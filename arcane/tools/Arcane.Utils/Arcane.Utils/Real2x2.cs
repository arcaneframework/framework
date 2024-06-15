//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Arcane
{
  [StructLayout(LayoutKind.Sequential)]
  public struct Real2x2
  {  
    public static readonly Real2x2 Zero = new Real2x2(Real2.Zero,Real2.Zero);

    public Real2 x; //!< première composante du couple
    public Real2 y; //!< deuxième composante du couple

    public Real2x2(Real2 _x,Real2 _y)
    {
      x = _x;
      y = _y;
    }
    public override string ToString()
    {
      return "("+x+","+y+")";
    }
  }
}
