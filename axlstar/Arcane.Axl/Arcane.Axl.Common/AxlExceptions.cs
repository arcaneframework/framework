//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;

namespace Arcane.Axl
{
  #region EXCEPTION
  [Serializable]
  public class AxlToolException : Exception
  {
    public AxlToolException ()
    {
    }
    
    public AxlToolException (string message)
      : base(message)
    {
      ;
    }
  }
  
  [Serializable]
  public class AxlToolExit : Exception
  {
    public AxlToolExit ()
    {
    }
    
    public AxlToolExit (string message)
      : base(message)
    {
      ;
    }
  }
  #endregion
}
