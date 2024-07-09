//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

using System;
using System.Runtime.InteropServices;

namespace Arcane
{
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  // IMPORTANT: vérifier que cette structure est identique à celle du C++
  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ItemLocalIdListContainerView
  {
    Int32* m_local_ids;
    Int32 m_local_id_offset;
    Int32 m_size;
  }
}
