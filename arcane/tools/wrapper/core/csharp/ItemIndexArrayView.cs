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
  public unsafe struct ItemIndexArrayView
  {
    internal Int32ConstArrayView m_local_ids;
    internal Int32 m_flags;

    // TODO: Rendre obsolète
    public ItemIndexArrayView(Int32ConstArrayView local_ids)
    {
      m_local_ids = local_ids;
      m_flags = 0;
    }

    public ItemIndexArrayView(Int32ConstArrayView local_ids,Int32 flags)
    {
      m_local_ids = local_ids;
      m_flags = flags;
    }

    public Int32 Size { get { return m_local_ids.Length; } }
    public Int32 Length { get { return m_local_ids.Length; } }

    public Int32 this[Int32 index] { get { return m_local_ids[index]; } }

    public Int32ConstArrayView LocalIds { get { return m_local_ids; } }

    public ItemIndexArrayView SubViewInterval(Int32 interval,Int32 nb_interval)
    {
      return new ItemIndexArrayView(m_local_ids.SubViewInterval(interval,nb_interval));
    }
  }
}
