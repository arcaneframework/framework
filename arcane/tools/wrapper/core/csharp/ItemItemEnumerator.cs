//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#if false
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

#if ARCANE_64BIT
using Integer = System.Int64;
using IntegerConstArrayView = Arcane.Int64ConstArrayView;
#else
using Integer = System.Int32;
using IntegerConstArrayView = Arcane.Int32ConstArrayView;
#endif

namespace Arcane
{
  [StructLayout(LayoutKind.Sequential)]
  public unsafe class ItemItemEnumerator
  {
    Integer m_current;
    Integer m_end;
    IntegerConstArrayView m_indexes;
    Int32ConstArrayView m_items_local_id;
    Int32ConstArrayView m_sub_items_local_id;
    ItemInternalList m_items_internal;
    ItemInternalList m_sub_items_internal;

    Item Current
    {
      get { return new Item(m_items_internal.m_ptr[m_current]); }
    }

    ItemCellEnumerator SubItems
    {
      get
      {
        return new ItemCellEnumerator(m_sub_items_internal.m_ptr,
                                      m_sub_items_local_id.m_ptr+m_indexes[m_current],
                                      m_indexes[m_current+1]-m_indexes[m_current]
                                      );
      }
    }

  }
}
#endif