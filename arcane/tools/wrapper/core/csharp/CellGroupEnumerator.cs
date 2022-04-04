//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Runtime.InteropServices;

#if ARCANE_64BIT
using Integer = System.Int64;
#else
using Integer = System.Int32;
#endif

namespace Arcane
{
 [StructLayout(LayoutKind.Sequential)]
 public unsafe class CellGroupEnumerator
 {
  public Integer m_current;
  public Integer m_end;
  public Int32* m_items_local_ids;
  public ItemInternal** m_items;
 }

 [StructLayout(LayoutKind.Sequential)]
 public unsafe class ItemGroupEnumerator
 {
  public Integer m_current;
  public Integer m_end;
  public Int32* m_items_local_ids;
  public ItemInternal** m_items;

  public void Reset()
  {
    m_current = -1;
  }

  public Item Current
  {
    get{ return new Item(m_items[m_items_local_ids[m_current]]); }
  }

  public bool MoveNext()
  {
    ++m_current;
    return m_current<m_end;
  }
 }

 [StructLayout(LayoutKind.Sequential)]
 public unsafe struct NodeEnumerator
 {
  public ItemInternal** m_items;
  public Int32* m_local_ids;
  public Integer m_index;
  public Integer m_count;
 }
}
