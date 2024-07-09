//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

#if ARCANE_64BIT
using Integer = System.Int64;
using IntegerConstArrayView = Arcane.Int64ConstArrayView;
using IntegerArrayView = Arcane.Int64ArrayView;
#else
using Integer = System.Int32;
using IntegerConstArrayView = Arcane.Int32ConstArrayView;
using IntegerArrayView = Arcane.Int32ArrayView;
#endif

namespace Arcane
{
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ItemVectorView<_ItemKind>
    where _ItemKind : IItem, new()
  {
    internal ItemInternalArrayView m_items;
    internal ItemIndexArrayView m_local_ids;

    public ItemVectorView(ItemInternalArrayView items,ItemIndexArrayView indexes)
    {
      m_items = items;
      m_local_ids = indexes;
    }
    public ItemVectorView(ItemVectorView gen_view)
    {
      m_items = gen_view.Items;
      m_local_ids = gen_view.Indexes;
    }

    public Integer Size { get { return m_local_ids.Length; } }

    public _ItemKind this[Integer index]
    {
      get
      {
        _ItemKind item = new _ItemKind();
        item.ItemBase = m_items[m_local_ids[index]].ItemBase;
        return item;
      }
    }

    public Int32ConstArrayView LocalIds
    {
      get { return m_local_ids.LocalIds; }
    }

    public ItemIndexArrayView Indexes
    {
      get { return m_local_ids; }
    }

    public ItemInternalArrayView Items
    {
      get { return m_items; }
    }

    public IndexedItemEnumerator<_ItemKind> GetEnumerator()
    {
      return new IndexedItemEnumerator<_ItemKind>(m_items.m_ptr,m_local_ids.m_local_ids._InternalData(),m_local_ids.Length);
    }

    public ItemVectorView<_ItemKind> SubViewInterval(Integer interval,Integer nb_interval)
    {
      return new ItemVectorView<_ItemKind>(m_items,m_local_ids.SubViewInterval(interval,nb_interval));
    }
  }
}
