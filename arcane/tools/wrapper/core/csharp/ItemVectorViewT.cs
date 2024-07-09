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
    //internal ItemInternalArrayView m_items;
    internal ItemIndexArrayView m_local_ids;
    internal ItemSharedInfo* m_shared_info;

    public ItemVectorView()
    {
      m_shared_info = ItemSharedInfo.Zero;
      m_local_ids = new ItemIndexArrayView();
    }
    public ItemVectorView(ItemVectorView gen_view)
    {
      //m_items = gen_view._Items;
      m_local_ids = gen_view.Indexes;
      m_shared_info = gen_view.m_shared_info;
    }

    [Obsolete("Use another constructor")]
    public ItemVectorView(ItemInternalArrayView items, ItemIndexArrayView indexes)
    {
      //m_items = items;
      m_local_ids = indexes;
      bool is_valid = (m_local_ids.Size > 0 && items.Size > 0);
      m_shared_info = (is_valid) ? items.m_ptr[0]->m_shared_info : ItemSharedInfo.Zero;
    }
    internal ItemVectorView(ItemSharedInfo* shared_info, ItemIndexArrayView indexes)
    {
      m_local_ids = indexes;
      m_shared_info = shared_info;
    }

    public Integer Size { get { return m_local_ids.Length; } }

    public _ItemKind this[Integer index]
    {
      get
      {
        _ItemKind item = new _ItemKind();
        item.ItemBase = new ItemBase(m_shared_info, m_local_ids[index]);
        return item;
      }
    }

    public ItemIndexArrayView Indexes { get { return m_local_ids; } }

    public IndexedItemEnumerator<_ItemKind> GetEnumerator()
    {
      return new IndexedItemEnumerator<_ItemKind>(m_shared_info->m_items_internal.m_ptr, m_local_ids.m_local_ids._InternalData(), m_local_ids.Length);
    }

    public ItemVectorView<_ItemKind> SubViewInterval(Int32 interval, Int32 nb_interval)
    {
      return new ItemVectorView<_ItemKind>(m_shared_info, m_local_ids.SubViewInterval(interval, nb_interval));
    }

    [Obsolete("Use Indexes property instead")]
    public Int32ConstArrayView LocalIds { get { return m_local_ids.LocalIds; } }

    [Obsolete("This method is internal to Arcane. Use ItemBase() instead.")]
    public ItemInternalArrayView Items { get { return new ItemInternalArrayView(m_shared_info->m_items_internal); } }

    // TODO: pour compatibilité avec l'existant. A supprimer
    internal Int32ConstArrayView _LocalIds { get { return m_local_ids.LocalIds; } }
  }
}
