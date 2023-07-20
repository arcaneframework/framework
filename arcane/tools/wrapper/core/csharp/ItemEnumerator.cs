//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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

  //WARNING: cette structure doit avoir le meme layout en C++ avec ItemGroupRangeIterator
  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ItemEnumerator2
  {
    internal Integer m_current;
    internal Integer m_end;
    internal Int32* m_local_ids;
    internal ItemInternal** m_items;

    internal ItemEnumerator2(ItemInternal** items,Int32* local_ids,Integer end)
    {
      m_items = items;
      m_local_ids = local_ids;
      m_current = -1;
      m_end = end;
    }

    public void Reset()
    {
      m_current = -1;
    }

    public Item Current
    {
      get{ return new Item(new ItemBase(m_items[m_local_ids[m_current]])); }
    }

    public bool MoveNext()
    {
      ++m_current;
      return m_current<m_end;
    }
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ItemEnumerator<_ItemKind> where _ItemKind : IItem, new()
  {
    //NOTE: normalement il s'agit d'un 'ItemInternal**' mais cela plante
    // avec les versions 2.10.* de mono (marche avec 2.8.1)
    IntPtr* m_items;
    Int32* m_local_ids;
    Integer m_current;
    Integer m_end;
    _ItemKind m_true_type;

    internal ItemEnumerator(ItemInternal** items,Int32* local_ids,Integer end)
    {
      m_items = (IntPtr*)items;
      m_local_ids = local_ids;
      m_current = -1;
      m_end = end;
      m_true_type = new _ItemKind();
    }

    public void Reset()
    {
      m_current = -1;
    }

    public _ItemKind Current
    {
      get{ return m_true_type; }
    }

    public bool MoveNext()
    {
      ++m_current;
      if (m_current>=m_end)
        return false;
      m_true_type.ItemBase = new ItemBase((ItemInternal*)m_items[m_local_ids[m_current]]);
      return true;
    }
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct IndexedItemEnumerator<_ItemKind> where _ItemKind : IItem, new()
  {
    Int32 m_current;
    Integer m_end;
    Int32* m_local_ids;
    ItemInternal** m_items;

    IndexedItem<_ItemKind> m_current_item;

    internal IndexedItemEnumerator(ItemInternal** items,Int32* local_ids,Integer end)
    {
      m_items = items;
      m_local_ids = local_ids;
      m_current = -1;
      m_end = end;
      m_current_item = new IndexedItem<_ItemKind>(new ItemBase(),0);
    }

    public IndexedItemEnumerator(ItemEnumerator ie)
    {
      m_items = ie.m_items;
      m_local_ids = ie.m_local_ids;
      m_current = -1;
      m_end = ie.m_end;
      m_current_item = new IndexedItem<_ItemKind>(new ItemBase(),0);
    }

    public void Reset()
    {
      m_current = -1;
    }

    public IndexedItem<_ItemKind> Current
    {
      get{ m_current_item._Set(m_items[m_local_ids[m_current]],m_current); return m_current_item; }
    }

    public bool MoveNext()
    {
      ++m_current;
      return m_current<m_end;
    }
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct IndexedNodeEnumerator
  {
    Int32 m_current;
    Integer m_end;
    Int32* m_local_ids;
    ItemInternal** m_items;

    internal IndexedNodeEnumerator(ItemInternal** items,Int32* local_ids,Integer end)
    {
      m_items = items;
      m_local_ids = local_ids;
      m_current = -1;
      m_end = end;
    }

    public IndexedNodeEnumerator(ItemEnumerator ie)
    {
      m_items = ie.m_items;
      m_local_ids = ie.m_local_ids;
      m_current = -1;
      m_end = ie.m_end;
    }

    public void Reset()
    {
      m_current = -1;
    }

    public IndexedNode Current
    {
      get{ return new IndexedNode(new ItemBase(m_items[m_local_ids[m_current]]),m_current); }
    }

    public bool MoveNext()
    {
      ++m_current;
      return m_current<m_end;
    }
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ItemList<_ItemKind> where _ItemKind : IItem, new()
  {
    ItemInternal** m_items;
    Int32* m_local_ids;
    Integer m_end;

    internal ItemList(ItemInternal** items,Int32* local_ids,Integer end)
    {
      m_items = items;
      m_local_ids = local_ids;
      m_end = end;
    }

    public IndexedItemEnumerator<_ItemKind> GetEnumerator()
    {
      return new IndexedItemEnumerator<_ItemKind>(m_items,m_local_ids,m_end);
    }
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ItemIndexArrayView
  {
    internal Int32ConstArrayView m_local_ids;
    internal Int32 m_flags;

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

    public Integer Size { get { return m_local_ids.Length; } }
    public Integer Length { get { return m_local_ids.Length; } }

    public Int32 this[Integer index]
    {
      get
      {
        return m_local_ids[index];
      }
    }

    public Int32ConstArrayView LocalIds
    {
      get { return m_local_ids; }
    }

    public ItemIndexArrayView SubViewInterval(Integer interval,Integer nb_interval)
    {
      return new ItemIndexArrayView(m_local_ids.SubViewInterval(interval,nb_interval));
    }
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ItemVectorView<_ItemKind>
    where _ItemKind : IItem, new()
  {
    internal ItemInternalArrayView m_items;
    internal ItemIndexArrayView m_local_ids;

    public ItemVectorView(ItemInternalArrayView items,Int32ConstArrayView local_ids)
    {
      m_items = items;
      m_local_ids = new ItemIndexArrayView(local_ids);
    }
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
  
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct NodeList
  {
    ItemInternal** m_items;
    Int32* m_local_ids;
    Integer m_end;

    public NodeList(ItemInternal** items,Int32* local_ids,Integer end)
    {
      m_items = items;
      m_local_ids = local_ids;
      m_end = end;
    }

    public IndexedNodeEnumerator GetEnumerator()
    {
      return new IndexedNodeEnumerator(m_items,m_local_ids,m_end);
    }
  }
  
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ItemPairList<_ItemKind1,_ItemKind2>
    where _ItemKind1 : IItem, new()
    where _ItemKind2 : IItem, new()
  {
    internal ItemInternal** m_sub_items;
    internal Int32* m_sub_local_ids;
    internal Integer m_nb_sub_item;
    internal Integer m_index;
    internal _ItemKind1 m_item;

    public IndexedItemEnumerator<_ItemKind2> GetEnumerator()
    {
      return new IndexedItemEnumerator<_ItemKind2>(m_sub_items,m_sub_local_ids,m_nb_sub_item);
    }

    public Integer NbItem
    {
      get { return m_nb_sub_item; }
    }

    //! Entité parente de la liste
    public _ItemKind1 Item { get { return m_item; } }

    //! Indice de l'entité parente dans la liste parente (si tableau 2D)
    public Integer Index { get { return m_index; } }
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  public unsafe class ItemPairEnumerator<_ItemKind1,_ItemKind2>
  : IEnumerator< ItemPairList<_ItemKind1,_ItemKind2> >
    where _ItemKind1 : IItem, new()
    where _ItemKind2 : IItem, new()
  {  
    Integer m_current;
    Integer m_end;
    IntegerConstArrayView m_indexes;
    Int32ConstArrayView m_items_local_id;
    Int32ConstArrayView m_sub_items_local_id;
    ItemInternalList m_items_internal;
    ItemInternalList m_sub_items_internal;
    ItemPairList<_ItemKind1,_ItemKind2> m_pair;

    public ItemPairEnumerator(ItemPairEnumerator e)
    {
      m_current = e.m_current;
      m_end = e.m_end;
      m_indexes = e.m_indexes;
      m_items_local_id = e.m_items_local_id;
      m_sub_items_local_id = e.m_sub_items_local_id;
      m_items_internal = e.m_items_internal;
      m_sub_items_internal = e.m_sub_items_internal;

      m_pair = new ItemPairList<_ItemKind1,_ItemKind2>();
      m_pair.m_sub_items = m_sub_items_internal.m_ptr;
    }

    public ItemPairList<_ItemKind1,_ItemKind2> Current
    {
      get { return m_pair; }
    }

    ItemPairList<_ItemKind1,_ItemKind2> IEnumerator< ItemPairList<_ItemKind1,_ItemKind2> >.Current
    {
      get { return Current; }
    }

    object System.Collections.IEnumerator.Current
    {
      get { return Current; }
    }
    
    public void Reset()
    {
      m_current = -1;
    }

    public bool MoveNext()
    {
      ++m_current;
      if (m_current>=m_end)
        return false;
      m_pair.m_sub_local_ids = m_sub_items_local_id._InternalData()+m_indexes[m_current];
      m_pair.m_nb_sub_item = m_indexes[m_current+1]-m_indexes[m_current];
      m_pair.m_index = m_current;
      m_pair.m_item.ItemBase = new ItemBase(m_items_internal.m_ptr[m_current]);
      return true;
    }
    void IDisposable.Dispose(){}
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  [StructLayout(LayoutKind.Sequential)]
  public struct ItemPair
  {
    Item m_first;
    Item m_second;
  
    public ItemPair(Item first,Item second)
    {
      m_first = first;
      m_second = second;
    }
    public Item First { get { return m_first; } }
    public Item Second { get { return m_second; } }
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  [StructLayout(LayoutKind.Sequential)]
  public struct ItemPair<_ItemKind1,_ItemKind2>
    where _ItemKind1 : IItem, new()
    where _ItemKind2 : IItem, new()
  {
    _ItemKind1 m_first;
    _ItemKind2 m_second;
    
    public ItemPair(_ItemKind1 first,_ItemKind2 second)
    {
      m_first = first;
      m_second = second;
    }
    public _ItemKind1 First { get { return m_first; } }
    public _ItemKind2 Second { get { return m_second; } }
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  public class ItemPairGroup<_ItemKind1,_ItemKind2>
  : IEnumerable< ItemPairList<_ItemKind1,_ItemKind2> >
  , IEnumerable< ItemPair<_ItemKind1,_ItemKind2> >
    where _ItemKind1 : IItem, new()
    where _ItemKind2 : IItem, new()
  {
    ItemPairGroup m_array;

    public ItemPairGroup(Arcane.ItemPairGroup a)
    {
      Item.CheckSameKind(a.ItemKind(),(new _ItemKind1()).Kind,
                         "invalid item_kind for type 1");
      Item.CheckSameKind(a.SubItemKind(),(new _ItemKind2()).Kind,
                         "invalid item_kind for type 2");
      m_array = a;
    }

    public ItemPairEnumerator<_ItemKind1,_ItemKind2> GetEnumerator()
    {
      return new ItemPairEnumerator<_ItemKind1,_ItemKind2>(m_array.GetEnumerator());
    }

    public IEnumerable< ItemPair<_ItemKind1,_ItemKind2> > Pairs()
    {
      //IEnumerable< ItemPairList<_ItemKind1,_ItemKind2> > v = this;
      foreach(ItemPairList<_ItemKind1,_ItemKind2> i in this){
        foreach(_ItemKind2 s in i)
          yield return new ItemPair<_ItemKind1,_ItemKind2>(i.Item,s);
      }
    }
    IEnumerator< ItemPair<_ItemKind1,_ItemKind2> >
      IEnumerable< ItemPair<_ItemKind1,_ItemKind2> >.GetEnumerator()
    {
      return Pairs().GetEnumerator();
    }

    IEnumerator< ItemPairList<_ItemKind1,_ItemKind2> >
      IEnumerable< ItemPairList<_ItemKind1,_ItemKind2> >.GetEnumerator()
    {
      return GetEnumerator();
    }

    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
    {
      return GetEnumerator();
    }

  }
}
