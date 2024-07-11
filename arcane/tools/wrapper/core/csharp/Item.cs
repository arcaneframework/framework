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
  //-----------------------------------------------------------------------------

  public unsafe interface IItem
  {
    [Obsolete("This method is internal to Arcane. Use ItemBase() instead.")]
    ItemInternal* Internal { get; set; }

    ItemBase ItemBase { get; internal set; }

    Int32 LocalId { get; }
    Int64 UniqueId { get; }
    bool IsNull { get; }
    eItemKind Kind { get; }
  }

  //-----------------------------------------------------------------------------

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct Item : IItem
  {
    private ItemBase m_item_base;

    [Obsolete("This method is internal to Arcane. Use constructor with ItemBase() instead.")]
    public Item(ItemInternal* ii)
    {
      m_item_base = new ItemBase(ii);
    }
    [Obsolete("This method is internal to Arcane. Use ItemBase() instead.")]
    public ItemInternal* Internal
    {
      get { return m_item_base.Internal; }
      set { m_item_base = new ItemBase(value); }
    }

    public Item(ItemBase ii)
    {
      m_item_base = ii;
    }
    public ItemBase ItemBase
    {
      get { return m_item_base; }
      set { m_item_base = value; }
    }

    public Item(Cell cell)
    {
      m_item_base = cell.ItemBase;
    }
    public Item(Face face)
    {
      m_item_base = face.ItemBase;
    }
    public Item(Node node)
    {
      m_item_base = node.ItemBase;
    }
    public Int32 LocalId
    {
      get { return m_item_base.LocalId; }
    }
    public Int64 UniqueId
    {
      get { return m_item_base.UniqueId; }
    }
    public eItemKind Kind
    {
      get { return m_item_base.Kind; }
    }
    public Integer NbNode
    {
      get { return m_item_base.NbNode; }
    }
    public Node Node(Integer index)
    {
      return new Node(m_item_base.Node(index));
    }
    public Int32 NodeLocalId(Integer index)
    {
      return m_item_base.NodeLocalId(index);
    }
    public Node ToNode()
    {
      return new Node(m_item_base);
    }
    public Face ToFace()
    {
      return new Face(m_item_base);
    }
    public Cell ToCell()
    {
      return new Cell(m_item_base);
    }
    public bool IsNull
    {
      get { return m_item_base.IsNull; }
    }

    static public void CheckSameKind(eItemKind k1,eItemKind k2,string message)
    {
      if (k1!=k2)
        throw new ArgumentException(String.Format("item_kind differs: k1={0} k2={1} msg={2}",
                                                  k1,k2,message));
    }

  }

  //-----------------------------------------------------------------------------

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct Cell : IItem
  {
    private ItemBase m_item_base;

    [Obsolete("This method is internal to Arcane. Use ItemBase() instead.")]
    public ItemInternal* Internal
    {
      get { return m_item_base.Internal; }
      set { m_item_base = new ItemBase(value); }
    }

    [Obsolete("Use Cell(ItemBase) instead.")]
    public Cell(Item ii)
    {
      m_item_base = ii.ItemBase;
    }

    public Cell(ItemBase ii)
    {
      m_item_base = ii;
    }
    public ItemBase ItemBase
    {
      get { return m_item_base; }
      set { m_item_base = value; }
    }

    public bool IsNull
    {
      get { return m_item_base.IsNull; }
    }
    public Int32 LocalId
    {
      get { return m_item_base.LocalId; }
    }
    public eItemKind Kind
    {
      get { return eItemKind.IK_Cell; }
    }

    public Integer NbNode
    {
      get { return m_item_base.NbNode; }
    }
    public Node Node(Integer index)
    {
      return new Node(m_item_base.Node(index));
    }
    public Int32 NodeLocalId(Integer index)
    {
      return m_item_base.NodeLocalId(index);
    }
    public NodeList Nodes
    {
      get { return m_item_base.Nodes; }
    }

    public Integer NbFace
    {
      get { return m_item_base.NbFace; }
    }
    public Face Face(Integer index)
    {
      return new Face(m_item_base.Face(index));
    }
    public Int32 FaceLocalId(Integer index)
    {
      return m_item_base.FaceLocalId(index);
    }
    public ItemList<Face> Faces
    {
      get { return m_item_base.Faces; }
    }

    public Int64 UniqueId
    {
      get { return m_item_base.UniqueId; }
    }
  }

  //-----------------------------------------------------------------------------

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct Face : IItem
  {
    private ItemBase m_item_base;

    [Obsolete("This method is internal to Arcane. Use ItemBase() instead.")]
    public ItemInternal* Internal
    {
      get { return m_item_base.Internal; }
      set { m_item_base = new ItemBase(value); }
    }

    public Face(ItemBase ii)
    {
      m_item_base = ii;
    }
    public ItemBase ItemBase
    {
      get { return m_item_base; }
      set { m_item_base = value; }
    }

    [Obsolete("Use Face(ItemBase) instead")]
     public Face(Item ii)
    {
      m_item_base = ii.ItemBase;
    }
    public bool IsNull
    {
      get { return m_item_base.IsNull; }
    }
    public Int32 LocalId
    {
      get { return m_item_base.LocalId; }
    }
    public eItemKind Kind
    {
      get { return eItemKind.IK_Face; }
    }

    public Integer NbNode
    {
      get { return m_item_base.NbNode; }
    }
    public Node Node(Integer index)
    {
      return new Node(m_item_base.Node(index));
    }
    public Int32 NodeLocalId(Integer index)
    {
      return m_item_base.NodeLocalId(index);
    }
    public NodeList Nodes
    {
      get { return m_item_base.Nodes; }
    }

    public Integer NbCell
    {
      get { return m_item_base.NbCell; }
    }
    public Cell Cell(Integer index)
    {
      return new Cell(m_item_base.Cell(index));
    }
    public Int32 CellLocalId(Integer index)
    {
      return m_item_base.CellLocalId(index);
    }
    public ItemList<Cell> Cells
    {
      get { return m_item_base.Cells; }
    }

    public Cell BackCell
    {
      get { return new Cell(m_item_base.BackCell); }
    }

    public Cell FrontCell
    {
      get { return new Cell(m_item_base.FrontCell); }
    }

    public Int64 UniqueId
    {
      get { return m_item_base.UniqueId; }
    }
  }

  //-----------------------------------------------------------------------------

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct Edge : IItem
  {
    private ItemBase m_item_base;

    [Obsolete("This method is internal to Arcane. Use ItemBase() instead.")]
    public ItemInternal* Internal
    {
      get { return m_item_base.Internal; }
      set { m_item_base = new ItemBase(value); }
    }

    public Edge(ItemBase ii)
    {
      m_item_base = ii;
    }
    public ItemBase ItemBase
    {
      get { return m_item_base; }
      set { m_item_base = value; }
    }

    public bool IsNull
    {
      get { return m_item_base.IsNull; }
    }
    public Int32 LocalId
    {
      get { return m_item_base.LocalId; }
    }
    public eItemKind Kind
    {
      get { return eItemKind.IK_Edge; }
    }

    public Int64 UniqueId
    {
      get { return m_item_base.UniqueId; }
    }
  }

  //-----------------------------------------------------------------------------

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct Node : IItem
  {
    ItemBase m_item_base;

    [Obsolete("Use Node(ItemBase) instead.")]
    public Node(Item ii)
    {
      m_item_base = ii.ItemBase;
    }

    [Obsolete("This method is internal to Arcane. Use ItemBase() instead.")]
    public ItemInternal* Internal
    {
      get { return m_item_base.Internal; }
      set { m_item_base = new ItemBase(value); }
    }

    public Node(ItemBase ii)
    {
      m_item_base = ii;
    }
    public ItemBase ItemBase
    {
      get { return m_item_base; }
      set { m_item_base = value; }
    }

    public bool IsNull
    {
      get { return m_item_base.IsNull; }
    }
    public Int32 LocalId
    {
      get { return m_item_base.LocalId; }
    }
    public Int64 UniqueId
    {
      get { return m_item_base.UniqueId; }
    }
    public eItemKind Kind
    {
      get { return eItemKind.IK_Node; }
    }

    public Cell Cell(Integer index)
    {
      return new Cell(m_item_base.Cell(index));
    }
    public Int32 CellLocalId(Integer index)
    {
      return m_item_base.CellLocalId(index);
    }
    public Integer NbCell
    {
      get { return m_item_base.NbCell; }
    }
    public ItemList<Cell> Cells
    {
      get { return m_item_base.Cells; }
    }
  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct IndexedItem<_ItemKind> where _ItemKind : IItem, new()
  {
    private Integer m_index;
    private _ItemKind m_item;

    public static implicit operator _ItemKind(IndexedItem<_ItemKind> m)
    {
      return m.m_item;
    }

    public _ItemKind Item { get { return m_item; } }

    public Integer Index { get { return m_index; } }

    public IndexedItem(ItemBase ii,Integer index)
    {
      m_item = new _ItemKind();
      m_item.ItemBase = ii;
      m_index = index;
    }

    public IndexedItem(_ItemKind item,Integer index)
    {
      m_item = item;
      m_index = index;
    }

    internal void _Set(ItemBase ii,Integer index)
    {
      m_item.ItemBase = ii;
      m_index = index;
    }
  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct IndexedNode
  {
    private Integer m_index;
    private Node m_item;

    public static implicit operator Node(IndexedNode m)
    {
      return m.m_item;
    }

    public Node Item { get { return m_item; } }

    public Integer Index { get { return m_index; } }

    public IndexedNode(ItemBase ii,Integer index)
    {
      m_item = new Node(ii);
      m_index = index;
    }

    public IndexedNode(Node item,Integer index)
    {
      m_item = item;
      m_index = index;
    }
  }
}
