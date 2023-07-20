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
  public unsafe interface IItem
  {
    [Obsolete("This method is internal to Arcane")]
    ItemInternal* Internal { get; set; }

    Int32 LocalId { get; }
    Int64 UniqueId { get; }
    bool IsNull { get; }
    eItemKind Kind { get; }
  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct Item : IItem
  {
    private ItemInternal* m_internal;

    [Obsolete("This method is internal to Arcane")]
    public Item(ItemInternal* ii)
    {
      m_internal = ii;
    }
    [Obsolete("This method is internal to Arcane")]
    public ItemInternal* Internal
    {
      get { return m_internal; }
      set { m_internal = value; }
    }

    public Item(Cell cell)
    {
      m_internal = cell.Internal;
    }
    public Item(Face face)
    {
      m_internal = face.Internal;
    }
    public Item(Node node)
    {
      m_internal = node.Internal;
    }
    public Int32 LocalId
    {
      get { return m_internal->m_local_id; }
    }
    public Int64 UniqueId
    {
      get { return m_internal->UniqueId(); }
    }
    public eItemKind Kind
    {
      get { return m_internal->Kind; }
    }
    public Integer NbNode
    {
      get { return m_internal->NbNode; }
    }
    public Node Node(Integer index)
    {
      return new Node(m_internal->Node(index));
    }
    public Int32 NodeLocalId(Integer index)
    {
      return m_internal->NodeLocalId(index);
    }
    public Node ToNode()
    {
      return new Node(m_internal);
    }
    public Face ToFace()
    {
      return new Face(m_internal);
    }
    public Cell ToCell()
    {
      return new Cell(m_internal);
    }
    public bool IsNull
    {
      get { return m_internal->IsNull; }
    }

    static public void CheckSameKind(eItemKind k1,eItemKind k2,string message)
    {
      if (k1!=k2)
        throw new ArgumentException(String.Format("item_kind differs: k1={0} k2={1} msg={2}",
                                                  k1,k2,message));
    }

  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct Cell : IItem
  {
    private ItemInternal* m_internal;

    [Obsolete("This method is internal to Arcane")]
    public ItemInternal* Internal
    {
      get { return m_internal; }
      set { m_internal = value; }
    }

    [Obsolete("This method is internal to Arcane")]
    public Cell(ItemInternal* ii)
    {
      m_internal = ii;
    }
    public Cell(Item ii)
    {
      m_internal = ii.Internal;
    }
    public bool IsNull
    {
      get { return m_internal->IsNull; }
    }
    public Int32 LocalId
    {
      get { return m_internal->m_local_id; }
    }
    public eItemKind Kind
    {
      get { return eItemKind.IK_Cell; }
    }

    public Integer NbNode
    {
      get { return m_internal->NbNode; }
    }
    public Node Node(Integer index)
    {
      return new Node(m_internal->Node(index));
    }
    public Int32 NodeLocalId(Integer index)
    {
      return m_internal->NodeLocalId(index);
    }
    public NodeList Nodes
    {
      get { return m_internal->Nodes; }
    }

    public Integer NbFace
    {
      get { return m_internal->NbFace; }
    }
    public Face Face(Integer index)
    {
      return new Face(m_internal->Face(index));
    }
    public Int32 FaceLocalId(Integer index)
    {
      return m_internal->FaceLocalId(index);
    }
    public ItemList<Face> Faces
    {
      get { return m_internal->Faces; }
    }

    public Int64 UniqueId
    {
      get { return m_internal->UniqueId(); }
    }
  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct Face : IItem
  {
    private ItemInternal* m_internal;

    [Obsolete("This method is internal to Arcane")]
    public ItemInternal* Internal
    {
      get { return m_internal; }
      set { m_internal = value; }
    }
    [Obsolete("This method is internal to Arcane")]
    public Face(ItemInternal* ii)
    {
      m_internal = ii;
    }
    public Face(Item ii)
    {
      m_internal = ii.Internal;
    }
    public bool IsNull
    {
      get { return m_internal->IsNull; }
    }
    public Int32 LocalId
    {
      get { return m_internal->m_local_id; }
    }
    public eItemKind Kind
    {
      get { return eItemKind.IK_Face; }
    }

    public Integer NbNode
    {
      get { return m_internal->NbNode; }
    }
    public Node Node(Integer index)
    {
      return new Node(m_internal->Node(index));
    }
    public Int32 NodeLocalId(Integer index)
    {
      return m_internal->NodeLocalId(index);
    }
    public NodeList Nodes
    {
      get { return m_internal->Nodes; }
    }

    public Integer NbCell
    {
      get { return m_internal->NbCell; }
    }
    public Cell Cell(Integer index)
    {
      return new Cell(m_internal->Cell(index));
    }
    public Int32 CellLocalId(Integer index)
    {
      return m_internal->CellLocalId(index);
    }
    public ItemList<Cell> Cells
    {
      get { return m_internal->Cells; }
    }

    public Cell BackCell
    {
      get { return new Cell(m_internal->BackCell()); }
    }

    public Cell FrontCell
    {
      get { return new Cell(m_internal->FrontCell()); }
    }

    public Int64 UniqueId
    {
      get { return m_internal->UniqueId(); }
    }
  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct Edge : IItem
  {
    private ItemInternal* m_internal;

    [Obsolete("This method is internal to Arcane")]
    public ItemInternal* Internal
    {
      get { return m_internal; }
      set { m_internal = value; }
    }
    [Obsolete("This method is internal to Arcane")]
    public Edge(ItemInternal* ii)
    {
      m_internal = ii;
    }

    public bool IsNull
    {
      get { return m_internal->IsNull; }
    }
    public Int32 LocalId
    {
      get { return m_internal->m_local_id; }
    }
    public eItemKind Kind
    {
      get { return eItemKind.IK_Edge; }
    }

    public Int64 UniqueId
    {
      get { return m_internal->UniqueId(); }
    }
  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct Node : IItem
  {
    private ItemInternal* m_internal;

    [Obsolete("This method is internal to Arcane")]
    public Node(ItemInternal* ii)
    {
      m_internal = ii;
    }
    public Node(Item ii)
    {
      m_internal = ii.Internal;
    }
    [Obsolete("This method is internal to Arcane")]
    public ItemInternal* Internal
    {
      get { return m_internal; }
      set { m_internal = value; }
    }
    public bool IsNull
    {
      get { return m_internal->IsNull; }
    }
    public Int32 LocalId
    {
      get { return m_internal->m_local_id; }
    }
    public Int64 UniqueId
    {
      get { return m_internal->UniqueId(); }
    }
    public eItemKind Kind
    {
      get { return eItemKind.IK_Node; }
    }

    public Cell Cell(Integer index)
    {
      return new Cell(m_internal->Cell(index));
    }
    public Int32 CellLocalId(Integer index)
    {
      return m_internal->CellLocalId(index);
    }
    public Integer NbCell
    {
      get { return m_internal->NbCell; }
    }
    public ItemList<Cell> Cells
    {
      get { return m_internal->Cells; }
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

    [Obsolete("This method is internal to Arcane")]
    public IndexedItem(ItemInternal* ii,Integer index)
    {
      m_item = new _ItemKind();
      m_item.Internal = ii;
      m_index = index;
    }
    public IndexedItem(_ItemKind item,Integer index)
    {
      m_item = item;
      m_index = index;
    }

    internal void _Set(ItemInternal* ii,Integer index)
    {
      m_item.Internal = ii;
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

    [Obsolete("This method is internal to Arcane")]
    public IndexedNode(ItemInternal* ii,Integer index)
    {
      m_item = new Node();
      m_item.Internal = ii;
      m_index = index;
    }

    public IndexedNode(Node item,Integer index)
    {
      m_item = item;
      m_index = index;
    }
  }
}
