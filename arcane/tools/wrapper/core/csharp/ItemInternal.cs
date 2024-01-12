//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#define USE_NEW_CONNECTIVITY

using System;
using System.Runtime.InteropServices;

#if ARCANE_64BIT
using Integer = System.Int64;
#else
using Integer = System.Int32;
#endif

namespace Arcane
{
  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ItemInternalList
  {
    internal Integer m_size;
    internal ItemInternal** m_ptr;
    internal ItemInternal* this[Int32 id]
    {
      get { return m_ptr[id]; }
    }
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct MeshItemInternalList
  {
    internal ItemInternalList nodes;
    internal ItemInternalList edges;
    internal ItemInternalList faces;
    internal ItemInternalList cells;
    internal ItemInternalList dualNodes;
    internal ItemInternalList links;
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  //TRES IMPORTANT: verifier que cette structure est identique à ItemInternalConnectivityList.h du C++
  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ItemInternalConnectivityList
  {
    public enum Idx
    {
      NODE_IDX = 0,
      EDGE_IDX = 1,
      FACE_IDX = 2,
      CELL_IDX = 3,
      HPARENT_IDX = 4,
      HCHILD_IDX = 5,
      MAX_ITEM_KIND = 6
    }

    public Int32 NodeLocalId(Int32 local_id,Int32 index)
    {
      return m_container_node.m_list[ m_container_node.m_indexes[local_id] + index ];
    }
    public ItemInternal* Node(Int32 local_id,Int32 index)
    {
      return m_items->nodes[ NodeLocalId(local_id,index) ];
    }
    public NodeList Nodes(Int32 local_id)
    {
      int nb_node = m_container_node.m_nb_item[local_id];
      return new NodeList(m_items->nodes.m_ptr,m_container_node.m_list._InternalData()+m_container_node.m_indexes[local_id],nb_node);
    }
    public Int32 NbNode(Int32 local_id)
    {
      return m_container_node.m_nb_item[local_id];
    }

    public Int32 FaceLocalId(Int32 local_id,Int32 index)
    {
      return m_container_face.m_list[ m_container_face.m_indexes[local_id] + index ];
    }
    public ItemInternal* Face(Int32 local_id,Int32 index)
    {
      return m_items->faces[ FaceLocalId(local_id,index) ];
    }
    public ItemList<Face> Faces(Int32 local_id)
    {
      int nb_face = m_container_face.m_nb_item[local_id];
      return new ItemList<Face>(m_items->faces.m_ptr,m_container_face.m_list._InternalData()+m_container_face.m_indexes[local_id],nb_face);
    }
    public Int32 NbFace(Int32 local_id)
    {
      return m_container_face.m_nb_item[local_id];
    }

    public Int32 CellLocalId(Int32 local_id,Int32 index)
    {
      return m_container_cell.m_list[ m_container_cell.m_indexes[local_id] + index ];
    }
    public ItemInternal* Cell(Int32 local_id,Int32 index)
    {
      return m_items->cells[ CellLocalId(local_id,index) ];
    }
    public ItemList<Cell> Cells(Int32 local_id)
    {
      int nb_cell = m_container_node.m_nb_item[local_id];
      return new ItemList<Cell>(m_items->cells.m_ptr,m_container_cell.m_list._InternalData()+m_container_cell.m_indexes[local_id],nb_cell);
    }
    public Int32 NbCell(Int32 local_id)
    {
      return m_container_cell.m_nb_item[local_id];
    }

    // NOTE: Une fois qu'on sera passé à la version C# 10, on pourra utiliser
    // des tableaux de taille fixe

    [StructLayout(LayoutKind.Sequential)]
    struct Container
    {
      public Int32ArrayView m_indexes;
      public Int32ArrayView m_nb_item;
      public Int32ArrayView m_list;
      public Int32ArrayView m_offset;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct KindInfo
    {
      public Int32 m_max_nb_item;
      public Int32 m_nb_item_null_data0;
      public Int32 m_nb_item_null_data1;
    }

    Container m_container_node;
    Container m_container_edge;
    Container m_container_face;
    Container m_container_cell;
    Container m_container_hparent;
    Container m_container_hchild;

    KindInfo m_kind_node;
    KindInfo m_kind_edge;
    KindInfo m_kind_face;
    KindInfo m_kind_cell;
    KindInfo m_kind_hparent;
    KindInfo m_kind_hchild;

    MeshItemInternalList* m_items;
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  //TRES IMPORTANT: verifier que cette structure est identique a ItemSharedInfo.h du C++
  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ItemSharedInfo
  {
    internal MeshItemInternalList* m_items;
    internal ItemInternalConnectivityList* m_connectivity;
    internal IntPtr m_family; // IItemFamily*;
    internal IntPtr m_item_type_mng; // ItemTypeMng*
    internal Int64ArrayView m_unique_ids;
    internal Int32ArrayView m_parent_item_ids;
    internal Int32ArrayView m_owners;
    internal Int32ArrayView m_flags;
    internal Int16ArrayView m_type_ids;
    internal eItemKind m_item_kind;
    internal Int32 m_nb_parent;
    internal ItemInternalList m_items_internal;

    //! Pour l'entité nulle
    private static ItemSharedInfo* null_item_shared_info = null;
    internal static ItemSharedInfo* Zero
    {
      get
      {
        if (null_item_shared_info==null){
          int size = Marshal.SizeOf(typeof(ItemSharedInfo));
          null_item_shared_info = (ItemSharedInfo*)Marshal.AllocHGlobal(size);
          Console.WriteLine("ALLOC NULL ITEM_SHARED_INFO: TODO fill structure");
        }
        return null_item_shared_info;
      }
    }
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  //TRES IMPORTANT: verifier que cette structure est identique a ItemSharedInfo.h du C++
  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ItemInternal
  {
    const Int32 NULL_ITEM_LOCAL_ID = -1;
    const Int32 II_Boundary = 1 << 1; //!< L'entité est sur la frontière
    const Int32 II_HasFrontCell = 1 << 2; //!< L'entité a une maille devant
    const Int32 II_HasBackCell  = 1 << 3; //!< L'entité a une maille derrière
    const Int32 II_FrontCellIsFirst = 1 << 4; //!< La première maille de l'entité est la maille devant
    const Int32 II_BackCellIsFirst  = 1 << 5; //!< La première maille de l'entité est la maille derrière
    const Int32 II_Own = 1 << 6; //!< L'entité est une entité propre au sous-domaine
    const Int32 II_Added = 1 << 7; //!< L'entité vient d'être ajoutée
    const Int32 II_Suppressed = 1 << 8; //!< L'entité vient d'être supprimée
    const Int32 II_Shared = 1 << 9; //!< L'entité est partagée par un autre sous-domaine
    const Int32 II_SubDomainBoundary = 1 << 10; //!< L'entité est à la frontière de deux sous-domaines
    //II_JustRemoved = 1 << 11; //!< L'entité vient d'être supprimé
    const Int32 II_JustAdded = 1 << 12; //!< L'entité vient d'être ajoutée
    const Int32 II_NeedRemove = 1 << 13; //!< L'entité doit être supprimé
    const Int32 II_SlaveFace = 1 << 14; //!< L'entité est une face esclave d'une interface
    const Int32 II_MasterFace = 1 << 15; //!< L'entité est une face maître d'une interface
    const Int32 II_Detached = 1 << 16; //!< L'entité est détachée du maillage
    const Int32 II_HasTrace = 1 << 17; //!< L'entité est marquée pour trace (pour débug)
    const Int32 II_Coarsen = 1 << 18; //!<  L'entité est marquée pour déraffinement
    const Int32 II_DoNothing = 1 << 19; //!<  L'entité est bloquée
    const Int32 II_Refine = 1 << 20; //!<  L'entité est marquée pour raffinement
 	  const Int32 II_JustRefined = 1 << 21; //!<  L'entité vient d'être raffinée
    const Int32 II_JustCoarsened = 1 << 22; //!<  L'entité vient d'être déraffiné
    const Int32 II_Inactive = 1 << 23; //!<  L'entité est inactive //COARSEN_INACTIVE,
    const Int32 II_CoarsenInactive = 1 << 24; //!<  L'entité est inactive et a des enfants tagués pour déraffinement
    const Int32 II_UserMark1 = 1 << 25; //!< Marque utilisateur old_value 1<<24
    const Int32 II_UserMark2 = 1 << 26; //!< Marque utilisateur  old_value 1<<25
    const Int32 II_InterfaceFlags = II_Boundary + II_HasFrontCell + II_HasBackCell +
    II_FrontCellIsFirst + II_BackCellIsFirst;


    internal static ItemInternal* null_item = null;
    public Int32 m_local_id;
    public Int32 m_padding;
    ItemSharedInfo* m_shared_info;
    
    internal static ItemInternal* Zero
    {
      get
      {
        if (null_item==null){
          int size = Marshal.SizeOf(typeof(ItemInternal));
          null_item = (ItemInternal*)Marshal.AllocHGlobal(size);
          null_item->m_local_id = NULL_ITEM_LOCAL_ID;
          null_item->m_shared_info = ItemSharedInfo.Zero;
        }
        return null_item;
      }
    }

    public bool IsNull { get { return m_local_id==NULL_ITEM_LOCAL_ID; } }
    
    //! Flags de l'entité
    public Integer Flags { get { return m_shared_info->m_flags.At(m_local_id); } }
    ItemInternalConnectivityList* _connectivity() { return m_shared_info->m_connectivity; }

    public ItemInternal* Node(Integer index)
    {
      return _connectivity()->Node(m_local_id,index);
    }
    public Int32 NodeLocalId(Integer index)
    {
      return _connectivity()->NodeLocalId(m_local_id,index);
    }
    public NodeList Nodes
    {
      get { return _connectivity()->Nodes(m_local_id); }
    }
    public Integer NbNode
    {
      get { return _connectivity()->NbNode(m_local_id); }
    }

    public ItemInternal* Cell(Integer index)
    {
      return _connectivity()->Cell(m_local_id,index);
    }
    public Int32 CellLocalId(Integer index)
    {
      return _connectivity()->CellLocalId(m_local_id,index);
    }
    public ItemList<Cell> Cells
    {
      get { return _connectivity()->Cells(m_local_id); }
    }
    public Int32 NbCell
    {
      get { return _connectivity()->NbCell(m_local_id); }
    }

    public ItemInternal* Face(Integer index)
    {
      return _connectivity()->Face(m_local_id,index);
    }
    public Int32 FaceLocalId(Integer index)
    {
      return _connectivity()->FaceLocalId(m_local_id,index);
    }
    public ItemList<Face> Faces
    {
      get { return _connectivity()->Faces(m_local_id); }
    }
    public Int32 NbFace
    {
      get { return _connectivity()->NbFace(m_local_id); }
    }

    public Int64 UniqueId()
    {
      return m_shared_info->m_unique_ids.At(m_local_id);
    }
    public eItemKind Kind
    {
      get { return m_shared_info->m_item_kind; }
    }

    public ItemInternal* BackCell()
    {
      if ((Flags & II_HasBackCell)!=0){
        //Console.WriteLine("HAS BACK CELL");
        return Cell(((Flags & II_BackCellIsFirst)!=0) ? 0 : 1);
      }
      //Console.WriteLine("NO BACK CELL");
      return ItemInternal.Zero;
    }
    //! Maille devant l'entité (0 si aucune)
    public ItemInternal* FrontCell()
    {
      if ((Flags & II_HasFrontCell)!=0){
        //Console.WriteLine("HAS FRONT CELL");
        return Cell(((Flags & II_FrontCellIsFirst)!=0) ? 0 : 1);
      }
      //Console.WriteLine("NO FRONT CELL");
      return ItemInternal.Zero;
    }
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  [StructLayout(LayoutKind.Sequential)]
  public unsafe class ItemInternalArrayView
  {
    private Integer m_size;
    internal ItemInternal** m_ptr;

    public Item this[Integer index]
    {
      get{
        return new Item(new ItemBase(m_ptr[index]));
      }
    }
    [Obsolete("This method is internal to Arcane. Use ItemBase() instead.")]
    public ItemInternal* ItemInternal(Integer index)
    {
      return m_ptr[index];
    }
    public ItemBase ItemBase(Integer index)
    {
      return new ItemBase(m_ptr[index]);
    }
    public Integer Size { get { return m_size; } }
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  public unsafe class ItemInternalListRef
  {
    public Item[] m_items;
    public ItemInternalListRef(ItemInternalArrayView ilist)
    {
      Integer s = ilist.Size;
      m_items = new Item[s];
      for( int i=0; i<s; ++i )
        m_items[i] = new Item(ilist.ItemBase(i));
    }
    public Integer Size
    {
      get{ return m_items.Length; }
    }

    public Item this[Integer index]
    {
      get{
        return m_items[index];
      }
    }
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ItemBase
  {
    internal ItemBase(ItemInternal* v)
    {
      m_item_internal = v;
    }

    internal ItemInternal* Internal { get { return m_item_internal; } }

    internal bool IsNull { get { return m_item_internal->IsNull; } }
    internal Int32 LocalId { get { return m_item_internal->m_local_id; } }
    internal Int64 UniqueId { get { return m_item_internal->UniqueId(); } }
    internal eItemKind Kind { get { return m_item_internal->Kind; } }

    internal Int32 NbNode { get { return m_item_internal->NbNode; } }
    internal ItemBase Node(Int32 index) { return new ItemBase(m_item_internal->Node(index)); }
    internal Int32 NodeLocalId(Int32 index) { return m_item_internal->NodeLocalId(index); }
    internal NodeList Nodes{ get { return m_item_internal->Nodes; } }

    internal Int32 NbFace { get { return m_item_internal->NbFace; } }
    internal ItemBase Face(Int32 index) { return new ItemBase(m_item_internal->Face(index)); }
    internal Int32 FaceLocalId(Int32 index) { return m_item_internal->FaceLocalId(index); }
    internal ItemList<Face> Faces { get { return m_item_internal->Faces; } }

    internal Int32 NbCell { get { return m_item_internal->NbCell; } }
    internal ItemBase Cell(Int32 index) { return new ItemBase(m_item_internal->Cell(index)); }
    internal Int32 CellLocalId(Int32 index) { return m_item_internal->CellLocalId(index); }
    internal ItemList<Cell> Cells { get { return m_item_internal->Cells; } }

    internal ItemBase BackCell { get { return new ItemBase(m_item_internal->BackCell()); } }
    internal ItemBase FrontCell { get { return new ItemBase(m_item_internal->FrontCell()); } }

    ItemInternal* m_item_internal;
  }
}
