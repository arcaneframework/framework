//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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
      return m_list_node[ m_indexes_node[local_id] + index ];
    }
    public ItemInternal* Node(Int32 local_id,Int32 index)
    {
      return m_items->nodes[ NodeLocalId(local_id,index) ];
    }
    public NodeList Nodes(Int32 local_id)
    {
      int nb_node = m_nb_item_node[local_id];
      return new NodeList(m_items->nodes.m_ptr,m_list_node._UnguardedBasePointer()+m_indexes_node[local_id],nb_node);
    }
    public Int32 NbNode(Int32 local_id)
    {
      return m_nb_item_node[local_id];
    }

    public Int32 FaceLocalId(Int32 local_id,Int32 index)
    {
      return m_list_face[ m_indexes_face[local_id] + index ];
    }
    public ItemInternal* Face(Int32 local_id,Int32 index)
    {
      return m_items->faces[ FaceLocalId(local_id,index) ];
    }
    public ItemList<Face> Faces(Int32 local_id)
    {
      int nb_face = m_nb_item_face[local_id];
      return new ItemList<Face>(m_items->faces.m_ptr,m_list_face._UnguardedBasePointer()+m_indexes_face[local_id],nb_face);
    }
    public Int32 NbFace(Int32 local_id)
    {
      return m_nb_item_face[local_id];
    }

    public Int32 CellLocalId(Int32 local_id,Int32 index)
    {
      return m_list_cell[ m_indexes_cell[local_id] + index ];
    }
    public ItemInternal* Cell(Int32 local_id,Int32 index)
    {
      return m_items->cells[ CellLocalId(local_id,index) ];
    }
    public ItemList<Cell> Cells(Int32 local_id)
    {
      int nb_cell = m_nb_item_cell[local_id];
      return new ItemList<Cell>(m_items->cells.m_ptr,m_list_cell._UnguardedBasePointer()+m_indexes_cell[local_id],nb_cell);
    }
    public Int32 NbCell(Int32 local_id)
    {
      return m_nb_item_cell[local_id];
    }

    Int32ArrayView m_indexes_node;
    Int32ArrayView m_indexes_edge;
    Int32ArrayView m_indexes_face;
    Int32ArrayView m_indexes_cell;
    Int32ArrayView m_indexes_hparent;
    Int32ArrayView m_indexes_hchild;

    Int32ArrayView m_nb_item_node;
    Int32ArrayView m_nb_item_edge;
    Int32ArrayView m_nb_item_face;
    Int32ArrayView m_nb_item_cell;
    Int32ArrayView m_nb_item_hparent;
    Int32ArrayView m_nb_item_hchild;

    Int32ConstArrayView m_list_node;
    Int32ConstArrayView m_list_edge;
    Int32ConstArrayView m_list_face;
    Int32ConstArrayView m_list_cell;
    Int32ConstArrayView m_list_hparent;
    Int32ConstArrayView m_list_hchild;

    MeshItemInternalList* m_items;

    Int64 m_nb_access_all;
    Int64 m_nb_access;

    IntPtr m_indexes_array0;
    IntPtr m_indexes_array1;
    IntPtr m_indexes_array2;
    IntPtr m_indexes_array3;
    IntPtr m_indexes_array4;
    IntPtr m_indexes_array5;

    IntPtr m_nb_item_array0;
    IntPtr m_nb_item_array1;
    IntPtr m_nb_item_array2;
    IntPtr m_nb_item_array3;
    IntPtr m_nb_item_array4;
    IntPtr m_nb_item_array5;

    Int32 m_nb_item_null_data_0_0;
    Int32 m_nb_item_null_data_0_1;
    Int32 m_nb_item_null_data_1_0;
    Int32 m_nb_item_null_data_1_1;
    Int32 m_nb_item_null_data_2_0;
    Int32 m_nb_item_null_data_2_1;
    Int32 m_nb_item_null_data_3_0;
    Int32 m_nb_item_null_data_3_1;
    Int32 m_nb_item_null_data_4_0;
    Int32 m_nb_item_null_data_4_1;
    Int32 m_nb_item_null_data_5_0;
    Int32 m_nb_item_null_data_5_1;
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  //TRES IMPORTANT: verifier que cette structure est identique a ItemSharedInfo.h du C++
  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ItemSharedInfo
  {
    const Integer OWNER_INDEX = 0;
    const Integer FLAGS_INDEX = 1;
    const Integer FIRST_NODE_INDEX = 2;
    internal const Integer COMMON_BASE_MEMORY = 2;

    internal Int32* m_infos;
    internal Integer m_first_node;
    Integer m_nb_node;
    internal Integer m_first_edge;
    Integer m_nb_edge;
    internal Integer m_first_face;
    Integer m_nb_face;
    internal Integer m_first_cell;
    Integer m_nb_cell;
    internal Integer m_first_parent;
    internal Integer m_nb_parent;
    //! AMR
    internal Int32 m_first_hParent;  
    internal Int32 m_first_hChild;
    internal Int32 m_nb_hParent;
    internal Int32 m_nb_hChildren;

    internal MeshItemInternalList* m_items;
    internal ItemInternalConnectivityList* m_connectivity;
    internal IntPtr m_family; //IItemFamily* m_family;
    internal Int64ArrayView* m_unique_ids;
    internal IntPtr m_item_type; //ItemTypeInfo* m_item_type;
    internal eItemKind m_item_kind; //eItemKind m_item_kind;
    internal Integer m_needed_memory;
    internal Integer m_minimum_needed_memory;
    internal Integer m_edge_allocated;
    internal Integer m_face_allocated;
    internal Integer m_cell_allocated;
    //! AMR
    internal Int32 m_hParent_allocated;
    internal Int32 m_hChild_allocated;

    internal Integer m_type_id;
    internal Integer m_index;
    internal Integer m_nb_reference;
    internal Int32 m_has_legacy_connectivity;

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

    bool hasLegacyConnectivity()
    {
      return m_has_legacy_connectivity == 0;
    }

    public Integer NbNode
    {
      get { return m_nb_node; }
    }
    public Integer NbFace
    {
      get { return m_nb_node; }
    }
    public Integer NbCell
    {
      get { return m_nb_cell; }
    }
    public ItemInternal* Node(Integer index,Integer data_index)
    {
      return m_items->nodes.m_ptr[m_infos[data_index+index+m_first_node]];
    }
    public Int32 NodeLocalId(Integer index,Integer data_index)
    {
      return m_infos[data_index+index+m_first_node];
    }
    public NodeList Nodes(Integer data_index)
    {
      return new NodeList(m_items->nodes.m_ptr,m_infos+data_index+m_first_node,m_nb_node);
    }

    public ItemInternal* Face(Integer index,Integer data_index)
    {
      return m_items->faces.m_ptr[m_infos[data_index+index+m_first_face]];
    }
    public Int32 FaceLocalId(Integer index,Integer data_index)
    {
      return m_infos[data_index+index+m_first_face];
    }
    public ItemList<Face> Faces(Integer data_index)
    {
      return new ItemList<Face>(m_items->faces.m_ptr,m_infos+data_index+m_first_face,m_nb_face);
    }

    public ItemInternal* Cell(Integer index,Integer data_index)
    {
      return m_items->cells.m_ptr[m_infos[data_index+index+m_first_cell]];
    }
    public Int32 CellLocalId(Integer index,Integer data_index)
    {
      return m_infos[data_index+index+m_first_cell];
    }
    public ItemList<Cell> Cells(Integer data_index)
    {
      return new ItemList<Cell>(m_items->cells.m_ptr,m_infos+data_index+m_first_cell,m_nb_cell);
    }

    internal Integer Flags(Integer data_index)
    {
      return m_infos[data_index+FLAGS_INDEX];
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
    public Int32 m_data_index;
    ItemSharedInfo* m_shared_info;
    // Le champ 'm_connectivity' n'est actif que si Arcane a été compilé
    // sans la macro 'ARCANE_USE_SHAREDINFO_CONNECTIVITY'
    // public ItemInternalConnectivityList* m_connectivity;
    
    internal static ItemInternal* Zero
    {
      get
      {
        if (null_item==null){
          int size = Marshal.SizeOf(typeof(ItemInternal));
          null_item = (ItemInternal*)Marshal.AllocHGlobal(size);
          null_item->m_local_id = NULL_ITEM_LOCAL_ID;
          null_item->m_data_index = 0;
          null_item->m_shared_info = ItemSharedInfo.Zero;
        }
        return null_item;
      }
    }

    public bool IsNull { get { return m_local_id==NULL_ITEM_LOCAL_ID; } }
    
    //! Flags de l'entité
    public Integer Flags { get { return m_shared_info->Flags(m_data_index); } }
    ItemInternalConnectivityList* _connectivity() { return m_shared_info->m_connectivity; }

#if USE_NEW_CONNECTIVITY
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

#else
    public ItemInternal* Node(Integer index)
    {
      return m_shared_info->Node(index,m_data_index);
    }
    public Int32 NodeLocalId(Integer index)
    {
      return m_shared_info->NodeLocalId(index,m_data_index);
    }
    public NodeList Nodes
    {
      get { return m_shared_info->Nodes(m_data_index); }
    }
    public Int32 NbNode
    {
      get { return m_shared_info->NbNode(); }
    }

    public ItemInternal* Cell(Integer index)
    {
      return m_shared_info->Cell(index,m_data_index);
    }
    public Int32 CellLocalId(Integer index)
    {
      return m_shared_info->CellLocalId(index,m_data_index);
    }
    public ItemList<Cell> Cells
    {
      get { return m_shared_info->Cells(m_data_index); }
    }
    public Int32 NbCell
    {
      get { return m_shared_info->NbCell(); }
    }

    public ItemInternal* Face(Integer index)
    {
      return m_shared_info->Face(index,m_data_index);
    }
    public Int32 FaceLocalId(Integer index)
    {
      return m_shared_info->FaceLocalId(index,m_data_index);
    }
    public ItemList<Face> Faces
    {
      get { return m_shared_info->Faces(m_data_index); }
    }
    public Int32 NbFace
    {
      get { return m_shared_info->NbFace(); }
    }
#endif

    public Int64 UniqueId()
    {
      return m_shared_info->m_unique_ids->At(m_local_id);
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
        return new Item(m_ptr[index]);
      }
    }
    public ItemInternal* ItemInternal(Integer index)
    {
      return m_ptr[index];
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
        m_items[i] = new Item(ilist.ItemInternal(i));
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
}
