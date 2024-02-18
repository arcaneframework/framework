using System.Runtime.InteropServices;
using System;

namespace Arcane.Materials
{
  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ComponentItemSharedInfo
  {
    // Structure de la classe C++ ComponentItemSharedInfoStorageView
    Int32 m_storage_size;
    Int32* m_first_sub_constituent_item_id_data;
    Int16* m_component_id_data;
    Int16* m_nb_sub_constituent_item_data;
    Int32* m_global_item_local_id_data;
    Int32* m_super_component_item_local_id_data;
    MatVarIndex* m_var_index_data;

    // Structure de la classe C++ ComponentItemSharedInfo
    internal ItemSharedInfo* m_item_shared_info;
    internal Int16 m_level;
    internal MeshEnvironmentListView m_components;
    internal ComponentItemSharedInfo* m_super_component_item_shared_info;
    internal ComponentItemSharedInfo* m_sub_component_item_shared_info;
    internal ComponentItemInternalConstArrayView m_component_item_internal_view;

    internal Cell GlobalCell(Int32 constituent_local_id)
    {
      Int32 global_local_id = m_global_item_local_id_data[constituent_local_id];
      return new Cell(m_item_shared_info->m_items_internal[global_local_id]);
    }
    internal MatVarIndex VarIndex(Int32 constituent_local_id)
    {
      return m_var_index_data[constituent_local_id];
    }
  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ComponentItemInternal
  {
    internal Int32 m_component_item_internal_local_id;
    internal ComponentItemSharedInfo* m_shared_info;
  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ComponentItemInternalConstArrayView
  {
    internal Int32 m_size;
    internal Arcane.Materials.ComponentItemInternal* m_ptr;
  }
}

namespace Arcane
{
  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ComponentItemInternalPtrConstArrayView
  {
    internal Int32 m_size;
    internal Arcane.Materials.ComponentItemInternal** m_ptr;
  }
}
