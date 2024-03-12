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
  public unsafe struct ConstituentItemBase
  {
    internal Int32 m_component_item_index;
    internal ComponentItemSharedInfo* m_shared_info;
    internal ConstituentItemBase(Int32 component_item_index,ComponentItemSharedInfo* shared_info)
    {
      m_component_item_index = component_item_index;
      m_shared_info = shared_info;
    }
    static internal ConstituentItemBase Null()
    {
      return new ConstituentItemBase(-1,null);
    }
    internal Cell GlobalCell { get { return m_shared_info->GlobalCell(m_component_item_index); } }
    internal MatVarIndex MatVarIndex { get { return m_shared_info->VarIndex(m_component_item_index); } }
  }
}
