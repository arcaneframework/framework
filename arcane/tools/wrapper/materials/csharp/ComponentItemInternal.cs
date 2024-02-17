using System.Runtime.InteropServices;
using System;

namespace Arcane.Materials
{
  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ComponentItemSharedInfo
  {
    internal ItemSharedInfo* m_item_shared_info;
    internal Int16 m_level;
    internal MeshEnvironmentListView m_components;
    internal ComponentItemSharedInfo* m_super_component_item_shared_info;
    internal ComponentItemSharedInfo* m_sub_component_item_shared_info;
    internal ComponentItemInternalConstArrayView m_component_item_internal_view;
  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ComponentItemInternal
  {
    internal MatVarIndex m_var_index;
    internal Int16 m_component_id;
    internal Int16 m_nb_sub_component_item;
    internal Int32 m_global_item_local_id;
    internal Int32 m_component_item_internal_local_id;
    internal Int32 m_super_component_item_local_id;
    internal Int32 m_first_sub_component_item_local_id;
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
