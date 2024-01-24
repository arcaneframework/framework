using System.Runtime.InteropServices;
using System;

namespace Arcane.Materials
{
  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ComponentItemSharedInfo
  {
    internal ItemSharedInfo* m_item_shared_info;
    internal Int16 m_level;
  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ComponentItemInternal
  {
    internal MatVarIndex m_var_index;
    internal Int16 m_component_id;
    internal Int16 m_nb_sub_component_item;
    internal Int32 m_global_item_local_id;
    internal ComponentItemInternal* m_super_component_item;
    internal ComponentItemInternal* m_first_sub_component_item;
    internal ComponentItemSharedInfo* m_shared_info;
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
