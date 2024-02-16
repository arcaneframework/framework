using System.Runtime.InteropServices;
using System;
using Arcane;

namespace Arcane.Materials
{
  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ConstituentItemLocalIdListView
  {
    internal ComponentItemSharedInfo* m_component_shared_info;
    internal Int32ConstArrayView m_ids;
    internal ComponentItemInternalPtrConstArrayView m_items_internal;
  }
 
  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ComponentItemVectorView
  {
    internal MatVarIndexConstArrayView m_matvar_indexes_view;
    internal ConstituentItemLocalIdListView m_constituent_list_view;
    internal Int32ConstArrayView m_items_local_id_view;
    internal IntPtr m_component; //IMeshComponent*

    public Int32 NbItem() { return m_matvar_indexes_view.m_size; }
  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct MatItemVectorView
  {
    internal ComponentItemVectorView m_component_vector_view;
    public Int32 NbItem() { return m_component_vector_view.NbItem(); }
    public static implicit operator ComponentItemVectorView(MatItemVectorView c) => c.m_component_vector_view;
  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct EnvItemVectorView
  {
    internal ComponentItemVectorView m_component_vector_view;
    public Int32 NbItem(){ return m_component_vector_view.NbItem(); }
    public static implicit operator ComponentItemVectorView(EnvItemVectorView c) => c.m_component_vector_view;
  }
}
