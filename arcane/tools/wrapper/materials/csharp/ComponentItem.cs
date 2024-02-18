using System.Runtime.InteropServices;
using System;
using Arcane;

namespace Arcane.Materials
{
  public unsafe interface IComponentItem
  {
    [Obsolete("This method is internal to Arcane")]
    ComponentItemInternal* Internal { get; set; }
  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ComponentItem : IComponentItem
  {
    internal ComponentItemInternal* m_internal;

    [Obsolete("This method is internal to Arcane")]
    public ComponentItemInternal* Internal
    {
      get { return m_internal; }
      set { m_internal = value; }
    }

    public Cell GlobalCell { get { return m_internal->m_shared_info->GlobalCell(m_internal->m_component_item_internal_local_id); } }
    public MatVarIndex MatVarIndex { get { return m_internal->m_shared_info->VarIndex(m_internal->m_component_item_internal_local_id); } }
    internal int _matvarArrayIndex { get { return MatVarIndex.ArrayIndex; } }
    internal int _matvarValueIndex { get { return MatVarIndex.ValueIndex; } }

    [Obsolete("This method is internal to Arcane")]
    public ComponentItem(ComponentItemInternal* ci)
    {
      m_internal = ci;
    }
  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct MatItem : IComponentItem
  {
    internal ComponentItemInternal* m_internal;

    [Obsolete("This method is internal to Arcane")]
    public ComponentItemInternal* Internal
    {
      get { return m_internal; }
      set { m_internal = value; }
    }
    public Cell GlobalCell { get { return m_internal->m_shared_info->GlobalCell(m_internal->m_component_item_internal_local_id); } }
    public MatVarIndex MatVarIndex { get { return m_internal->m_shared_info->VarIndex(m_internal->m_component_item_internal_local_id); } }
    internal int _matvarArrayIndex { get { return MatVarIndex.ArrayIndex; } }
    internal int _matvarValueIndex { get { return MatVarIndex.ValueIndex; } }
    [Obsolete("This method is internal to Arcane")]
    public MatItem(ComponentItemInternal* ci)
    {
      m_internal = ci;
    }
    public static implicit operator ComponentItem(MatItem c) => new ComponentItem(c.m_internal);
  }
  public unsafe struct EnvItem : IComponentItem
  {
    internal ComponentItemInternal* m_internal;
    [Obsolete("This method is internal to Arcane")]
    public ComponentItemInternal* Internal
    {
      get { return m_internal; }
      set { m_internal = value; }
    }

    public Cell GlobalCell { get { return m_internal->m_shared_info->GlobalCell(m_internal->m_component_item_internal_local_id); } }
    public MatVarIndex MatVarIndex { get { return m_internal->m_shared_info->VarIndex(m_internal->m_component_item_internal_local_id); } }
    internal int _matvarArrayIndex { get { return MatVarIndex.ArrayIndex; } }
    internal int _matvarValueIndex { get { return MatVarIndex.ValueIndex; } }

    [Obsolete("This method is internal to Arcane")]
    public EnvItem(ComponentItemInternal* ci)
    {
      m_internal = ci;
    }

    public static implicit operator ComponentItem(EnvItem c) => new ComponentItem(c.m_internal);
  }
}
