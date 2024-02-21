using System.Runtime.InteropServices;
using System;
using Arcane;

namespace Arcane.Materials
{
  public unsafe interface IComponentItem
  {
    ConstituentItemBase ItemBase { get; set; }
  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct ComponentItem : IComponentItem
  {
    internal ConstituentItemBase m_item_base;

    public ConstituentItemBase ItemBase
    {
      get { return m_item_base; }
      set { m_item_base = value; }
    }

    public Cell GlobalCell { get { return m_item_base.GlobalCell; } }
    public MatVarIndex MatVarIndex { get { return m_item_base.MatVarIndex; } }
    internal int _matvarArrayIndex { get { return MatVarIndex.ArrayIndex; } }
    internal int _matvarValueIndex { get { return MatVarIndex.ValueIndex; } }

    public ComponentItem(ConstituentItemBase ci)
    {
      m_item_base = ci;
    }
  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct MatItem : IComponentItem
  {
    internal ConstituentItemBase m_item_base;

    public ConstituentItemBase ItemBase
    {
      get { return m_item_base; }
      set { m_item_base = value; }
    }
    public Cell GlobalCell { get { return m_item_base.GlobalCell; } }
    public MatVarIndex MatVarIndex { get { return m_item_base.MatVarIndex; } }
    internal int _matvarArrayIndex { get { return MatVarIndex.ArrayIndex; } }
    internal int _matvarValueIndex { get { return MatVarIndex.ValueIndex; } }
    public MatItem(ConstituentItemBase ci)
    {
      m_item_base = ci;
    }
    public static implicit operator ComponentItem(MatItem c) => new ComponentItem(c.m_item_base);
  }

  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct EnvItem : IComponentItem
  {
    internal ConstituentItemBase m_item_base;

    public ConstituentItemBase ItemBase
    {
      get { return m_item_base; }
      set { m_item_base = value; }
    }
    public Cell GlobalCell { get { return m_item_base.GlobalCell; } }
    public MatVarIndex MatVarIndex { get { return m_item_base.MatVarIndex; } }
    internal int _matvarArrayIndex { get { return MatVarIndex.ArrayIndex; } }
    internal int _matvarValueIndex { get { return MatVarIndex.ValueIndex; } }

    public EnvItem(ConstituentItemBase ci)
    {
      m_item_base = ci;
    }

    public static implicit operator ComponentItem(EnvItem c) => new ComponentItem(c.m_item_base);
  }
}
