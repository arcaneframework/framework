using System.Runtime.InteropServices;
using System;
using Arcane;

namespace Arcane.Materials
{
  [StructLayout(LayoutKind.Sequential)]
  public struct MatVarIndex
  {
    internal Int32 m_array_index;
    internal Int32 m_value_index;
    public Int32 ArrayIndex { get { return m_array_index; } }
    public Int32 ValueIndex { get { return m_value_index; } }
  }
}

namespace Arcane
{
  [StructLayout(LayoutKind.Sequential)]
  public unsafe struct MatVarIndexConstArrayView
  {
    internal Int32 m_size;
    internal Arcane.Materials.MatVarIndex* m_ptr;
  }
}
