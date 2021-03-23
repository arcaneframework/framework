using System;
using System.Runtime.InteropServices;

#if ARCANE_64BIT
using Integer = System.Int64;
#else
using Integer = System.Int32;
#endif
using Real = System.Double;

namespace Arcane
{
  [StructLayout(LayoutKind.Sequential)]
  public unsafe class AnyArrayView
  {
    public Integer m_size;
    public void* m_ptr;
    public AnyArrayView(AnyArrayView v)
    {
      m_size = v.m_size;
      m_ptr = v.m_ptr;
    }
  }
 
}
