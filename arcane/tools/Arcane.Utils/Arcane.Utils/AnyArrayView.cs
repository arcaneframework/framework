using System;
using System.Runtime.InteropServices;

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
