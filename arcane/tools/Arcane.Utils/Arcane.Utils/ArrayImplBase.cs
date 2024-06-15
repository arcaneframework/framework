//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("Arcane.Core")]

namespace Arcane
{
  internal unsafe struct ArrayImplBase
  {
    internal Integer capacity;
    internal Integer size;
    static internal bool m_verbose;
    
    static void _MemCopy(ArrayImplBase* dest,ArrayImplBase* src,Int64 size)
    {
      Byte* dest_b = (Byte*)dest;
      Byte* src_b = (Byte*)src;
      if (dest_b==null)
        throw new NullReferenceException("Bad destination");
      if (src_b==null)
        throw new NullReferenceException("Bad source");
      for( Int64 i=0; i<size; ++i )
        dest_b[i] = src_b[i];
    }
    static public ArrayImplBase* allocate(Int64 sizeof_true_impl,Int64 new_capacity,
                                          Int64 sizeof_true_type,ArrayImplBase* init)
    {
      Int64 elem_size = sizeof_true_impl + (new_capacity - 1) * sizeof_true_type;
      ArrayImplBase* p = (ArrayImplBase*)Marshal.AllocHGlobal((IntPtr)elem_size);
      if (p==null)
        throw new OutOfMemoryException(String.Format("Can not allocate {0} bytes",elem_size));
      GC.AddMemoryPressure(elem_size);
      //Console.WriteLine("RETURN p={0}",(IntPtr)p);
      Int64 s = (new_capacity>init->capacity) ? init->capacity : new_capacity;
      _MemCopy(p, init,sizeof_true_impl + (s - 1) * sizeof_true_type);
      //Console.WriteLine("MEMCOPY NEEDED!");
      //std::cout << " RETURN p=" << (((Int64)p)%16) << '\n';
      //if (elem_size>700000){
        //StackTrace st = new StackTrace(true);
        if (m_verbose)
          Console.WriteLine("ALLOC MEMTRACE p={0} size={1}",(IntPtr)p,new_capacity);
        //Console.WriteLine(st.ToString());
      //}
      return p;
    }
    
    static public ArrayImplBase* reallocate(Int64 sizeof_true_impl,Int64 new_capacity,
                                            Int64 sizeof_true_type,ArrayImplBase* current)
    {
      Int64 elem_size = sizeof_true_impl + (new_capacity - 1) * sizeof_true_type;
      Int64 elem_current_size = sizeof_true_impl + (current->capacity - 1) * sizeof_true_type;
      GC.RemoveMemoryPressure(elem_current_size);
      ArrayImplBase* p = (ArrayImplBase*)Marshal.ReAllocHGlobal((IntPtr)current,(IntPtr)elem_size);
      if (m_verbose)
        Console.WriteLine("REALLOCATE p={0} new_p={1} s={2}",(IntPtr)current,(IntPtr)p,elem_size);
      GC.AddMemoryPressure(elem_size);
      //std::cout << " RETURN p=" << p << '\n';
      return p;
    }
    
    static public void deallocate(Int64 sizeof_true_impl,Int64 sizeof_true_type,ArrayImplBase* current)
    {
      Int64 elem_current_size = sizeof_true_impl + (current->capacity - 1) * sizeof_true_type;
      GC.RemoveMemoryPressure(elem_current_size);
      if (m_verbose)
        Console.WriteLine("DEALLOCATE p={0} s={1}",(IntPtr)current,current->capacity);
      Marshal.FreeHGlobal((IntPtr)current);
    }
  }
}
