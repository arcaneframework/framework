//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;

#if ARCANE_64BIT
using Integer = System.Int64;
#else
using Integer = System.Int32;
#endif
using Real = System.Double;

namespace Arcane
{
  public class EntryPoint
  {
   private EntryPoint_INTERNAL m_cpp_entry_point;
   private IFunctor m_functor;
   public EntryPoint(IModule module, 
                     string name,
                     IFunctor.FunctorDelegate callback,
                     string where,
                     int property)
   {
     m_functor = new IFunctor.Wrapper(callback);
     m_cpp_entry_point = EntryPoint_INTERNAL.Create(new EntryPointBuildInfo(module,name,m_functor,where,property,false));
   }
  }
}
