//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;

namespace Arcane
{
  public class CaseOptionServiceT<InterfaceType> : CaseOptionServiceImpl where InterfaceType : class
  {
    CaseOptionServiceContainer<InterfaceType> m_container = new CaseOptionServiceContainer<InterfaceType>();
    public CaseOptionServiceT(Arcane.CaseOptionBuildInfo opt,bool allow_null,bool is_optional)
    : base(opt,allow_null,is_optional)
    {
      SetContainer(m_container);
      //Console.WriteLine("CREATE C# CASE OPTION SERVICE INSTANCE");
    }
    // TODO: rendre obsolète et utiliser `Instance` à la place.
    public InterfaceType value()
    {
      return Instance;
    }

    public InterfaceType Instance
    {
      get {
        InterfaceType[] v = m_container.Values;
        if (v.Length!=0)
          return v[0];
        //Console.WriteLine("SERVICE VALUE {0}",m_instance);
        return null;
      }
    }
  }
}
