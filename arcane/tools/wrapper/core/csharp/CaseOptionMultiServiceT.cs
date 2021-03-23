//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Arcane
{
  public class CaseOptionMultiServiceT<InterfaceType>
    : CaseOptionMultiServiceImpl where InterfaceType : class
  {
    CaseOptionServiceContainer<InterfaceType> m_container = new CaseOptionServiceContainer<InterfaceType>();

    public CaseOptionMultiServiceT(Arcane.CaseOptionBuildInfo opt,bool allow_null)
    : base(opt,allow_null)
    {
      SetContainer(m_container);
      Console.WriteLine("CREATE C# CASE OPTION MULTI SERVICE INSTANCE");
    }

    public InterfaceType[] Values
    {
      get { return m_container.Values; }
    }
  }
}
