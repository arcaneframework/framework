//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;

namespace Arcane
{
  public class ServiceAttribute : Attribute
  {
    //! Nom du service
    public string Name { set; get; }
    
    public System.Type InterfaceType { set; get; }

    public Arcane.ServiceType Type { set; get; }

    public ServiceAttribute(string name,Type interface_type)
    {
      Name = name;
      InterfaceType = interface_type;
      this.Type = Arcane.ServiceType.ST_CaseOption | Arcane.ServiceType.ST_SubDomain;
    }

    public ServiceAttribute(string name,Type interface_type,Arcane.ServiceType service_type)
    {
      Name = name;
      InterfaceType = interface_type;
      this.Type = service_type;
    }
  }
}
