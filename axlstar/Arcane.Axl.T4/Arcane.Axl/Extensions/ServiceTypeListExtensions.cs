//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;

namespace Arcane.Axl
{
  public static class ServiceTypeListExtensions
  {
    public static string Name(this Xsd.ServiceTypeList type)
    {
      switch(type){
      case Xsd.ServiceTypeList.application:
        return "Application";
      case Xsd.ServiceTypeList.caseoption:
        return "CaseOption";
      case Xsd.ServiceTypeList.session:
        return "Session";
      case Xsd.ServiceTypeList.subdomain:
        return "SubDomain";
      case Xsd.ServiceTypeList.mesh:
        return "Mesh";
      default:
        throw new TypeUnloadedException (); 
      }
    }
  }
}

