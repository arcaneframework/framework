//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Linq;
using System.Collections.Generic;
using Newtonsoft.Json;

namespace Arcane.Axl.Xsd
{
  public partial class Service
  {
    [JsonIgnore ()]
    public IEnumerable<testsTest> Tests { get { return tests.test; } }

    public bool HasTests { get { return tests != null && Tests.Count () > 0; } }

    public string ParentName
    {
      get {
        if (String.IsNullOrEmpty(parentname))
        {
          if (type == ServiceTypeList.mesh)
            return "Arcane::BasicMeshService";
          else if (type == ServiceTypeList.caseoption)
            return "Arcane::BasicCaseOptionService";
          else if (type == ServiceTypeList.subdomain)
            return "Arcane::BasicSubDomainService";
          else
            return "Arcane::AbstractService";
        }
        return parentname;
      }
    }

    public IEnumerable<@interface> Interfaces { get { return @interface ?? new @interface[0]; } }

    public IEnumerable<@interface> InheritedInterfaces
    {
      get {
        return Interfaces.Where(p => p.inheritedSpecified ? p.inherited: true);
      }
    }

    public bool IsNotCaseOption {
      get {
        return type != ServiceTypeList.caseoption;
      }
    }

    public IEnumerable<Xsd.variablesVariable> Variables {
      get {
        if (variables == null)
          return new List<Xsd.variablesVariable> ();
        else
          return new List<Xsd.variablesVariable> (variables);
      }
    }
    public bool IsSingleton {
      get {
        return singleton;
      }
    }
    public void CheckValid ()
    {
      foreach (var v in Variables)
        v.InitializeAndValidate ();
    }
  }
}

