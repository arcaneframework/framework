//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
﻿using System;
using System.Collections.Generic;
using System.Linq;

namespace Arcane.Axl.Xsd
{
  public partial class Module
  {
    public string ParentName {
      get {
        if (String.IsNullOrEmpty (parentname)) {
          return "Arcane::BasicModule";
        }
        return parentname;
      }
    }

    public IEnumerable<@interface> Interfaces {
      get {
        if (@interface==null)
          return new List<Xsd.@interface> ();
        return @interface;
      }
    }

    public IEnumerable<@interface> InheritedInterfaces {
      get {
        return Interfaces.Where (p => p.inheritedSpecified ? p.inherited : true);
      }
    }

    public IEnumerable<entrypointsEntrypoint> EntryPoints {
      get {
        if (entrypoints == null)
          return new List<entrypointsEntrypoint> ();
        else
          return new List<entrypointsEntrypoint> (entrypoints);
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

    public void CheckValid ()
    {
      foreach (var v in Variables)
        v.InitializeAndValidate ();
    }
  }
}

