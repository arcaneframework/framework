//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
﻿using System;
using System.Linq;
using System.Collections.Generic;
namespace Arcane.Axl
{
  public partial class CaseOptionsT4
  {
    public string Version { get; private set; }

    public OptionHandler Xml { get; private set; }

    public Xsd.Base Base { get; private set; }

    public bool ComplexContainsServiceInstance { get; private set; }

    public string IncludePath { get; private set; }

    public CaseOptionsT4 (Xsd.Module module, string include_path, string version)
    {
      IncludePath = include_path.Replace ("_", "/");
      Version = version;
      Xml = new OptionHandler (module);
      Base = module;
      _SetHasServiceInstance ();
    }

    public CaseOptionsT4 (Xsd.Service service, string include_path, string version)
    {
      IncludePath = include_path.Replace ("_", "/");
      Version = version;
      Xml = new OptionHandler (service);
      Base = service;
      _SetHasServiceInstance ();
    }
    void _SetHasServiceInstance ()
    {
      ComplexContainsServiceInstance = false;
      foreach (var complex in Xml.FlatteningComplex.Where (p => !p.IsRef)) {
        foreach (var e in complex.Xml.ServiceInstance) {
          ComplexContainsServiceInstance = true;
          break;
        }
      }
    }
  }
}

