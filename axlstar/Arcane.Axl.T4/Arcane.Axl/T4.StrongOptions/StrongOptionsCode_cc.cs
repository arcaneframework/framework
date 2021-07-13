//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Linq;

namespace Arcane.Axl
{
  public partial class StrongOptions_cc
  {
    public string Version { get; private set; }
    
    public OptionHandler Xml { get; private set; }
    
    public Xsd.Base Base { get; private set; }
    
    public StrongOptions_cc (Xsd.Module module, string version)
    {
      Version = version;
      Xml = new OptionHandler (module);
      Base = module;
    }
    
    public StrongOptions_cc (Xsd.Service service, string version)
    {
      Version = version;
      Xml = new OptionHandler (service);
      Base = service;
    }
  }
}


