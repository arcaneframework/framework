//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using Axlstar.Axl;

namespace Arcane.Axl.Xsd
{
  public partial class Base
  {
    public string Name { get { return name1; } }

    public IEnumerable<name> Names {
      get {
        if (name == null)
          return new List<name> ();
        else
          return new List<name> (name);
      }
    }

    public IEnumerable<defaultvalue> DefaultValues {
      get {
        if (defaultvalue == null)
          return new List<defaultvalue> ();
        else
          return new List<defaultvalue> (defaultvalue);
      }
    }

    /// <summary>
    /// Gets or sets the content of the axl file encoded in Base64
    /// </summary>
    /// <value>The content of the axl.</value>
    public FileContent AxlContent { get; set; }

    public Base()
    {
      AxlContent = new FileContent ();
    }
  }
}
