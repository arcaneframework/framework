//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Xml;
using System.Xml.Linq;
using System.Text;
using System.IO;
using Arcane.Axl;
using Arcane.AxlDoc.UserInterfaces;

namespace Arcane.AxlDoc
{
  public class DefaultApplicationPages : IApplicationPages
  {
    public DefaultApplicationPages ()
    {

    }

    // Configure plugin options
    public void Configure (Mono.Options.OptionSet options) 
    {

    }

    public void Summary()
    {

    }

    // Filter Xml node
    public bool Filter(ServiceOrModuleInfo info)
    {
      return true;
    }

    // Filter Xml node
    public bool Filter(Option option)
    {
      return true;
    }

    // Write an application page for a given service or module
    public void writeApplicationPage(ServiceOrModuleData data, string output_path)
    {

    }
    
    // Write applicative description for an option (where applicative xml nodes are defined within description node)
    public void writeApplicationDescription(Option option, IEnumerable<XElement> option_description_children_nodes, TextWriter option_description_text_stream)
    {

    }
  }
}

