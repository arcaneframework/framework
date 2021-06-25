//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Xml;
using System.Xml.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using Arcane.Axl;

namespace Arcane.AxlDoc
{
  namespace UserInterfaces
  {
    public class BaseData
    {
      protected string m_name;

      public string Name { get { return m_name; } }

      protected List<OptionData> m_options; // for complex option only
      public List<OptionData> Options { get { return m_options; } }

      protected XElement m_description;

      public XElement Description { get { return m_description; } }
    }
     
    public class OptionData : BaseData
    {
      public OptionData (string name, List<OptionData> sub_options = null, XElement description = null)
      { 
        this.m_name = name; 
        this.m_options = sub_options ?? new List<OptionData> ();
        this.m_description = description ?? new XElement ("EmptyElement");
      }
    }
    
    public class ServiceOrModuleData : BaseData
    {
      public ServiceOrModuleData (string name, List<OptionData> options = null)
      {
        this.m_name = name;
        m_options = options ?? new List<OptionData> ();
      }
    }
    
    public interface IApplicationPages
    {
      // Configure options for plugins
      void Configure (Mono.Options.OptionSet options);

      // Summary execution
      void Summary ();

      // Filter Service or Module info
      bool Filter (ServiceOrModuleInfo info);

      // Filter individual option
      bool Filter (Option option);

      // Write an application page for a given service or module
      void writeApplicationPage (ServiceOrModuleData data, string output_path);
        
      // Write applicative description for an option (where applicative xml nodes are defined within description node)
      void writeApplicationDescription (Option option, IEnumerable<XElement> option_description_children_nodes, TextWriter option_description_text_stream);
    }
  }
}
