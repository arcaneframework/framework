//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Xml.Linq;
using System.Text;
using System.IO;
using System.Reflection;
using Arcane.Axl;
using Arcane.AxlDoc.UserInterfaces;

namespace Arcane.AxlDoc
{
  public class ApplicationPageVisitor : IExtraDescriptionWriter
  {
    public ApplicationPageVisitor (Config config)
    {
      m_config = config;
    }

    public void VisitServiceOrModuleInfo (ServiceOrModuleInfo info)
    {
      OptionInfoVisitor visitor = new OptionInfoVisitor (m_config.private_app_pages);
      foreach (Option option in info.Options) {
        if (m_config.private_app_pages.Filter (option))
          option.Accept(visitor);
      }
      m_config.private_app_pages.writeApplicationPage (new ServiceOrModuleData (info.Name, visitor.OptionDataList), m_config.output_path);
    }

    void IExtraDescriptionWriter.writeDescription (Option option, IEnumerable<XElement> description_nodes, TextWriter stream)
    {
      m_config.private_app_pages.writeApplicationDescription (option, description_nodes, stream);
    }

    private static XElement _descriptionElement (Option o)
    {
      return (o.DescriptionElement == null ? null : XElement.Parse (o.DescriptionElement.OuterXml));
    }

    public class OptionInfoVisitor : IOptionInfoVisitor
    {
      public OptionInfoVisitor (IApplicationPages private_app_pages)
      {
        m_private_app_pages = private_app_pages;
        OptionDataList = new List<OptionData> ();
      }
      
      void IOptionInfoVisitor.VisitComplex (ComplexOptionInfo o)
      {
        OptionInfoVisitor visitor = new OptionInfoVisitor (m_private_app_pages);
        o.AcceptChildren (visitor, x => m_private_app_pages.Filter (x));
        OptionDataList.Add (new OptionData (o.Name, visitor.OptionDataList, _descriptionElement (o)));
      }
      
      void IOptionInfoVisitor.VisitEnumeration (EnumerationOptionInfo o)
      {
        OptionDataList.Add (new OptionData (o.Name, null, _descriptionElement (o)));
      }
      
      void IOptionInfoVisitor.VisitExtended (ExtendedOptionInfo o)
      {
        OptionDataList.Add (new OptionData (o.Name, null, _descriptionElement (o)));
      }
      
      void IOptionInfoVisitor.VisitScript (ScriptOptionInfo o)
      {
        OptionDataList.Add (new OptionData (o.Name, null, _descriptionElement (o)));
      }
      
      void IOptionInfoVisitor.VisitSimple (SimpleOptionInfo o)
      {
        OptionDataList.Add (new OptionData (o.Name, null, _descriptionElement (o)));
      }
      
      void IOptionInfoVisitor.VisitServiceInstance (ServiceInstanceOptionInfo o)
      {
        OptionDataList.Add (new OptionData (o.Name, null, _descriptionElement (o)));
      }

      #region MEMBERS
      IApplicationPages m_private_app_pages;
      public List<OptionData> OptionDataList { get; private set; }
      #endregion
    }

    #region MEMBERS
    private Config m_config;
    #endregion
  }
}
