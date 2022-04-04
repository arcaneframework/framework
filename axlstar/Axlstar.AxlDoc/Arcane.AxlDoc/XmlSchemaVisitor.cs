//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
﻿using System;
using System.Collections.Generic;
using System.Xml.Linq;
using System.Linq;
using Arcane.Axl;

namespace Arcane.AxlDoc
{
  public class XmlSchemaVisitor : IOptionInfoVisitor
  {
    XmlSchemaFile m_xsd_file;
    CodeInfo m_code_info;
    bool m_is_activated;
    bool m_do_export_info;
    SortMode m_do_sort;

    public XmlSchemaVisitor (XmlSchemaFile xsd_file, CodeInfo code_info)
    {
      m_xsd_file = xsd_file;
      m_code_info = code_info;
      m_is_activated = xsd_file.IsActivated;
      m_do_export_info = xsd_file.IsExportInfoNeeded;
      m_do_sort = xsd_file.DoSort;
    }

    internal void VisitServiceOrModule (ServiceOrModuleInfo info)
    {
      if (!m_is_activated) return;
      // Pour retrouver facilement une option dans la .xsd du service ou module, on peut la trier par ordre alphabetique (cd DoSort)
      List<Option> not_sorted_options = new List<Option> ();
      foreach (Option o in info.Options) {
        if (m_xsd_file.private_app_pages.Filter (o)) {
          string name = o.GetTranslatedName (m_code_info.Language);
          // Attention, il est possible par erreur que deux options aient le même nom traduit.
          // Pour corriger cette erreur spécifique peut être générée
          if (not_sorted_options.Exists (op => op.GetTranslatedName (m_code_info.Language) == name)) {
            throw new ApplicationException (String.Format ("option {0} has same traduction '{1}' as another option at the same level",
                                                           o.GetIdString (), name));
          }
          not_sorted_options.Add (o);
        }
      }
      if (m_do_sort != SortMode.None) {
        not_sorted_options.Sort ((a, b) => String.Compare (a.GetTranslatedName (m_code_info.Language),
                                                     b.GetTranslatedName (m_code_info.Language)));
      }
      m_xsd_file.beginBuildModuleOrServiceSchema (info);
      foreach (Option o in not_sorted_options) {
        o.Accept (this);
      }
      m_xsd_file.endBuildModuleOrServiceSchema (info, m_code_info);
    }

    public void VisitComplex (ComplexOptionInfo o)
    {
      if (!m_is_activated)
        return;
      if (m_xsd_file.DoSort != SortMode.None) {
        o.AcceptChildrenSorted (this, "fr", x => m_xsd_file.private_app_pages.Filter (x));
      } else {
        o.AcceptChildren (this, x => m_xsd_file.private_app_pages.Filter (x));
      }
      XElement option_element = new XElement (m_xsd_file.Schema_namespace + "element", new XAttribute ("name", XmlSchemaFile.ConvertName (o.Name)),
                                        new XElement (m_xsd_file.Schema_namespace + "complexType",
                                            new XElement (m_xsd_file.Schema_namespace + "sequence", from c in m_xsd_file.Current_service_or_module_options where (c.Value.Option.ParentOption != null && c.Value.Option.ParentOption.Name == o.Name) select c.Value.Option_node)));

      _handleMinMaxOccurs (o, option_element);
      m_xsd_file.addOptions (_generateUniqueOptionName (o, o.Name), new OptionXmlNodeHandler (option_element, o));
    }

    public void VisitEnumeration (EnumerationOptionInfo o)
    {
      if (!m_is_activated)
        return;
      var option_element = new XElement (m_xsd_file.Schema_namespace + "element", new XAttribute ("name", XmlSchemaFile.ConvertName (o.Name)),
                                         new XElement (m_xsd_file.Schema_namespace + "simpleType",
                                            new XElement (m_xsd_file.Schema_namespace + "restriction", new XAttribute ("base", "xs:string"),
                                                from e in o.EnumValues select new XElement (m_xsd_file.Schema_namespace + "enumeration", new XAttribute ("value", e.Name)))));
      _handleMinMaxOccurs (o, option_element);
      m_xsd_file.addOptions (_generateUniqueOptionName (o, o.Name), new OptionXmlNodeHandler (option_element, o));
    }

    public void VisitExtended (ExtendedOptionInfo o)
    {
      if (!m_is_activated)
        return;
      throw new NotImplementedException ();
    }

    public void VisitScript (ScriptOptionInfo o)
    {
      if (!m_is_activated)
        return;
      throw new NotImplementedException ();
    }

    public void VisitServiceInstance (ServiceInstanceOptionInfo o)
    {
      if (!m_is_activated)
        return;
      //  version with unique service type (incompatible with service having same interface and same options => no it's ok)
      XElement option_element = new XElement (m_xsd_file.Schema_namespace + "element", new XAttribute ("name", XmlSchemaFile.ConvertName (o.Name)), new XAttribute ("type", XmlSchemaFile.generateServiceTypeName (o.Type)));

      _handleMinMaxOccurs (o, option_element);
      m_xsd_file.addOptions (_generateUniqueOptionName (o, o.Name), new OptionXmlNodeHandler (option_element, o));
      // Prepare info to export (needed @IFPEN to validate .arc file using xsi::type)
      if (m_do_export_info) {
        CodeInterfaceInfo interface_info;
        if (m_code_info.Interfaces.TryGetValue (o.Type, out interface_info)) {
          var si = from service_info in interface_info.Services
                   from name in service_info.Names
                   select
                       new XElement (m_xsd_file.Schema_namespace + "annotation",
                           new XElement (m_xsd_file.Schema_namespace + "appinfo", o.Name + "_" + name),
                              new XElement (m_xsd_file.Schema_namespace + "documentation",
                                  XmlSchemaFile.generateServiceInterfaceName (o.Type) + "_" + service_info.Name + "-type"));
          m_xsd_file.Document.Element (m_xsd_file.Schema_namespace + "schema").Add (si);
        }
      }
    }

    public void VisitSimple (SimpleOptionInfo o)
    {
      if (!m_is_activated)
        return;
      // Handle real and integer type, for all other types, put "string" (= weak validation)
      string type;
      if (o.Type == "real")
        type = "xs:double";
      else if (o.Type == "integer")
        type = "xs:integer";
      else
        type = "xs:string";
      XElement option_element = new XElement (m_xsd_file.Schema_namespace + "element", new XAttribute ("name", XmlSchemaFile.ConvertName (o.Name)), new XAttribute ("type", type));
      _handleMinMaxOccurs (o, option_element);

      m_xsd_file.addOptions (_generateUniqueOptionName (o, o.Name), new OptionXmlNodeHandler (option_element, o));
    }

    private string _generateUniqueOptionName (Option option, string name)
    {
      string unique_name;
      if (option.ParentOption != null) {
        unique_name = option.ParentOption.Name + "_" + option.Name + "_" + name;
      } else {
        unique_name = XmlSchemaFile.ConvertName (option.ServiceOrModule.Name) + "_" + option.Name + "_" + name;
      }
      return unique_name;
    }

    private void _handleMinMaxOccurs (Option option, XElement option_element)
    {
      if (option.HasMaxOccursAttribute)
        option_element.Add (new XAttribute ("maxOccurs", option.MaxOccurs > 0 ? option.MaxOccurs.ToString () : "unbounded"));
      if (option.HasMinOccursAttribute)
        option_element.Add (new XAttribute ("minOccurs", option.MinOccurs));
      // handle complex option
      bool is_optional = false;
      if (option is ComplexOptionInfo) {
        is_optional = _isComplexOptionOptional ((ComplexOptionInfo)option);
      } else {
        is_optional = option.HasDefault || option.IsOptional;
      }
      if (is_optional) {
        if (option.HasMinOccursAttribute && option.MinOccurs > 0)
          option_element.Attribute ("minOccurs").SetValue (0);
        else if (!option.HasMinOccursAttribute) {
          option_element.Add (new XAttribute ("minOccurs", 0));
        }
      }
    }

    private bool _isComplexOptionOptional (ComplexOptionInfo option)
    {
      bool is_optional = true;
      foreach (Option sub_option in option.Options) {
        if (sub_option is ComplexOptionInfo)
          is_optional = is_optional && _isComplexOptionOptional ((ComplexOptionInfo)sub_option);
        else
          is_optional = is_optional && (sub_option.HasDefault || sub_option.IsOptional || (sub_option.HasMinOccursAttribute && sub_option.MinOccurs == 0));
      }
      return is_optional;
    }
  }
}
