//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Text;
using System.Net;
using System.Xml.Linq;
using System.Linq;
using Arcane.Axl;
using Arcane.AxlDoc.UserInterfaces;

namespace Arcane.AxlDoc
{
  public class OptionXmlNodeHandler
  {
    private XElement m_option_node;

    public XElement Option_node { get { return m_option_node; } set { m_option_node = value; } }

    private Option m_option;

    public Option Option { get { return m_option; } set { m_option = value; } }

    public OptionXmlNodeHandler (XElement option_node, Option option)
    {
      m_option_node = option_node;
      m_option = option;
    }
  }

  public class XmlSchemaFile
  {
    private XNamespace schema_namespace = "http://www.w3.org/2001/XMLSchema";
    private XDocument m_xsd_file;

    public XDocument Document { get { return m_xsd_file; } }
        
    private SortedList<string, OptionXmlNodeHandler> m_current_service_or_module_options;

    public SortedList<string, OptionXmlNodeHandler> Current_service_or_module_options { get { return m_current_service_or_module_options; } }
        
    private string m_current_service_or_module_name;
    private string m_current_service_or_module_type_name;
    private string m_filename;
    private Config m_config;

    public bool IsActivated { get { return m_config.do_generate_xsd != XmlGenerationMode.None; } }

    public bool IsExportInfoNeeded { get { return m_config.do_generate_xsd == XmlGenerationMode.WithInfo; } }

    public SortMode DoSort { get { return m_config.do_sort; } }

    public IApplicationPages private_app_pages { get { return m_config.private_app_pages; } }

    public XNamespace Schema_namespace { get { return schema_namespace; } set { schema_namespace = value; } }

    public XmlSchemaFile (Config config, string application_name)
    {
      m_config = config;

      if (m_config.do_generate_xsd == XmlGenerationMode.None)
        return;
      m_filename = application_name + ".xsd";
      // XNamespace empNM = "urn:lst-emp:emp";
      m_xsd_file = new XDocument (
                // root node
            new XDeclaration ("1.0", "UTF-16", null), 
                new XElement (schema_namespace + "schema", 
                new XAttribute (XNamespace.Xmlns + "xs", schema_namespace.NamespaceName), new XAttribute ("elementFormDefault", "qualified"),
                    new XComment (" Generated at " + DateTime.Now + " by " + Environment.UserName + " "),
                new XElement (schema_namespace + "element", new XAttribute ("name", "case"), 
                    new XElement (schema_namespace + "complexType",
                    new XElement (schema_namespace + "all",
                    // arcane node
                    new XElement (schema_namespace + "element", new XAttribute ("name", "arcane"),
                    new XElement (schema_namespace + "complexType",
                    new XElement (schema_namespace + "all",
                    new XElement (schema_namespace + "element", new XAttribute ("name", "title"), new XAttribute ("type", "xs:string")),
                    new XElement (schema_namespace + "element",
                    new XAttribute ("name", "timeloop"), new XAttribute ("type", "xs:string")
      )))), // end arcane node
                    // mesh node
                    new XElement (schema_namespace + "element",
                    new XAttribute ("name", "mesh"),
                    new XElement (schema_namespace + "complexType",
                    new XElement (schema_namespace + "all",
                    new XElement (schema_namespace + "element",
                    new XAttribute ("name", "file"),
                    new XElement (schema_namespace + "complexType",
                    new XElement (schema_namespace + "simpleContent",
                    new XElement (schema_namespace + "extension",
                    new XAttribute ("base", "xs:string"),
                    new XElement (schema_namespace + "attribute",
                    new XAttribute ("name", "unique"),
                    new XAttribute ("type", "xs:normalizedString")),
                    new XElement (schema_namespace + "attribute",
                    new XAttribute ("name", "partitioner"),
                    new XAttribute ("type", "xs:normalizedString")),
                    new XElement (schema_namespace + "attribute",
                    new XAttribute ("name", "internal-partition"),
                    new XAttribute ("type", "xs:normalizedString"))
      ))))))) // end mesh node
      ),
                    // root node attributes
                    new XElement (schema_namespace + "attribute", new XAttribute ("name", "codename"), new XAttribute ("type", "xs:string")),
                    new XElement (schema_namespace + "attribute", new XAttribute ("name", "codeversion"), new XAttribute ("type", "xs:string")),
                    new XElement (schema_namespace + "anyAttribute", new XAttribute ("namespace", "http://www.w3.org/XML/1998/namespace"), new XAttribute ("processContents", "lax"))
      ))));

    }

    internal void beginBuildModuleOrServiceSchema (ServiceOrModuleInfo info)
    {
      if (m_config.do_generate_xsd == XmlGenerationMode.None)
        return;
      m_current_service_or_module_name = Utils.ToLowerWithDash(info.Name);
      m_current_service_or_module_type_name = XmlSchemaFile.ConvertName (info.Name) + "-module-type";
      // if module, add type in main element
      if (info.IsModule) {
        // Add module node only if module has options
        if (info.Options.Count > 0) {
          XElement module_element = m_xsd_file.Element (schema_namespace + "schema").Element (schema_namespace + "element").Element (schema_namespace + "complexType").Element (schema_namespace + "all");
          module_element.Add (new XElement (schema_namespace + "element", new XAttribute ("name", m_current_service_or_module_name), new XAttribute ("type", m_current_service_or_module_type_name), new XAttribute ("minOccurs", "0")));
        }
      } else { // if service 
        string service_type_name = generateServiceTypeName (info.Name);
      }
      // Initialize option list
      m_current_service_or_module_options = new SortedList<string, OptionXmlNodeHandler> (info.Options.Count);
    }

    internal int applySortTransform(int i) {
      if (m_config.do_sort == SortMode.IndexAlpha)
        return i;
      else 
        return 0;
    }

    internal void endBuildModuleOrServiceSchema (ServiceOrModuleInfo info, CodeInfo code_info)
    {
      if (m_config.do_generate_xsd == XmlGenerationMode.None)
        return;
      // Select module and service options and Re order following the order that may be defined by the application (using node description/option-index in axl files)
      var grouped_by_index_current_options = from c in m_current_service_or_module_options 
                                             where c.Value.Option.ParentOption == null 
                                             group c.Value by applySortTransform (c.Value.Option.OptionIndex) into newGroup 
                                             orderby newGroup.Key 
                                             select newGroup;
      // Create service or module complexType
      if (info.IsModule && info.Options.Count > 0) {
        XElement module_complex_type = new XElement (schema_namespace + "complexType", new XAttribute ("name", m_current_service_or_module_type_name),
                                                    new XElement (schema_namespace + "sequence", from index_group_options in grouped_by_index_current_options 
                                                                                              from option in index_group_options select option.Option_node));                                    
                                    
        m_xsd_file.Element (schema_namespace + "schema").Add (module_complex_type);
      } else if (!info.IsModule) {
        // Create service base type if needed (ie not already defined)
        createBaseServiceNode (generateServiceTypeName (info.Type));
        // Create derive type for the current service implementation, add it on the schema top node
        string service_name = info.Name;
        CodeInterfaceInfo interface_info;
        if (code_info.Interfaces.TryGetValue (info.Type, out interface_info)) {
          CodeServiceInfo service_info = interface_info.Services.Find (csi => csi.Name == service_name);
          if (service_info != null) {
            foreach (string name in service_info.AliasesName) {
              service_name += "|" + name;
            }
          }
        }
        m_xsd_file.Element (schema_namespace + "schema").Add (
                    new XElement (Schema_namespace + "complexType", new XAttribute ("name", generateServiceInterfaceName (info.Type) + "_" + info.Name + "-type"),
                        new XElement (Schema_namespace + "complexContent", 
                            new XElement (Schema_namespace + "extension", new XAttribute ("base", generateServiceTypeName (info.Type)),
                                new XElement (schema_namespace + "sequence", from index_group_options in grouped_by_index_current_options 
                                                                            from option in index_group_options select option.Option_node),
                                new XElement (Schema_namespace + "attribute", new XAttribute ("name", "name"),
                                    new XElement (Schema_namespace + "simpleType",
                                        new XElement (Schema_namespace + "restriction", new XAttribute ("base", "xs:string"),
                                            new XElement (Schema_namespace + "pattern", new XAttribute ("value", service_name)))))))));
                                                                            
      }

    }

    public void addOptions (string name, OptionXmlNodeHandler option_element)
    {
      m_current_service_or_module_options.Add (name, option_element);
    }

    public void write ()
    {
      if (m_config.do_generate_xsd == XmlGenerationMode.None)
        return;
      // Check all service or module types have been created. If not, create an empty type. This situation can be valid if the application does not compile a service that could be needed but never used
      var all_types = from type in m_xsd_file.Descendants (Schema_namespace + "element") 
                            where (type.Attribute ("type") != null 
                                && (type.Attribute ("type").Value != "xs:string" 
                                 && type.Attribute ("type").Value != "xs:double" 
                                 && type.Attribute ("type").Value != "xs:integer")) 
                            select type.Attribute ("type").Value; 
      var non_existing_types = from type in all_types 
                where (m_xsd_file.Element (Schema_namespace + "schema")
                                 .Elements (Schema_namespace + "complexType").DefaultIfEmpty (null).FirstOrDefault (complex_type => complex_type.Attribute ("name").Value == type) == null) 
                select type;
      foreach (string non_existing_type in non_existing_types) {
        Console.WriteLine ("FOUND NON EXISTING TYPE " + non_existing_type + ". Create an empty type " + non_existing_type);
        m_xsd_file.Element (Schema_namespace + "schema").Add (new XElement (Schema_namespace + "complexType", new XAttribute ("name", non_existing_type)));
      }
      m_xsd_file.Save (m_filename);
    }

    public static string generateServiceTypeName (string service_type)
    {
      return service_type.Replace ("::", "-") + "-type";
    }

    public static string generateServiceInterfaceName (string service_interface)
    {
      return service_interface.Replace ("::", "-");
    }

    public void createBaseServiceNode (string service_type_name)
    {
      // When creating a service : create the node for the first occurence of a service implementing a given interface
      if (m_config.verbose)
        System.Console.WriteLine ("Treating service " + service_type_name);
      var existing_service_element = from element in m_xsd_file.Element (schema_namespace + "schema").Elements (schema_namespace + "complexType")
                                            where element.Attribute ("name").Value == service_type_name
                                            select element;
      if (existing_service_element.Count () == 0) {
        if (m_config.verbose)
          System.Console.WriteLine ("Adding service " + service_type_name);
        m_xsd_file.Element (schema_namespace + "schema").Add (new XElement (schema_namespace + "complexType", new XAttribute ("name", service_type_name)));
      }
    }

    public static string ConvertName(string name) {
      return name;
      // No more conversion but legacy conversion is kept here 
      // return Utils.ToLowerWithDash(name);
    }
  }
}
