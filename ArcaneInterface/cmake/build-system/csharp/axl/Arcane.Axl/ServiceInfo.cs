using System.Collections.Generic;
using System.Xml;
using System;
namespace Arcane.Axl
{
  /**
   * Classe stockant les informations de l'élément XML "service". 
   */
  public class ServiceInfo : ServiceOrModuleInfo
  {
    public class Interface
    {
      public Interface()
      {
        m_is_inherited = true;
      }
      public Interface(string name, bool is_inherited)
      {
        m_name = name;
        m_is_inherited = is_inherited;
      }
      string m_name;
      //! Nom de l'interface
      public string Name { get { return m_name; } }

      bool m_is_inherited;
      //! Vrai si on dérive de l'interface
      public bool IsInherited { get { return m_is_inherited; } }
    }

    /** Vrai si le service n'est pas du type CaseOption. */
    bool m_not_caseoption;
    public bool NotCaseOption { get { return m_not_caseoption; } }

    /** Type du service */
    ServiceType m_service_type;
    public ServiceType ServiceType { get { return m_service_type; } }

    /** Liste des interfaces qu'implémente le service. */
    List<Interface> m_interfaces = new List<Interface>();
    public IList<Interface> Interfaces { get { return m_interfaces; } }

    /** Liste des méthodes de test */
    public TestsInfo m_tests_info;

    public ServiceInfo(AXLParser parser,XmlElement node,string file_base_name)
    : base(node,file_base_name,false)
    {
      m_not_caseoption = false;
      m_service_type = ServiceType.ST_Unknown;

      {
        string st = node.GetAttribute("type");
        if (st == "application")
          m_service_type = ServiceType.ST_Application;
        else if (st == "session")
          m_service_type = ServiceType.ST_Session;
        else if (st == "subdomain")
          m_service_type = ServiceType.ST_SubDomain;
        else if (st == "caseoption")
          m_service_type = ServiceType.ST_CaseOption;
        if (m_service_type == ServiceType.ST_Unknown)
          {
            Console.WriteLine("WARNING: Attribute 'type' will use default value 'caseoption'");
            m_service_type = ServiceType.ST_CaseOption;
            // AttrError(node, "type");
          }
        if (m_service_type != ServiceType.ST_CaseOption)
          m_not_caseoption = true;
      }

      if (String.IsNullOrEmpty(ParentName)) {
        if (m_service_type == ServiceType.ST_CaseOption || m_service_type == ServiceType.ST_SubDomain) {
          ParentName = "Arcane::BasicService";
        }
        else
          ParentName = "Arcane::AbstractService";
      }

      foreach (XmlNode e in node) {
        string name = e.Name;
        if (name == "interface") {
          XmlElement elem = (XmlElement)e;
          string interface_name = elem.GetAttribute("name");
          if (interface_name == null)
            AttrError(elem, "name");
          bool is_inherited = Utils.XmlParseNode(elem.GetAttributeNode("inherited"), true);
          m_interfaces.Add(new Interface(interface_name, is_inherited));
        }
        else if (name == "options") {
          _CreateOptions(parser,e);
        }
        else if (name == "variables") {
          _CreateVariables(parser,e);
        }
        else if (name == "tests") {
          m_tests_info = new TestsInfo(e as XmlElement);
          if (m_tests_info.m_test_info_list.Count > 0)
            m_interfaces.Add(new Interface("Arcane::IXmlUnitTest", false));
        }
      }
      if (Version >= 1.0 && m_interfaces.Count == 0) {
        throw new ArgumentException("service must have at least one tag <interface>");
      }
    }
  }
}

