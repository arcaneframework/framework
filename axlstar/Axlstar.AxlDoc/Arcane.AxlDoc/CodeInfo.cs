//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*
 CodeInfo.cs (C) 2000-2023

 Informations sur les modules et service d'un code
 disponibles dans un code. Cela necessite d'avoir
 generer les infos en lancant arcane avec l'option -arcane_opt arcane_internal
*/

using System.Xml;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Arcane.Axl
{    
  /*!
   * \brief Informations sur les modules et service d'un code.
   */
  public class CodeInfo
  {
    private List<CodeModuleInfo> m_modules = new List<CodeModuleInfo>();
    public List<CodeModuleInfo> Modules { get { return m_modules; } }

    private Dictionary<string,CodeInterfaceInfo> m_interfaces = new Dictionary<string,CodeInterfaceInfo>();
    public Dictionary<string,CodeInterfaceInfo> Interfaces { get { return m_interfaces; } }

    private Dictionary<string, List<string>> m_services_aliases = new Dictionary<string, List<string>>();
    public Dictionary<string, List<string>> ServiceAliases { get { return m_services_aliases; } }

    private List<CodeServiceInfo> m_services = new List<CodeServiceInfo>();
    public List<CodeServiceInfo> Services { get { return m_services; } }

    private string m_filename;
    private string m_application_name;
    public string ApplicationName { get { return m_application_name; } }
  
    string m_language;
    //! Langage utilisé pour la génération de la documentation
    public string Language
    {
      get { return m_language; }
      set {
        if (m_language == value)
          return;
        m_language = value;
        Translation = new Translation(m_language);
      }
    }

    bool m_legacy;
    //! Permet de dire si l'on veut l'ancienne version de la documentation.
    public bool Legacy
    {
      get { return m_legacy; }
      set { m_legacy = value; }
    }

    public Translation Translation { get; private set; }

    public CodeInfo(string filename)
    {
      m_language = "en";
      m_filename = filename;
      if (!File.Exists(m_filename)){
        Console.WriteLine("WARNING: CodeInfo file '{0}' does not exists or is unreadable.",filename);
        return;
      }
      _Read();
    }

    private void _Read()
    {
      XmlDocument doc = new XmlDocument();
      doc.Load(m_filename);
      XmlElement root = doc.DocumentElement;
      if (root.HasAttribute("application-name")) m_application_name = root.GetAttribute("application-name");
      else m_application_name = "arcane-application";
      foreach(XmlNode node in root){
        XmlElement elem = node as XmlElement;
        if (elem==null)
          continue;
        if (elem.Name=="modules"){
          _ReadModules(elem);
        }
        else if (elem.Name=="services"){
          _ReadServices(elem);
        }
      }
    }

    private void _ReadModules(XmlElement elem)
    {
      XmlNodeList modules = elem.SelectNodes("module");
      foreach(XmlNode node in modules){
        string module_name = ((XmlElement)node).GetAttribute("name");
        XmlNodeList user_nodes = ((XmlElement)node).SelectNodes("has-service-instance"); // used to list services or modules using a given interface (to add hyperlinks)
        CodeModuleInfo module = new CodeModuleInfo(module_name);
        m_modules.Add(module);
        foreach (XmlNode user_node in user_nodes)
        {
            _AddUserForInterface(module.Name, ((XmlElement)user_node).GetAttribute("option-name"), ((XmlElement)user_node).GetAttribute("interface"),true);
        }
      }
    }

    private void _ReadServices(XmlElement elem)
    {
      XmlNodeList modules = elem.SelectNodes("service");
      foreach(XmlNode node in modules){
        XmlElement sub_elem = (XmlElement)node;
        string service_name = sub_elem.GetAttribute("name");
        string filebase_name = sub_elem.GetAttribute("file-base-name");
        XmlNodeList interface_nodes = sub_elem.SelectNodes("implement-class/@name");
        List<string> interfaces_name = new List<string>();
        foreach(XmlNode sub_node_interface in interface_nodes){
          string interface_name = ((XmlAttribute)sub_node_interface).Value;
          interfaces_name.Add(interface_name);
        }
        XmlNodeList alias_nodes = sub_elem.SelectNodes("alias/@name");
        List<string> aliases_name = new List<string>();
        foreach (XmlNode sub_node_alias in alias_nodes)
        {
            string alias_name = ((XmlAttribute)sub_node_alias).Value;
            aliases_name.Add(alias_name);
        }
        XmlNodeList user_nodes = sub_elem.SelectNodes("has-service-instance"); // used to list the possible service or module using a given service (to add hyperlinks)
        CodeServiceInfo service = new CodeServiceInfo(service_name,interfaces_name.ToArray(), aliases_name);
        m_services.Add(service);
        if (filebase_name!=null)
          service.FileBaseName = filebase_name;
        foreach(string s in interfaces_name){
          _AddServiceForInterface(service,s);
        }
        foreach(XmlNode user_node in user_nodes)
        {
            _AddUserForInterface( service.Name,((XmlElement) user_node).GetAttribute("option-name"), ((XmlElement)user_node).GetAttribute("interface"),false);
        }
      }
    }

    private void _AddServiceForInterface(CodeServiceInfo service,string interface_name)
    {
      if (!m_interfaces.ContainsKey(interface_name)){
        m_interfaces.Add(interface_name,new CodeInterfaceInfo(interface_name));
      }
      CodeInterfaceInfo interface_info = m_interfaces[interface_name];
      interface_info.Services.Add(service);
    }

     private void _AddUserForInterface(string user_name, string hyperlink, string interface_name, bool is_module)
    {
      if (!m_interfaces.ContainsKey(interface_name))
      {
          m_interfaces.Add(interface_name, new CodeInterfaceInfo(interface_name));
      }
      CodeInterfaceInfo interface_info = m_interfaces[interface_name];
      interface_info.Users.Add(new CodeInterfaceUserInfo(user_name,hyperlink,is_module));
    }
  }

  public class CodeModuleInfo
  {
    private string m_name;
    public string Name { get { return m_name; } }

    private List<string> m_variables = new List<string>();
    public List<string> Variables { get { return m_variables; } }

    public CodeModuleInfo(string name)
    {
      m_name = name;
    }
  }

  public class CodeServiceInfo
  {
    private string m_name;
    public string Name { get { return m_name; } }

    private string m_file_base_name;
    public string FileBaseName
    {
      get { return m_file_base_name; }
      set { m_file_base_name = value; }
    }

    readonly string[] m_interfaces_name;
    string[] m_aliases_name;
    string[] m_names; // Noms possibles pour le service (axl + ARCANE_REGISTER_SERVICE dans le cc)
    /// <summary>
    /// Liste des interfaces implementees par ce service
    /// </summary>
    public string[] InterfacesName { get { return m_interfaces_name; } }
    /// <summary>
    /// Liste des alias que peut prendre ce service (via les noms employés dans ARCANE_REGISTER_SERVICE)
    /// </summary>
    public string[] AliasesName { get { return m_aliases_name; } }
    public string[] Names { get { return m_names; } }

    public CodeServiceInfo(string name,string[] interfaces_name, List<string> aliases_name)
    {
      m_name = name;
      m_interfaces_name = interfaces_name;
      m_aliases_name = aliases_name.ToArray();
      List<string> names = new List<string>(aliases_name);
      names.Add(name);
      m_names = names.ToArray();
    }
  }

  public class CodeInterfaceInfo
  {
    private string m_name;
    public string Name { get { return m_name; } }

    private List<CodeServiceInfo> m_services = new List<CodeServiceInfo>();
    public List<CodeServiceInfo> Services { get { return m_services; } }

    private List<CodeInterfaceUserInfo> m_users = new List<CodeInterfaceUserInfo>();
    public List<CodeInterfaceUserInfo> Users { get { return m_users; } }

    public CodeInterfaceInfo(string name)
    {
      m_name = name;
    }
  }

  public class CodeInterfaceUserInfo
  {
      private string m_name;
      public string Name { get { return m_name; } }

      private string m_hyperlink;
      public string Hyperlink { get { return m_hyperlink; } }

      private string m_hyperlink_name;
      public string HyperlinkName { get { return m_hyperlink_name; } }

      private bool m_is_module;
      public bool IsModule { get { return m_is_module; } }
      
      public CodeInterfaceUserInfo(string user_name, string hyperlink_name, bool is_module)
      {
          m_name = user_name;
          m_is_module = is_module;
          m_hyperlink_name = hyperlink_name;
          m_hyperlink = hyperlink_name.Replace("-","_");
      }
  }
}
