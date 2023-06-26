/*
 CodeInfo.cs (C) 2000-2011

 Informations sur les modules et service d'un code
 disponibles dans un code. Cela necessite d'avoir
 generer les infos en lancant arcane avec l'option -arcane_opt arcane_internal
*/

using System.Xml;
using System;
using System.Collections.Generic;
using System.IO;

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

    private List<CodeServiceInfo> m_services = new List<CodeServiceInfo>();
    public List<CodeServiceInfo> Services { get { return m_services; } }

    private string m_filename;
  
    string m_language;
    //! Langage utilisé pour la génération de la documentation
    public string Language { get { return m_language; } set { m_language = value; } }

    public CodeInfo(string filename)
    {
      m_language = "en";
      m_filename = filename;
      if (!File.Exists(m_filename)){
        Console.WriteLine("WARNING: file '{0}' does not exists or is unreadable.",filename);
        return;
      }
      _Read();
    }

    private void _Read()
    {
      XmlDocument doc = new XmlDocument();
      doc.Load(m_filename);
      XmlElement root = doc.DocumentElement;
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
        CodeModuleInfo module = new CodeModuleInfo(module_name);
        m_modules.Add(module);
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

        CodeServiceInfo service = new CodeServiceInfo(service_name,interfaces_name.ToArray());
        m_services.Add(service);
        if (filebase_name!=null)
          service.FileBaseName = filebase_name;
        foreach(string s in interfaces_name){
          _AddServiceForInterface(service,s);
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

    private string[] m_interfaces_name;
    /// <summary>
    /// Liste des interfaces implementees par ce service
    /// </summary>
    public string[] InterfacesName { get { return m_interfaces_name; } }

    public CodeServiceInfo(string name,string[] interfaces_name)
    {
      m_name = name;
      m_interfaces_name = interfaces_name;
    }
  }

  public class CodeInterfaceInfo
  {
    private string m_name;
    public string Name { get { return m_name; } }

    private List<CodeServiceInfo> m_services = new List<CodeServiceInfo>();
    public List<CodeServiceInfo> Services { get { return m_services; } }

    public CodeInterfaceInfo(string name)
    {
      m_name = name;
    }
  }
}
