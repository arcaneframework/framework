/*---------------------------------------------------------------------------*/
/* OptionInfo.cs                                               (C) 2000-2007 */
/*                                                                           */
/* Classes de base de toutes les options d'un fichier AXL.                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using System;
using System.Xml;
using System.Text;

namespace Arcane.Axl
{
 public class OptionBuildInfo
 {
  private IAXLParser m_parser;
  public IAXLParser Parser { get { return m_parser; } }

  private XmlElement m_element;
  public XmlElement Element { get { return m_element; } }

  private Option m_parent_option;
  public Option ParentOption { get { return m_parent_option; } }

  private ServiceOrModuleInfo m_service_or_module;
  public ServiceOrModuleInfo ServiceOrModule { get { return m_service_or_module; } }

  private double m_version;
  public double Version { get { return m_version; } }

  public OptionBuildInfo(OptionBuildInfo build_info,XmlElement element,Option parent_option)
  {
    m_parser = build_info.Parser;
    m_element = element;
    m_parent_option = parent_option;
    m_service_or_module = parent_option.ServiceOrModule;
    m_version = build_info.Version;
  }

  public OptionBuildInfo(IAXLParser parser,XmlElement element,ServiceOrModuleInfo service_or_module,double version)
  {
    m_parser = parser;
    m_element = element;
    m_parent_option = null;
    if (service_or_module==null)
      throw new ArgumentException(String.Format("null 'service_or_module' for option '{0}'",element.Name));
    m_service_or_module = service_or_module;
    m_version = version;
  }

 }

 /**
  * Classes de base de toutes les options d'un fichier AXL. 
  */
  public abstract class Option : XmlInfo
  {
    public static int UNBOUNDED = -1;

    public Option(OptionBuildInfo build_info)
    {
      m_parent_option = build_info.ParentOption;
      m_service_or_module_info = build_info.ServiceOrModule;
      _Init(build_info.Element);
    }

    void _Init(XmlElement node)
    {
      m_node = node;
      m_node_name = node.Name;
      m_alternative_names = new NameTranslationsInfo(node);

      m_name = Utils.XmlGetAttributeValue(node,"name");
      m_type = Utils.XmlGetAttributeValue(node,"type");
      // Genere toujours le nom C++ correspondant
      if (m_type!=null)
        m_type = m_type.Replace(".","::");
      m_default = Utils.XmlGetAttributeValue(node,"default");

      string min_occurs = Utils.XmlGetAttributeValue(node,"minOccurs");
      if (String.IsNullOrEmpty(min_occurs))
        m_min_occurs = 1;
      else {
        m_has_min_occurs_attribute = true;
        bool is_ok = int.TryParse(min_occurs, out m_min_occurs);
        if (!is_ok) {
          Console.WriteLine("Valeur invalide pour l'attribut \"minOccurs\" de l'element \""
               + node.Name + "\". Utilise 1. (valeur='{0}')",min_occurs);
          m_min_occurs = 1;
        }
      }

      string max_occurs = Utils.XmlGetAttributeValue(node,"maxOccurs");
      if (String.IsNullOrEmpty (max_occurs))
        m_max_occurs = 1;
      else if (max_occurs == "unbounded") {
        m_has_max_occurs_attribute = true;
        m_max_occurs = UNBOUNDED;
      }
      else {
        m_has_max_occurs_attribute = true;
        bool is_ok = int.TryParse(max_occurs, out m_max_occurs);
        if (!is_ok) {
          Console.WriteLine("Valeur invalide pour l'attribut \"maxOccurs\" de l'element \""
               + node.Name + "\". Utilise 1. (valeur='{0}')",max_occurs);
          m_max_occurs = 1;
        }
      }
      
      m_description_element = Utils.GetElementIfExists(node,"description");
      m_user_classes = Utils.GetElementsValue(node,"userclass");

      {
        XmlNode optional_attr = node.GetAttributeNode("optional");
        if (optional_attr != null)
          m_is_optional = Utils.XmlParseStringToBool(optional_attr.Value);
      }
      if (m_is_optional && (m_max_occurs!=1 || m_min_occurs!=1))
        Error (node,"L'attribut 'optional' est incompatible avec les attributes 'minOccurs' et 'maxOccurs'");
    }

    public abstract void Accept(IOptionInfoVisitor v);
    /** Retourne <code>true</code> si l'option peut être présente plusieurs fois. */
    public bool IsMulti { get { return m_min_occurs != 1 || m_max_occurs != 1; } }
    /** Retourne <code>true</code> s'il y a une valeur par défaut pour l'option. */
    public bool HasDefault { get { return m_default != null; } }

    private ServiceOrModuleInfo m_service_or_module_info;

    /// Service ou module auquel apartient l'option
    public ServiceOrModuleInfo ServiceOrModule
    {
      get { return m_service_or_module_info; }
    } 

    private XmlElement m_node;
    public XmlElement Node { get { return m_node; } }

    protected string m_name;
    /** Valeur de l'attribut XML "name", nom de l'option */
    public string Name { get { return m_name; } }

    private string m_node_name;
    public string NodeName { get { return m_node_name; } }

    protected string m_type;
    /** Valeur de l'attribut XML "type", type de l'option */
    public string Type { get { return m_type; } }

    private string m_default;
    /** Valeur de l'attribut XML "default", valeur par défaut de l'option. */
    public string DefaultValue { get { return m_default; } }

    /**
     * Valeur de l'attribut XML "minoccurs",
     * nombre minimum d'occurences de l'option.
     */
    private int m_min_occurs;
    /**
     * Valeur de l'attribut XML "minoccurs",
     * nombre minimum d'occurences de l'option.
     */
    public int MinOccurs { get { return m_min_occurs; } }

    bool m_has_min_occurs_attribute;
    //! Indique si l'attribut 'minOccurs' est présent.
    public bool HasMinOccursAttribute { get { return m_has_min_occurs_attribute; } }

    /**
     * Valeur de l'attribut XML "maxoccurs",
     * nombre maximum d'occurences de l'option.
     */
    private int m_max_occurs;
    /**
     * Valeur de l'attribut XML "maxoccurs",
     * nombre maximum d'occurences de l'option.
     */
    public int MaxOccurs { get { return m_max_occurs; } }

    bool m_has_max_occurs_attribute;
    //! Indique si l'attribut 'maxOccurs' est présent.
    public bool HasMaxOccursAttribute { get { return m_has_max_occurs_attribute; } }

    private bool m_is_optional;
    /**
     * Indique si l'option est facultative.
     */
    public bool IsOptional { get { return m_is_optional; } }

    /** Différentes traductions de l'élément XML "name". */
    public NameTranslationsInfo m_alternative_names;

    private XmlElement m_description_element;
    /// Elément contenant la description de l'option (null si aucun)
    public XmlElement DescriptionElement { get { return m_description_element; } }

    private Option m_parent_option;

    /// Option parente (null si aucune)
    public Option ParentOption { get { return m_parent_option; } }

    /// Nom complet de l'option, préfixé par le nom complet de son parent
    public string FullName
    {
      get
      {
        if (m_parent_option!=null)
          return m_parent_option.FullName +"/"+Name;
        return Name;
      }
    }

    /// Nom de l'option dans la langue \a lang
    public string GetTranslatedName(string lang)
    {
      if (m_alternative_names.m_names.ContainsKey(lang))
        return m_alternative_names.m_names[lang];
      return m_name;
    }

    /// Nom complet de l'option dans la langue \a lang
    public string GetTranslatedFullName(string lang)
    {
      string tname = GetTranslatedName(lang);
      if (m_parent_option!=null)
        return m_parent_option.GetTranslatedFullName(lang) +"/"+tname;
      return tname;
    }

    string[] m_user_classes;
    public string[] UserClasses { get { return m_user_classes; } }

    //! Récupère une chaîne de caractère identifiant l'option (pour les messages d'erreur par exemple)
    public string GetIdString()
    {
      Option o = this;
      StringBuilder sb = new StringBuilder();
      sb.Append("(");
      sb.Append("option=");
      sb.Append(o.Name);
      ServiceOrModuleInfo smi = o.ServiceOrModule;
      sb.Append(",");
      if (smi.IsModule)
        sb.Append("module=");
      else
        sb.Append("service=");
      sb.Append(smi.Name);
      sb.Append(")");
      return sb.ToString();
    }

  }
}

