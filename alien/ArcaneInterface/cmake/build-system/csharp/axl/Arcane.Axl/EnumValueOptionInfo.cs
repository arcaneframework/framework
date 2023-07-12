/*---------------------------------------------------------------------------*/
/* EnumValueOptionInfo.h                                       (C) 2000-2007 */
/*                                                                           */
/* Classe stockant les informations de l'élément XML "enumvalue".            */
/*---------------------------------------------------------------------------*/
using System.Xml;

namespace Arcane.Axl
{
  /**
   * Classe stockant les informations de l'élément XML "enumvalue". 
   */
  public class EnumValueOptionInfo : XmlInfo
  {
    public EnumValueOptionInfo(XmlElement node)
    {
      m_alternative_names = new NameTranslationsInfo(node);
      m_name = node.GetAttribute("name");
      if (m_name == null)
        AttrError(node, "name");
      m_genvalue = node.GetAttribute("genvalue");
      if (m_genvalue == null)
        AttrError(node, "genvalue");
      m_description_element = Utils.GetElementIfExists(node,"description");
    }

    /** Valeur de l'attribut XML "name", nom de la valeur de l'énuméré. */
    private string m_name;
    
    /** Valeur de l'attribut XML "name", nom de la valeur de l'énuméré. */
    public string Name { get { return m_name; } }

    public string GetTranslatedName(string lang)
    {
      if (m_alternative_names.m_names.ContainsKey(lang))
        return m_alternative_names.m_names[lang];
      return m_name;
    }

    /**
     * Valeur de l'attribut XML "genvalue".
     * attribut C++ correspondant a l'énuméré.
     */
    private string m_genvalue;

    /**
     * Valeur de l'attribut XML "genvalue".
     * attribut C++ correspondant a l'énuméré.
     */
    public string GeneratedValue { get { return m_genvalue; } }

    /** Différentes traductions de l'élément XML "name". */
    public NameTranslationsInfo m_alternative_names;

    private XmlElement m_description_element;
    /// Elément contenant la description de l'option (null si aucun)
    public XmlElement DescriptionElement { get { return m_description_element; } }
  }
}
