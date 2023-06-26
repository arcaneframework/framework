/*---------------------------------------------------------------------------*/
/* EnumerationOptionInfo.h                                     (C) 2000-2008 */
/*                                                                           */
/* Classe stockant les informations de l'élément XML "enumeration".          */
/*---------------------------------------------------------------------------*/
using System.Xml;
using System.Collections.Generic;

namespace Arcane.Axl
{
  /**
   * Classe stockant les informations de l'élément XML "enumeration". 
   */
  public class EnumerationOptionInfo : Option
  {
    public EnumerationOptionInfo(OptionBuildInfo build_info)
    : base(build_info)
    {
      XmlElement node = build_info.Element;
      if (m_name == null)
        AttrError(node, "name");
      if (m_type == null)
        AttrError(node, "type");

      // Regarde la liste des éléments de l'énumération
      foreach (XmlNode elem in node) {
        if (elem.Name != "enumvalue")
          continue;
        EnumValueOptionInfo ev = new EnumValueOptionInfo((XmlElement)elem);
        m_enum_value_list.Add(ev);
      }
    }

    public override void Accept(IOptionInfoVisitor v)
    {
      v.VisitEnumeration(this);
    }

    /** Contenu de l'élément XML "enumeration". Liste des éléments "enumvalue". */
    private List<EnumValueOptionInfo> m_enum_value_list = new List<EnumValueOptionInfo>();

    public List<EnumValueOptionInfo> EnumValues { get { return m_enum_value_list; } }

    /// Nom de l'option énumérée \a value dans le langage \a lang
    public string GetTranslatedEnumerationName(string value,string lang)
    {
      foreach(EnumValueOptionInfo o in EnumValues){
        if (o.Name==value)
          return o.GetTranslatedName(lang);
      }
      return value;
    }
  }
}
