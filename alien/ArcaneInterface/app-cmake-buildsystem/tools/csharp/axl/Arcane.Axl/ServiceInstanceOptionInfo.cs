/*---------------------------------------------------------------------------*/
/* ServiceInstanceOptionInfo.h                                 (C) 2000-2007 */
/*                                                                           */
/* Classe stockant les informations de l'élément XML "service-instance".     */
/*---------------------------------------------------------------------------*/
using System.Xml;

namespace Arcane.Axl
{
  /**
   * Classe stockant les informations de l'élément de configuration XML "service". 
   */
  public class ServiceInstanceOptionInfo
    : Option
  {

    /**
     * \brief \a true si le service spécifié n'est pas indispensable.
     */
    private bool m_allow_null;
    //! \a true si le service spécifié n'existe pas forcément.
    public bool AllowNull { get { return m_allow_null; } }
    /**
     * \brief \a true si le service spécifié est optionel.
     *
     * Si un service optionel n'est pas présent dans le jeu de données, il n'est
     * par alloué.
     */
    //private bool m_is_optional;
    //! \a true si le service spécifié est optionel.
    //public bool IsOptional { get { return m_is_optional; } }

    public ServiceInstanceOptionInfo(OptionBuildInfo build_info)
    : base(build_info)
    {
      XmlElement node = build_info.Element;
      if (m_name == null)
        AttrError(node, "name");
      if (m_type == null)
        AttrError(node, "type");

      {
        XmlNode allow_null_attr = node.GetAttributeNode("allow-null");
        if (allow_null_attr != null)
          m_allow_null = Utils.XmlParseStringToBool(allow_null_attr.Value);
      }
      // Maintenant dans la classe de base
      //{
      //  XmlNode optional_attr = node.GetAttributeNode("optional");
      //  if (optional_attr != null)
      //    m_is_optional = Utils.XmlParseStringToBool(optional_attr.Value);
      //}
    }

    public override void Accept(IOptionInfoVisitor v)
    { v.VisitServiceInstance(this); }
  }
}
