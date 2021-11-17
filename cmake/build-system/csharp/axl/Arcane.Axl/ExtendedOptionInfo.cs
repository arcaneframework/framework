/*---------------------------------------------------------------------------*/
/* ExtendedOptionInfo.h                                        (C) 2000-2007 */
/*                                                                           */
/* Classe stockant les informations de l'élément XML "extended".             */
/*---------------------------------------------------------------------------*/
using System.Xml;
namespace Arcane.Axl
{
/**
 * Classe stockant les informations de l'élément XML "extended". 
 */
  public class ExtendedOptionInfo : Option
  {
    public ExtendedOptionInfo(OptionBuildInfo build_info)
    : base(build_info)
    {
      XmlElement node = build_info.Element;
      if (m_name == null)
        AttrError(node, "name");
      if (m_type == null)
        AttrError(node, "type");
    }

    public override void Accept(IOptionInfoVisitor v)
    {
      v.VisitExtended(this);
    }
  }
}
