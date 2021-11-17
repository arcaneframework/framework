/*---------------------------------------------------------------------------*/
/* NameTranslationsInfo.h                                      (C) 2000-2006 */
/*                                                                           */
/* Classe permettant de stocker les traductions des éléments XML "name".     */
/* Par exemple: <name lang='fr'>...</name>.                                  */
/*---------------------------------------------------------------------------*/
using System;
using System.Xml;
using System.Collections.Specialized;

namespace Arcane.Axl
{
  /**
   * Classe permettant de stocker les traductions des éléments XML "name".
   * Par exemple: <name lang='fr'>...</name>.
   */
  public class NameTranslationsInfo : XmlInfo
  {
    public NameTranslationsInfo(XmlElement node)
    {
      foreach (XmlNode node_elem in node) {
        if (node_elem.Name != "name")
          continue;
        XmlElement elem = (XmlElement)node_elem;
        string alt_lang = elem.GetAttribute("lang");
        if (alt_lang == null)
          continue;
        string alt_value = elem.InnerText;
        if (alt_value == null)
          continue;
        m_names.Add(alt_lang, alt_value);
      }
    }
    
    /**
      * Liste de noms dépendants du langage.
      * La clé est est le nom du langage, au format utilisé par XML
      * (par exemple, 'en' pour anglais, 'fr' pour francais.
      * La valeur est le nom correspondant dans ce langage.
      */
    public StringDictionary m_names = new StringDictionary();
  }
}
