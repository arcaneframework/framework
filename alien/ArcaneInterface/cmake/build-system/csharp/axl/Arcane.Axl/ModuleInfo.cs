using System.Collections.Generic;
using System.Xml;
using System;

namespace Arcane.Axl
{
  /**
   * Classe stockant les informations de l'élément XML "module". 
   * Note: pour des raisons de compatibilité avec les anciennes versions des
   * fichiers axl (anciennement CaseOptions.xml), cette classe
   * stocke aussi les informations des éléments XML "module-info".
   */
  public class ModuleInfo : ServiceOrModuleInfo
  {
    bool m_is_autoload;
    /**
     * Valeur de l'attribut XML "autoload", vrai si le module est chargé
     * tout le temps, i.e. meme s'il n'est pas utilisé dans la boucle en temps.
     */
    public bool IsAutoload { get { return m_is_autoload; } }

    List<EntryPointInfo> m_entry_point_info_list = new List<EntryPointInfo>();
    /**
     * Contenu de l'élément XML "entry-points" définissant les points
     * d'entrée du code.
     */
    public IList<EntryPointInfo> EntryPointInfoList { get { return m_entry_point_info_list; } }

    public ModuleInfo(AXLParser parser,XmlElement node,string file_base_name)
      : base(node,file_base_name,true)
    {
      m_is_autoload = false;

      m_is_autoload = Utils.XmlParseNode(node.GetAttributeNode("autoload"),false);

      if (String.IsNullOrEmpty(ParentName)) {
        ParentName = "Arcane::BasicModule";
      }
      foreach (XmlNode e in node) {
        string name = e.Name;
        if (name == "options") {
          _CreateOptions(parser,e);
        }
        else if (name == "variables") {
          _CreateVariables(parser,e);
        }
        else if (name == "entry-points") {
          foreach (XmlNode elem in e) {
            if (elem.Name == "entry-point") {
              EntryPointInfo entry_point = new EntryPointInfo((XmlElement)elem);
              m_entry_point_info_list.Add(entry_point);
            }
          }
        }
        else if (name == "tests") {
          throw new ArgumentException("module can't have a tag <tests>");		
        }
      }
    }
  }
}
