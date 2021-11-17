using System.Xml;
using System;

namespace Arcane.Axl
{
  /**
   * Classe stockant les informations de l'élément XML "entry-point". 
   */
  public class EntryPointInfo : XmlInfo
  {
    public EntryPointInfo(XmlElement node)
    {
      m_method_name = node.GetAttribute("method-name");

      if (m_method_name == null)
        AttrError(node, "method-name");
      
      // La premiere lettre doit etre une minuscule en C++ et une majuscule en C#.
      m_method_name = m_method_name.Substring(0,1).ToLower() + m_method_name.Substring(1);
      
      m_name = node.GetAttribute("name");
      if (m_name == null)
        AttrError(node, "name");

      // where
      m_where = node.GetAttribute("where");
      if (m_where != "compute-loop" &&
          m_where != "build" &&
          m_where != "init" &&
          m_where != "on-mesh-changed" &&
          m_where != "on-mesh-refinement" &&
          m_where != "continue-init" &&
          m_where != "start-init" &&
          m_where != "restore" &&
          m_where != "exit") {
        Console.WriteLine("** ERREUR: attribut \"where\" de l'option <" + node.Name
                          + "> invalide (" + m_where + ").\n");
        Console.WriteLine("** Les types reconnus sont 'compute-loop', 'build', 'init', 'on-mesh-changed', "
             + "'on-mesh-refinement', 'continue-init', 'start-init', 'restore' et 'exit'.\n");
        Error(node, "mauvaise valeur pour l'attribut \"where\"");
      }

      // property
      string property = node.GetAttribute("property");
      if (property == "none")
        m_property = Property.PNone;
      else if (property == "auto-load-begin")
        m_property = Property.PAutoLoadBegin;
      else if (property == "auto-load-end")
        m_property = Property.PAutoLoadEnd;
      else {
        Console.WriteLine("** ERREUR: attribut \"property\" de l'option <" + node.Name
             + "> invalide (" + property + ").\n");
        Console.WriteLine("** Les types reconnus sont 'none', 'auto-load-begin', "
             + " et 'auto-load-end'.\n");
        Error(node, "mauvaise valeur pour l'attribut \"property\"");
      }
    }

    string m_method_name;
    /**
     * Valeur de l'attribut XML "method-name", nom de la méthode C++ correspondant
     * au point d'entrée.
     */
    public string MethodeName { get { return m_method_name; } }

    string m_name;
    /** Valeur de l'attribut XML "name", nom déclaé du point d'entrée. */
    public string Name { get { return m_name; } }

    string m_where;
    /**
     * Valeur de l'attribut XML "where", endroit d'appel du point d'entrée
     * (cf tyoe Arcane correspondant, fichier arcane/IEntryPoint.h).
     */
    public string Where { get { return m_where; } }

    Property m_property;
    //! Propriété du point d'entrée..
    public Property Property { get { return m_property; } }

    public string UpperMethodName
    {
      get { return m_method_name.Substring(0,1).ToUpper() + m_method_name.Substring(1); }
    }
 
  }
}
