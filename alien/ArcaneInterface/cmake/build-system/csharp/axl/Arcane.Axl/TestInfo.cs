/*---------------------------------------------------------------------------*/
/* TesrInfo.h                                                  (C) 2000-2012 */
/*                                                                           */
/* Classe stockant les informations de l'élément XML "test".                 */
/*---------------------------------------------------------------------------*/
using System.Xml;
using System;
namespace Arcane.Axl
{
  /**
   * Classe stockant les informations de l'élément XML "test". 
   */
  public class TestInfo : XmlInfo
  {
    public TestInfo(XmlElement node)
    {
      m_method_name = node.GetAttribute("method-name");

      if (m_method_name == null)
        AttrError(node, "method-name");
      
      // La premiere lettre doit etre une minuscule en C++ et une majuscule en C#.
      m_method_name = m_method_name.Substring(0,1).ToLower() + m_method_name.Substring(1);
      
      m_name = node.GetAttribute("name");
      if (m_name == null)
        AttrError(node, "name");
    }

    /**
     * Valeur de l'attribut XML "method-name", nom de la méthode C++ correspondant
     * à la méthode de test.
     */
    public string m_method_name;
    /** Valeur de l'attribut XML "name", nom déclaré de la méthode de test (utile pour le rapport de test). */
    public string m_name;
    
    public string UpperMethodName
    {
      get { return m_method_name.Substring(0,1).ToUpper() + m_method_name.Substring(1); }
    }
  }
}
