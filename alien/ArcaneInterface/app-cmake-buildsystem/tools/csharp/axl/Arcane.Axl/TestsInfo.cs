/*---------------------------------------------------------------------------*/
/* TesrInfo.h                                                  (C) 2000-2012 */
/*                                                                           */
/* Classe stockant les informations de l'élément XML "test".                 */
/*---------------------------------------------------------------------------*/
using System.Xml;
using System;
using System.Collections.Generic;
namespace Arcane.Axl
{
  /**
   * Classe stockant les informations de l'élément XML "test". 
   */
  public class TestsInfo : XmlInfo
  {
    public TestsInfo(XmlElement node)
    {
	  m_class_set_up = node.GetAttribute("class-set-up");
	  m_test_set_up = node.GetAttribute("test-set-up");
	  m_class_tear_down = node.GetAttribute("class-tear-down");
	  m_test_tear_down = node.GetAttribute("test-tear-down");

	  // La premiere lettre doit etre une minuscule en C++ et une majuscule en C#.
	  m_class_set_up = ToFirstLower(m_class_set_up);
	  m_test_set_up = ToFirstLower(m_test_set_up);
	  m_class_tear_down = ToFirstLower(m_class_tear_down);
	  m_test_tear_down = ToFirstLower(m_test_tear_down);

      foreach (XmlNode elem in node) {
        if (elem.Name == "test") {
          TestInfo test = new TestInfo((XmlElement)elem);
          m_test_info_list.Add(test);
        }
      }
	}

    /**
     * Valeur de l'attribut XML "class-set-up", nom de la méthode C++ correspondant à la méthode d'initialisation de la classe de test.
     */
    public string m_class_set_up;
    /**
     * Valeur de l'attribut XML "test-set-up", nom de la méthode C++ correspondant à la méthode d'initialisation de chaque test.
     */
    public string m_test_set_up;
    /**
     * Valeur de l'attribut XML "class-tear-down", nom de la méthode C++ correspondant à la méthode appelée après tous les tests de la classe.
     */
    public string m_class_tear_down;
    /**
     * Valeur de l'attribut XML "test-tear-down", nom de la méthode C++ correspondant à la méthode appelée après chaque test.
     */
    public string m_test_tear_down;
	/**
	 * Contenu de l'élément XML "tests" définissant les méthodes de test.
	 */
	public List<TestInfo> m_test_info_list = new List<TestInfo>();
    
    public string UpperClassSetUp { get { return ToFirstUpper(m_class_set_up); } }
	public string UpperTestSetUp { get { return ToFirstUpper(m_test_set_up); } }
    public string UpperClassTearDown { get { return ToFirstUpper(m_class_tear_down); } }
	public string UpperTestTearDown { get { return ToFirstUpper(m_test_tear_down); } }
		
	private string ToFirstUpper(string name)
	{
	  if (String.IsNullOrEmpty(name)) return name;
	  else return name.Substring(0,1).ToUpper() + name.Substring(1);
	}
		
	private string ToFirstLower(string name)
	{
	  if (String.IsNullOrEmpty(name)) return name;
	  else return name.Substring(0,1).ToLower() + name.Substring(1);
	}
  }
}
