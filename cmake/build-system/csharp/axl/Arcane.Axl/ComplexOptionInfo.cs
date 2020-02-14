/*---------------------------------------------------------------------------*/
/* ComplexOptionInfo.cc                                        (C) 2000-2006 */
/*                                                                           */
/* Classe stockant les informations de l'élément XML "complex".              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
using System;
using System.Xml;
using System.Collections.Generic;

namespace Arcane.Axl
{
  /**
  * Classe stockant les informations de l'élément XML "complex". 
  */
  public class ComplexOptionInfo
    : Option
  {
    public ComplexOptionInfo(OptionBuildInfo build_info)
      : base(build_info)
    {
      XmlElement node = build_info.Element;
      m_reference_type_name = node.GetAttribute("ref");
      m_interface_type = node.GetAttribute("interface");
      if (String.IsNullOrEmpty(m_reference_type_name) && String.IsNullOrEmpty(m_type))
        Error (node,String.Format("L'attribut 'type' est manquant pour l'option complexe '{0}'",m_name));
      if (!String.IsNullOrEmpty (m_reference_type_name)) {
        if (HasMaxOccursAttribute) {
          Console.WriteLine ("WARNING: L'attribut 'maxOccurs' est incompatible avec l'attribut 'ref' (option='{0}')", m_name);
          //Error (node, String.Format ("L'attribut 'maxOccurs' est incompatible avec l'attribut 'ref' (option='{0}')", m_name));
        }
        if (HasMinOccursAttribute){
          Console.WriteLine ("WARNING: L'attribut 'minOccurs' est incompatible avec l'attribut 'ref' (option='{0}')",m_name);
          //Error (node,String.Format("L'attribut 'minOccurs' est incompatible avec l'attribut 'ref' (option='{0}')",m_name));
        }
      }
      foreach (XmlNode node_elem in node) {
        if (node_elem.NodeType != XmlNodeType.Element)
          continue;
        OptionBuildInfo sub_build_info = new OptionBuildInfo(build_info,(XmlElement)node_elem,this);
        Option opt = build_info.Parser.ParseSubElements(sub_build_info);
        if (opt != null){
          m_option_info_list.Add(opt);
        }
      }
    }

    public override void Accept(IOptionInfoVisitor v)
    {
      v.VisitComplex(this);
    }

    public void AcceptChildren(IOptionInfoVisitor v)
    {
      foreach(Option oi in Options){
        oi.Accept(v);
      }
    }

    public void AcceptChildrenSorted(IOptionInfoVisitor v,string lang)
    {
      SortedList<string,Option> sorted_options = new SortedList<string, Option>();
      bool use_sorted = false;
      try{
        foreach (Option o in Options) {
          string name = o.GetTranslatedName(lang);
          sorted_options.Add(name,o);
        }
        use_sorted = true;
      }
      catch(Exception e){
        Console.WriteLine("WARNING: Exception caught during sorting e={0}",e.Message);
        foreach (Option o in Options) {
          string tr_name = o.GetTranslatedName(lang);
          Console.WriteLine("Option name={0} translated_name={1}",o.Name,tr_name);
        }
        use_sorted = false;
      }
      if (use_sorted){
        foreach( KeyValuePair<string,Option> opt in sorted_options){
          opt.Value.Accept(v);
        }
      }
      else{
        foreach(Option oi in Options){
          oi.Accept(v);
        }
      }
    }

    string m_reference_type_name;
    /**
     * Valeur de l'attribut XML "ref", référencant un type complexe
     * définit précédemment
     */
    public string ReferenceTypeName { get { return m_reference_type_name; } }

    ComplexOptionInfo m_reference_type;
    //! Option de référence (seulement si ReferenceTypeName est non nul)
    public ComplexOptionInfo ReferenceType { get { return m_reference_type; } set { m_reference_type = value; } }

    string m_interface_type;
    //! Interface dont hérite l'option (facultatif, nul si aucune)
    public string InterfaceType { get { return m_interface_type; } }

    //! Contenu de l'élément XML "options".
    private List<Option> m_option_info_list = new List<Option>();

    //! Liste des options fille de cette option
    public IList<Option> Options { get { return m_option_info_list; } }
  }
}
