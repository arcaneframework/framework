using System.Collections.Generic;
using System.Xml;
using System;

namespace Arcane.Axl
{
  /**
   * \brief Classe stockant les informations de l'élément XML "module"
   * ou "service".
   *
   * Note: pour des raisons de compatibilité avec les anciennes versions des
   * fichiers axl (anciennement CaseOptions.xml), cette classe
   * stocke aussi les informations des éléments XML "module-info" et "service-info".
   */
  public class ServiceOrModuleInfo : XmlInfo
  {
    string m_tag_name;
    //! Nom de l'élément XML racine (stocké pour des raisons de compatibilité). Vaut "module" ou "service".
    public string TagName { get { return m_tag_name; } }
    
    string m_name;
    /** Valeur de l'attribut XML "name", nom du module */
    public string Name { get { return m_name; } }

    /** Valeur de l'attribut XML "parent-name", classe de base du service. */
    string m_parent_name;
    /** Valeur de l'attribut XML "parent-name", classe de base du service. */
    public string ParentName { get { return m_parent_name; } protected set { m_parent_name = value; } }

    string m_namespace_macro_name;
    /** Valeur de l'attribut XML "namespace-macro-name" du module */
    public string NamespaceMacroName { get { return m_namespace_macro_name; } }

    string m_namespace_name;
    /** Valeur de l'attribut XML "namespace-name" du service */
    public string NamespaceName { get { return m_namespace_name; } }

    double m_version;
    /** Version (valeur de l'attribut XML "version") du service. */
    public double Version { get { return m_version; } }

    /** Contenu de l'élément XML "options". */
    List<Option> m_option_info_list = new List<Option>();
    public IList<Option> Options { get { return m_option_info_list; } }

    /** Contenu de l'élément XML "variables" définissant les variables du code. */
    List<VariableInfo> m_variable_info_list = new List<VariableInfo>();
    public IList<VariableInfo> VariableInfoList { get { return m_variable_info_list; } }

    NameTranslationsInfo m_alternative_names;
    /** Différentes traductions de l'élément XML "name". */
    public NameTranslationsInfo AlternativeNames { get { return m_alternative_names; } }

    bool m_is_module;
    //! Vrai s'il s'agit d'un module
    public bool IsModule { get { return m_is_module; } }

    string m_file_base_name;
    public string FileBaseName { get { return m_file_base_name; } }

    XmlElement m_description_element;
    /// Elément contenant la description de l'option (null si aucun)
    public XmlElement DescriptionElement { get { return m_description_element; } }

    string[] m_user_classes;
    public string[] UserClasses { get { return m_user_classes; } }

    public ServiceOrModuleInfo(XmlElement node,string file_base_name,bool is_module)
    {
      m_file_base_name = file_base_name;
      m_is_module = is_module;
      m_version = 0.0;
      m_alternative_names = new NameTranslationsInfo(node);

      m_tag_name = node.Name;
      m_name = Utils.XmlGetAttributeValue(node,"name");
      if (m_name == null)
        AttrError(node, "name");

      m_namespace_macro_name = Utils.XmlGetAttributeValue(node,"namespace-macro-name");
      m_namespace_name = Utils.XmlGetAttributeValue(node,"namespace-name");
      if (m_namespace_macro_name != null && m_namespace_name != null) {
        Console.WriteLine("VALUE: namespace-macro-name {0}",m_namespace_macro_name);
        Console.WriteLine("VALUE: namespace-name {0}",m_namespace_name);
        throw new ArgumentException("Only one attribute 'namespace-macro-name' or 'namespace-name' may be specified");
      }

      {
        string version_str = Utils.XmlGetAttributeValue(node,"version");
        if (version_str == null) {
          Console.WriteLine("WARNING: Attribute 'version' must have a value");
        }
        else {
          if (version_str == "1.0") {
            m_version = 1.0;
          }
          else
            throw new ArgumentException("Bad value for attribute 'version' (only value '1.0' is supported) value=" + version_str);
        }
      }

      m_parent_name = Utils.XmlGetAttributeValue(node,"parent-name");
      m_description_element = Utils.GetElementIfExists(node,"description");
      m_user_classes = Utils.GetElementsValue(node,"userclass");
    }

    //! Nom du service dans la lang \a lang
    public string GetTranslatedName(string lang)
    {
      if (m_alternative_names.m_names.ContainsKey(lang))
        return m_alternative_names.m_names[lang];
      return m_name;
    }

    /** Cree les infos sur les variables */
    protected void _CreateVariables(IAXLParser parser,XmlNode variables_elem)
    {
      IAXLObjectFactory factory = parser.Factory;
      foreach (XmlNode elem in variables_elem) {
        if (elem.Name == "variable") {
          VariableInfo variable = factory.CreateVariableInfo((XmlElement)elem);
          m_variable_info_list.Add(variable);
        }
      }
    }

    /** Cree les infos sur les options. */
    protected void _CreateOptions(IAXLParser parser,XmlNode options_elem)
    {
      foreach (XmlNode elem in options_elem) {
        if (!(elem is XmlElement))
          continue;
        OptionBuildInfo sub_build_info = new OptionBuildInfo(parser,(XmlElement)elem,this,m_version);
        Option opt = parser.ParseSubElements(sub_build_info);
        if (opt != null){
          Options.Add(opt);
        }
      }

      // Pour les options complexes qui référencent un type, il faut trouver le type correspondant.
      Dictionary<string,ComplexOptionInfo> dict = new Dictionary<string,ComplexOptionInfo>();
      // Liste des options complexes faisant référence à une autre option complexe.
      List<ComplexOptionInfo> ref_list = new List<ComplexOptionInfo>();
      _FillComplexTypes(dict,ref_list,Options);
      foreach(ComplexOptionInfo coi in ref_list){
        string ref_name = coi.ReferenceTypeName;
        ComplexOptionInfo ref_coi = null;
        bool is_found = dict.TryGetValue(ref_name,out ref_coi);
        if (!is_found)
          throw new ApplicationException(String.Format("No reference type with name '{0}' exist",ref_name));
        coi.ReferenceType = ref_coi;
      }
    }

    void _FillComplexTypes(Dictionary<string,ComplexOptionInfo> dict,List<ComplexOptionInfo> ref_list,IList<Option> opts)
    {
      foreach(Option opt in opts){
        ComplexOptionInfo copt = opt as ComplexOptionInfo;
        if (copt==null)
          continue;
        if (!String.IsNullOrEmpty(copt.ReferenceTypeName)){
          ref_list.Add(copt);
          continue;
        }
        string ctype = copt.Type;
        if (!String.IsNullOrEmpty(ctype)){
          if (dict.ContainsKey(ctype))
            throw new ApplicationException(String.Format("Duplicate type with name '{0}'",ctype));
          dict.Add(ctype,copt);
        }
        _FillComplexTypes(dict,ref_list,copt.Options);
      }
    }

    /**
     * Applique le visiteur \a visitor aux options du service ou module.
     */
    public void ApplyVisitor(IOptionInfoVisitor visitor)
    {
      foreach(Option o in Options)
        o.Accept(visitor);
    }
  }
}
