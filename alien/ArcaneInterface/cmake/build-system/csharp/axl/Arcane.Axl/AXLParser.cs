using System.Xml;
using System;
using System.IO;
using System.Collections.Generic;
using System.Reflection;

namespace Arcane.Axl
{    
  /**
   * Classe permettant de lire le fichier AXL. Les informations lues sont stockées
   * dans les classes dont le nom est suffixé par Info: ModuleInfo, OptionInfo...
   * 
   * La creation d'un parseur se fait via la class AXLParserFactory.CreateParser().
   * 
   * Ensuite, pour lire les infos du fichier, il faut appeler ParseAXLFile() ou
   * ParseAXLFileForDocumentation(). La premiere methode est utilisee pour
   * la generation du fichier C++ ou C# et effectue certains tests de coherence
   * supplementaires par rapport a la deuxieme.
   */
  public class AXLParser : IAXLParser
  {
    /**
   * Retourne l'objet stockant les informations de l'élément XML module.
   * Cet élément est renseigné si l'objet m_service_info ne l'est pas.
   * @return l'instance de ModuleInfo, 0 sinon.
   */
    public ModuleInfo Module { get { return m_module; } }
    /**
   * Retourne l'objet stockant les informations de l'élément XML service.
   * Cet élément est renseigné si l'objet m_module_info ne l'est pas.
   * @return l'instance de ServiceInfo, 0 sinon.
   */
    public ServiceInfo Service { get { return m_service; } }

  /** Nom du fichier AXL a lire (sans l'extension). */
  private string m_file_name;
  /** Chemin complet du fichier AXL a lire. */
  private string m_full_file_name;
  /** Extension attendue du fichier AXL. */
  private string m_file_extension;
  /** Flux contenant le schema AXL. */
  private Stream m_schema_stream;
  /** Noeud racine du fichier AXL. */
  private XmlElement m_root;
  public XmlDocument Document
  {
    get
    {
      if (m_root!=null)
        return m_root.OwnerDocument;
      return null;
    }
  }
    
  /**
   * \brief Objet stockant les informations de l'élément XML module.
   *
   * Cet élément est renseigné si l'objet m_service_info ne l'est pas.
   */
  private ModuleInfo m_module;
  /**
   * Objet stockant les informations de l'élément XML service.
   * Cet élément est renseigné si l'objet m_module_info ne l'est pas.
   */
  private ServiceInfo m_service;

  private ServiceOrModuleInfo m_service_or_module;

  public ServiceOrModuleInfo ServiceOrModule { get { return m_service_or_module; } }

  private string m_user_class;

    private IAXLObjectFactory m_object_factory;
    /** Fabrique pour les objets des fichiers AXL. */
    public IAXLObjectFactory Factory { get { return m_object_factory; } set { m_object_factory = value; } }

    [Obsolete("Utiliser AXLParserFactory a la place")]
    public AXLParser(string full_file_name,string user_class)
    {
      m_user_class = user_class;
      m_file_name = Path.GetFileNameWithoutExtension(full_file_name);
      m_file_extension = Path.GetExtension(full_file_name);
      m_full_file_name = full_file_name;
      m_schema_stream = Utils.GetAxlSchemaAsStream();
      m_object_factory = new DefaultAXLObjectFactory();
    }

    internal AXLParser(string full_file_name,string user_class,Stream schema_stream,IAXLObjectFactory obj_factory)
    {
      m_user_class = user_class;
      m_file_name = Path.GetFileNameWithoutExtension(full_file_name);
      m_file_extension = Path.GetExtension(full_file_name);
      m_full_file_name = full_file_name;
      m_schema_stream = schema_stream; Utils.GetAxlSchemaAsStream();
      m_object_factory = obj_factory; //new DefaultAXLObjectFactory();
    }

  /**
   * \brief Lit et analyse le fichier AXL.
   *
   * Les informations lues sont stockées dans les classes dont le nom
   * est suffixé par Info: ModuleInfo, OptionInfo...
   */
  public void ParseAXLFile()
  {
    m_root = XmlInfo.RootNode(m_full_file_name,m_schema_stream);
    // Importe tous les fichiers <include-sub-axl>
    _SearchAndImportSubAxlFiles(new XmlElement[]{m_root},0);

    if(GlobalContext.Instance.Verbose){ 
      Console.WriteLine("ROOT NODE {0}",m_root);
    }
    // juste une petite vérification (cf message ci-dessous)
    string name_value = m_root.GetAttribute("name");
    if (m_file_extension == ".axl" && name_value != m_file_name)
    {
      string s;
      s = "La valeur de l'attribut \"name\" de l'élément racine du fichier \""
         + m_file_name + "\" doit etre identique"
          + " au nom du fichier \"axl\" donné dans le fichier \"config\".";
      throw new InternalErrorException(s);
    }
    
    string root_name = m_root.Name;
    if (root_name == "module"){
      m_module = new ModuleInfo(this,m_root,m_file_name);
      m_service_or_module = m_module;
    }
    else if (root_name == "service"){
      m_service = new ServiceInfo(this,m_root,m_file_name);
      m_service_or_module = m_service;
    }
    else
    {
      string s = "La valeur de l'attribut racine du fichier \""
      + m_file_name + "\" est incorrecte: " + root_name;
      throw new InternalErrorException(s);
    }
  }

  /**
   * Lit et analyse le fichier AXL pour documentation.
   * Les informations lues sont stockées dans les classes dont le nom
   * est suffixé par Info: ModuleInfo, OptionInfo...
   */
  public void ParseAXLFileForDocumentation()
  {
    m_root = XmlInfo.RootNode(m_full_file_name,m_schema_stream);
    // Importe tous les fichiers <include-sub-axl>
    _SearchAndImportSubAxlFiles(new XmlElement[]{m_root},0);
    if(GlobalContext.Instance.Verbose){ 
      Console.WriteLine("ROOT NODE {0}",m_root);
    }
    
    string root_name = m_root.Name;
    if (root_name == "module"){
      m_module = new ModuleInfo(this,m_root,m_file_name);
      m_service_or_module = m_module;
    }
    else if (root_name == "service"){
      m_service = new ServiceInfo(this,m_root,m_file_name);
      m_service_or_module = m_service;
    }
    else
    {
      string s = "La valeur de l'attribut racine du fichier \""
      + m_file_name + "\" est incorrecte: " + root_name;
      throw new InternalErrorException(s);
    }
  }

  public Option ParseSubElements(OptionBuildInfo build_info)
  {
    Option opt = null;
    XmlElement element = build_info.Element;
    if (!_IsValidUserClass(element))
      return null;
    string name = element.Name;
    if (name == "extern")
      opt = new SimpleOptionInfo(build_info);
    if (name == "simple")
      opt = new SimpleOptionInfo(build_info);
    else if (name == "script")
      opt = new ScriptOptionInfo(build_info);
    else if (name == "extended")
      opt = new ExtendedOptionInfo(build_info);
    else if (name == "enumeration")
      opt = new EnumerationOptionInfo(build_info);
    else if (name == "complex")
      opt = new ComplexOptionInfo(build_info);
    else if (name == "service-instance")
      opt = new ServiceInstanceOptionInfo(build_info);
    else if (name == "include-sub-axl"){
      // Rien à faire, l'importation se fait au moment de l'analyse du fichier.
    }
    else if (name == "description"){
      // Rien à faire
    }
    else if (name == "name"){
      // Rien à faire
    }
    else if (name == "userclass"){
      // Rien à faire
    }
    else
      Console.WriteLine("In AXLParser.ParseSubElements() Unexpected element <{0}>",name);
    return opt;
  }

  /**
   * \brief Cherche et importe les fichiers sub-axl.
   *
   * Cherche récursivement parmi les éléments \a elements ceux qui
   * correspondent à une inclusion d'un fichier sub-axl. Ensuite,
   * procède à l'inclusion de ces fichiers en les important comme
   * s'ils étaient directement présent dans le fichier axl d'origine.
   *
   * Les nouveaux éléments importés sont ensuite analysés récursivement
   * au cas où ils importeraient aussi d'autres fichiers sub-axl.
   *
   * Pour se prémunir d'éventuelles dépendances cycliques
   * entre les fichiers inclus, on limite la récursivité.
   *
   * \todo supprimer les éléments \a 'include-sub-axl' après inclusion.
   */
  private void _SearchAndImportSubAxlFiles(XmlElement[] elements,int level)
  {
    if (level>256)
      throw new InternalErrorException("Too many inclusion in sub-axl files. "+
                                       "Check for cyclic dependencies between files");

    List<XmlElement> sub_axl_elements = new List<XmlElement>();
    foreach(XmlElement e in elements)
      _SearchSubAxlFiles(e,sub_axl_elements);
    if (sub_axl_elements.Count==0)
      return;

    List<XmlElement> new_elements = new List<XmlElement>();
    Console.WriteLine("SUB AXL AXL COUNT={0} FILE={1}",sub_axl_elements.Count,m_full_file_name);
    foreach(XmlElement sub_elem in sub_axl_elements){
      _ImportSubAxlFile(sub_elem,new_elements);
    }
    // Maitenant, supprime les éléments d'inclusion
    foreach(XmlElement sub_elem in sub_axl_elements){
      sub_elem.ParentNode.RemoveChild(sub_elem);
    }
    if (new_elements.Count!=0)
      _SearchAndImportSubAxlFiles(new_elements.ToArray(),level+1);
  }

  /*!
   * \brief Cherche récursivement les éléments qui importent un fichier sub-axl.
   *
   * Les éléments trouvés sont ajoutés dans la liste \a sub_axl_elements
   */
  private void _SearchSubAxlFiles(XmlElement parent_elem,List<XmlElement> sub_axl_elements)
  {
    if (parent_elem.Name=="include-sub-axl")
      sub_axl_elements.Add(parent_elem);
    foreach(XmlNode node in parent_elem){
      if (node.NodeType!=XmlNodeType.Element)
        continue;
      XmlElement sub_elem = node as XmlElement;
      _SearchSubAxlFiles(sub_elem,sub_axl_elements);
    }
  }

  /*!
   * \brief Importe les fichiers sub-axl.
   *
   * Lit l'attribut '@file-name' de \a element et inclut le fichier
   * correspondant au niveau de \a element. Une fois ceci terminé, \a element
   * doit être supprimé.
   *
   * En retour, construit une liste des éléments importés, qui servira pour
   * l'inclusion récursive d'autres fichiers sub-axl.
   *
   * Lorsqu'on importe plusieurs éléments, il faut faire attention à ce que ce soit
   * dans le même ordre que dans le fichier inclut.
   */
  private void _ImportSubAxlFile(XmlElement element,List<XmlElement> created_children)
  {
    string file_name = Utils.XmlGetAttributeValue(element,"file-name");
    Console.WriteLine("WARNING: Parse sub axl file name='{0}'",file_name);
    if (String.IsNullOrEmpty(file_name))
      throw new InternalErrorException("missing attribut 'file-name' for element <include-sub-axl>");
    string parent_path = Path.GetDirectoryName(m_full_file_name);
    string full_sub_name = Path.Combine(parent_path,file_name);
    XmlElement sub_root = XmlInfo.RootNode(full_sub_name);

    XmlNodeList children = sub_root.ChildNodes;
    // Noeud après lequel il faudra insérer le prochain élément importé
    XmlNode node_to_append = element;
    foreach(XmlNode node in children){
      //if (node.NodeType == XmlNodeType.Element){
      //XmlElement sub_element = node as XmlElement;
      XmlNode import_node = element.OwnerDocument.ImportNode(node,true);
      XmlNode new_node = element.ParentNode.InsertAfter(import_node,node_to_append);
      node_to_append = new_node;
      if (new_node.NodeType==XmlNodeType.Element){
        Console.WriteLine("ADD IMPORTED NODE name='{0}'",new_node.Name);
        created_children.Add((XmlElement)new_node);
      }
    }
  }

  public bool _IsValidUserClass(XmlElement element)
  {
    if (m_user_class==null)
      return true;
    string[] values = Utils.GetElementsValue(element,"userclass");
    foreach(string s in values){
      if (s==m_user_class)
        return true;
    }
    return false;
  }

  }
  /**
   * Fabrique pour les parseurs AXL.
   */
  public class AXLParserFactory
  {
    static object m_lock = new object();
    static bool m_is_init;
    static Stream m_schema_stream;
    static IAXLObjectFactory m_object_factory;

    public static AXLParser CreateParser(string full_file_name,string user_class)
    {
      lock(m_lock){
        if (!m_is_init){
          _Init();
        }
        m_is_init = true;
      }
      return new AXLParser(full_file_name,user_class,m_schema_stream,m_object_factory);
    }

    static void _Init()
    {
      _TryAddAssembly("ArcaneCea.Axl.dll");
      if (m_object_factory==null)
        m_object_factory = new DefaultAXLObjectFactory();
      m_schema_stream = Utils.GetAxlSchemaAsStream();
    }

    /**
     * Essaie d'ouvrir l'assemblee donne par le fichier \a file_name
     * et de trouver une implementation de IAXLObjectFactory.
     * 
     * Pour cela, parcours tous les types de l'assemblee dont
     * le nom se termine par 'AXLObjectFactory' et cree
     * l'instance correspondante. Ce type doit avoir
     * un constructeur par defaut (sans arugments).
     */
    static void _TryAddAssembly(string file_name)
    {
      Assembly a = Assembly.GetExecutingAssembly();
      string assembly_full_path = a.Location;
      string dir = Path.GetDirectoryName(assembly_full_path);

      string full_path = Path.Combine(dir,file_name);
      //Console.WriteLine("TRY ADD ASSEMBLY {0}",file_name);

      try{
        if (!File.Exists(full_path))
          return;
        Assembly b = Assembly.LoadFile(full_path);
        //Assembly.L
        if (b!=null){
          //m_additional_assemblies.Add(b);
          Console.WriteLine("Found custom AXL implementation named '{0}'",full_path);
          foreach(Type t in b.GetTypes()){
            if (t.Name.EndsWith("AXLObjectFactory")){
              Console.WriteLine("Found type named '{0}'",t.FullName);
              ConstructorInfo m = t.GetConstructor(new Type[0]);
              //Console.WriteLine("EXEC_METHOD = {0}",m);
              object o = m.Invoke(new object[0]);
              Console.WriteLine("CREATE Ob={0}",o);
              m_object_factory = o as IAXLObjectFactory;
              Console.WriteLine("Custom factory ={0}",m_object_factory);
              if (m_object_factory!=null)
                return;
            }
          }
        }
      }
      catch(Exception ex){
        Console.WriteLine(String.Format("Can not load assembly '{0}' ex={1}",file_name,ex.Message));
      }
    }
  }

}
