/*---------------------------------------------------------------------------*/
/* CaseOptionsGenerator.cc                                     (C) 2000-2007 */
/*                                                                           */
/* Classe générant le code des classes CaseOptions en C#.                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using System;
using System.IO;
using System.Collections.Generic;
using Integer = System.Int32;

namespace Arcane.Axl
{

  /**
   * Classe générant le code des classes CaseOptions en C#.
   */
  class CSharpCaseOptionsGenerator : CSharpCodeGenerator
  {
    /**
      * Retourne le nom de la classe générée.
      * @return Nom de la classe générée.
      */
    public string getClassName() { return m_class_name; }

    /** Nom du module/service propriétaire du CaseOptions, par exemple hydro, photo... */
    private string m_name;

    /** Nom du namespace (null si aucun) */
    private string m_namespace_name;
    /** Version du fichier axl */
    private double m_version;
    public double Version { get { return m_version; } }
    
    /** Contenu de l'élément XML "options". */
    IList<Option> m_option_info_list;
    /** Différentes traductions de l'élément XML "name". */
    private NameTranslationsInfo m_alternative_names;
    //! \a true si le service n'est pas de type caseoption
    private bool m_not_caseoption;

    public CSharpCaseOptionsGenerator(string path,
                                string output_path,
                                string name,
                                string namespace_macro_name,
                                string namespace_name,
                                double version,
                                IList<Option> option_info_list,
                                NameTranslationsInfo alternative_names,
                                bool not_caseoption)
      : base(path, output_path)
    {
      m_name = name;
      if (!String.IsNullOrEmpty(namespace_macro_name))
        Console.WriteLine("namespace-macro-name is not used in C# generator");

      m_namespace_name = namespace_name;
      m_version = version;
      m_option_info_list = option_info_list;
      m_alternative_names = alternative_names;
      m_not_caseoption = not_caseoption;

      m_class_name = "CaseOptions";
      m_class_name += m_name;
    }

    /** Génération du fichier .h */
    public override void writeFile()
    {
      TextWriter file_stream = new StringWriter();
      WriteInfo(file_stream);
      file_stream.WriteLine("using System;");
      file_stream.WriteLine("#if ARCANE_64BIT");
      file_stream.WriteLine("using Integer = System.Int64;");
      file_stream.WriteLine("#else");
      file_stream.WriteLine("using Integer = System.Int32;");
      file_stream.WriteLine("#endif");
      file_stream.WriteLine("using Real = System.Double;");

      //string class_name_upper_case = m_class_name.ToUpper();
      WriteComments(file_stream);

      OptionTypeCounterVisitor counter = new OptionTypeCounterVisitor();
      foreach( Option opt in m_option_info_list)
        opt.Accept(counter);


      //bool has_namespace = m_namespace_macro_name != null;
      //bool need_type = (counter.NbExtended!=0 || counter.NbEnumeration!=0 || has_namespace);

      if (m_namespace_name != null) {
        file_stream.Write("namespace " + m_namespace_name + " {\n");
      }

      // Ajoute un attribut pour indiquer au compilateur Arcane de ne pas generer de version
      // C++ de cette classe puisqu'il existe un axl.h correspondant
      file_stream.Write("//! Options\n");
      file_stream.Write("[Arcane.Compiler.Directives.CppClassGenerationInfo(NotGenerated=true)]\n");
      file_stream.Write("public class " + m_class_name + "\n");
      file_stream.Write("{\n");
      file_stream.Write('\n');

      CSharpClassDefinitionGenerator cdg = new CSharpClassDefinitionGenerator(file_stream);
      foreach( Option opt in m_option_info_list)
        opt.Accept(cdg);


      file_stream.Write('\n');
      // Destructeur
      {
        string service_lower_case_name = Utils.ToLowerWithDash(m_name);

        if (m_not_caseoption)
          file_stream.Write("  public " + m_class_name
                            + "(" + "Arcane.ICaseMng cm)\n");
        else
          file_stream.Write("  public " + m_class_name + "("
                            + "Arcane.ICaseMng cm,"
                            + "Arcane.ICaseOptions co)\n");
        file_stream.Write("{\n");
        if (m_not_caseoption) {
          file_stream.Write("   m_case_options = new "
                            + "Arcane.CaseOptions(cm,\"" + service_lower_case_name + "\");\n");
        }
        else {
          file_stream.Write("  m_case_options = co;\n");
        }

        CSharpOptionBuilderGenerator obg = new CSharpOptionBuilderGenerator(file_stream,"configList()",
                                                                            "new Arcane.XmlNode()");
        foreach(Option opt in m_option_info_list)
          opt.Accept(obg);

        CSharpBuilderBodyGenerator bbg = new CSharpBuilderBodyGenerator(file_stream);
        foreach(Option opt in m_option_info_list)
          opt.Accept(bbg);

        writeNameTranslations(file_stream,m_alternative_names,null);
        file_stream.Write("}\n");
      }

      file_stream.Write('\n');
      file_stream.Write("  public Arcane.ICaseOptions caseOptions() { return m_case_options; }\n");
      file_stream.Write("  public Arcane.ICaseOptionList configList() { return m_case_options.configList(); }\n");
      file_stream.Write("  public void setCaseModule(Arcane.IModule m) { m_case_options.setCaseModule(m); }\n");
      file_stream.Write("  public void setCaseServiceInfo(Arcane.IServiceInfo si) { m_case_options.setCaseServiceInfo(si); }\n");
      file_stream.Write("  public void addAlternativeNodeName(string lang,string name)\n");
      file_stream.Write("    { m_case_options.addAlternativeNodeName(lang,name); }\n");
      file_stream.Write('\n');
      file_stream.Write('\n');
      //file_stream.Write("   static Arcane.ICaseOptions _createCaseOption("
      //                  + "Arcane.ICaseMng cm,Arcane.ICaseOptions co);\n");
      file_stream.Write('\n');
      file_stream.Write("   Arcane.ICaseOptions m_case_options;\n");
      file_stream.Write('\n');
      file_stream.Write('\n');

      CSharpVariableDefinitionGenerator vdg = new CSharpVariableDefinitionGenerator(file_stream);
      foreach( Option opt in m_option_info_list){
        opt.Accept(vdg);
      }

      file_stream.Write('\n');
      file_stream.Write("}\n");
      WriteComments(file_stream);
      if (m_namespace_name != null)
        file_stream.Write("}\n");
      WriteComments(file_stream);

      // write file
      _writeClassFile(m_name + "_axl.cs", file_stream.ToString(), true);
    }
  }
}
