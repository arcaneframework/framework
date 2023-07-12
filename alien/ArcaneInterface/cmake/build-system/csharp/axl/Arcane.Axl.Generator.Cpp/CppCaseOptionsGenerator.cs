/*---------------------------------------------------------------------------*/
/* CaseOptionsGenerator.cc                                     (C) 2000-2006 */
/*                                                                           */
/* Classe générant le code des classes CaseOptions (fichiers .h et .cc).     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using System;
using System.IO;
using System.Collections.Generic;
using Integer = System.Int32;

namespace Arcane.Axl
{

  /**
   * Classe générant le code des classes CaseOptions (fichiers .h et .cc). 
   */
  class CppCaseOptionsGenerator : CppCodeGenerator
  {
    /**
      * Retourne le nom de la classe générée.
      * @return Nom de la classe générée.
      */
    public string getClassName() { return m_class_name; }

    /** Nom du module/service propriétaire du CaseOptions, par exemple hydro, photo... */
    private string m_name;
    /** Nom de la macro définissant le namespace (null si aucun) */
    private string m_namespace_macro_name;
    /** Nom du namespace (null si aucun) */
    private string m_namespace_name;
    /** Version du fichier axl */
    private double m_version;
    /** Contenu de l'élément XML "options". */
    IList<Option> m_option_info_list;
    /** Différentes traductions de l'élément XML "name". */
    private NameTranslationsInfo m_alternative_names;
    //! \a true si le service n'est pas du type CaseOption
    private bool m_not_caseoption;

    public CppCaseOptionsGenerator(string path,
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
      m_namespace_macro_name = namespace_macro_name;
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
      string class_name_upper_case = m_class_name.ToUpper();
      file_stream.Write("#ifndef ARCANE_" + class_name_upper_case + "_H\n");
      file_stream.Write("#define ARCANE_" + class_name_upper_case + "_H\n");
      WriteComments(file_stream);

      OptionTypeCounterVisitor counter = new OptionTypeCounterVisitor();
      foreach( Option opt in m_option_info_list)
        opt.Accept(counter);

      file_stream.Write('\n');
      file_stream.Write("#include \"arcane/utils/String.h\"\n");
      file_stream.Write('\n');
      file_stream.Write("#include \"arcane/CaseOptions.h\"\n");
      file_stream.Write("#include \"arcane/CaseOptionsMulti.h\"\n");
      file_stream.Write("#include \"arcane/CaseOptionBuildInfo.h\"\n");
      file_stream.Write("#include \"arcane/XmlNodeList.h\"\n");
      if (counter.NbExtended!=0)
        file_stream.Write("#include \"arcane/ItemGroup.h\"\n");
      if (counter.NbServiceInstance != 0)
        file_stream.Write("#include \"arcane/CaseOptionService.h\"\n");
      if (counter.NbScript != 0)
        file_stream.Write("#include \"arcane/CaseOptionScript.h\"\n");


      bool has_namespace = m_namespace_macro_name != null;
      //bool need_type = (counter.getNbExtended()!=0 || counter.getNbEnumeration()!=0
      //                 || counter.getNbServiceInstance()!=0 || has_namespace);
      bool need_type = (counter.NbExtended!=0 || counter.NbEnumeration!=0 || has_namespace);
      if (m_version < 1.0) {
        if (need_type)
          file_stream.Write("#include \"" + m_path
                             + "/Types" + m_name + ".h\"\n");
      }

      if (counter.NbServiceInstance!=0)
        file_stream.Write("#include \"arcane/CaseOptionService.h\"\n");

      if (m_option_info_list.Count > 0) {
        // Ces déclarations doivent être dans le namespace 'Arcane'
        WriteComments(file_stream);
        file_stream.Write("ARCANE_BEGIN_NAMESPACE\n");
        CppExternFunctionGenerator function_gen = new CppExternFunctionGenerator(file_stream);
        foreach( Option opt in m_option_info_list)
          opt.Accept(function_gen);
        file_stream.Write("ARCANE_}\n");
        WriteComments(file_stream);
      }

      if (m_version < 1.0) {
        WriteComments(file_stream);
        file_stream.Write("ARCANE_BEGIN_NAMESPACE\n");
        WriteComments(file_stream);
      }

      if (m_namespace_macro_name != null) {
        WriteComments(file_stream);
        file_stream.Write(m_namespace_macro_name + "_BEGIN_NAMESPACE\n");
        WriteComments(file_stream);
      }
      if (m_namespace_name != null) {
        WriteComments(file_stream);
        file_stream.Write("namespace " + m_namespace_name + " {\n");
        WriteComments(file_stream);
      }

      file_stream.Write("//! Options\n");
      file_stream.Write("class " + m_class_name + "\n");
      if (m_version < 1.0) {
        if (need_type)
          file_stream.Write(": public Types" + m_name + "\n");
      }
      file_stream.Write("{\n");
      file_stream.Write(" public:\n");
      file_stream.Write('\n');

      CppClassDefinitionGenerator cdg = new CppClassDefinitionGenerator(file_stream);
      foreach( Option opt in m_option_info_list)
        opt.Accept(cdg);

      //if (has_interface || always_generate_interface_method){
      foreach( Option opt in m_option_info_list){
        CppInterfaceImplementationGenerator iig = new CppInterfaceImplementationGenerator(file_stream);
        opt.Accept(iig);
      }

      file_stream.Write(" public:\n");
      file_stream.Write('\n');
      // Destructeur
      {
        string service_lower_case_name = Utils.ToLowerWithDash(m_name);
        if(GlobalContext.Instance.Verbose){ 
          Console.WriteLine("SERVICE NAME: {0} lower={1}",m_name,service_lower_case_name);
        }
        if (m_not_caseoption)
          file_stream.Write("  " + m_class_name
                            + "(" + "Arcane::ICaseMng* cm)\n");
        else
          file_stream.Write("  " + m_class_name + "("
                            + "Arcane::ICaseMng* cm,"
                            + "Arcane::ICaseOptions* co)\n");

        if (m_not_caseoption) {
          file_stream.Write(": m_case_options(new "
                            + "Arcane::CaseOptions(cm,\"" + service_lower_case_name + "\"))\n");
        }
        else {
          file_stream.Write(": m_case_options(co)\n");
        }

        CppOptionBuilderGenerator obg = new CppOptionBuilderGenerator(file_stream, "configList()", "Arcane::XmlNode(0)");
        foreach(Option opt in m_option_info_list)
          opt.Accept(obg);

        file_stream.Write("{\n");
        CppBuilderBodyGenerator bbg = new CppBuilderBodyGenerator(file_stream);
        foreach(Option opt in m_option_info_list)
          opt.Accept(bbg);

        writeNameTranslations(file_stream,m_alternative_names,null);
        file_stream.Write("}\n");
      }

      // Ajoute un destructeur virtuel car il peut y avoir des méthodes virtuelles lors de la génération
      {
        file_stream.Write("  virtual ~" + m_class_name + "(){}\n");
      }
      file_stream.Write('\n');
      file_stream.Write(" public:\n");
      file_stream.Write('\n');
      file_stream.Write("  Arcane::ICaseOptions* caseOptions() const { return m_case_options; }\n");
      file_stream.Write("  Arcane::ICaseOptionList* configList() const { return m_case_options->configList(); }\n");
      file_stream.Write("  void setCaseModule(Arcane::IModule* m) { m_case_options->setCaseModule(m); }\n");
      file_stream.Write("  void setCaseServiceInfo(Arcane::IServiceInfo* si) { m_case_options->setCaseServiceInfo(si); }\n");
      file_stream.Write("  void addAlternativeNodeName(const Arcane::String& lang,const Arcane::String& name)\n");
      file_stream.Write("    { m_case_options->addAlternativeNodeName(lang,name); }\n");
      file_stream.Write('\n');
      file_stream.Write(" private:\n");
      file_stream.Write('\n');
      file_stream.Write("   static Arcane::ICaseOptions* _createCaseOption("
                        + "Arcane::ICaseMng* cm,Arcane::ICaseOptions* co);\n");
      file_stream.Write('\n');
      file_stream.Write("   Arcane::ICaseOptions* m_case_options;\n");
      file_stream.Write('\n');
      file_stream.Write(" public:\n");
      file_stream.Write('\n');

      CppVariableDefinitionGenerator vdg = new CppVariableDefinitionGenerator(file_stream);
      foreach( Option opt in m_option_info_list)
        opt.Accept(vdg);

      file_stream.Write('\n');
      file_stream.Write("};\n");
      WriteComments(file_stream);
      if (m_namespace_macro_name != null)
        file_stream.Write(m_namespace_macro_name + "_}\n");
      if (m_namespace_name != null)
        file_stream.Write("}\n");
      if (m_version < 1.0) {
        file_stream.Write("ARCANE_}\n");
      }
      WriteComments(file_stream);
      file_stream.Write("#endif\n");

      // write file
      _writeClassFile(m_name + "_axl.h", file_stream.ToString(), true);
    }
  }
}
