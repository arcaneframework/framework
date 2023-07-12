/*---------------------------------------------------------------------------*/
/* ModuleBaseGenerator.cc                                      (C) 2000-2006 */
/*                                                                           */
/* Classe générant le code de la classe de base d'un module.                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
using System.IO;
using Integer = System.Int32;

namespace Arcane.Axl
{
  /**
   * Classe générant le code de la classe de base d'un module. 
   */
  public class CppModuleBaseGenerator : CppCodeGenerator
  {
    /** Objet stockant les informations de l'élément XML "module". */
    private ModuleInfo m_info;
    /** Générateur des fichiers "CaseOptions.h" et "CaseOptions.cc". */
    private CppCaseOptionsGenerator m_co_generator;

    public CppModuleBaseGenerator(string path,
                               string output_path,
                               ModuleInfo info)
      : base(path, output_path)
    {
      m_info = info;
      m_class_name = "Arcane" + m_info.Name + "Object";
      m_co_generator = new CppCaseOptionsGenerator(path,
                                                m_output_path,
                                                m_info.Name,
                                                m_info.NamespaceMacroName,
                                                m_info.NamespaceName,
                                                m_info.Version,
                                                m_info.Options,
                                                m_info.AlternativeNames,
                                                true);
    }

    public override void writeFile()
    {
      double version = m_info.Version;

      _writeClassFile(m_info.Name + "_axl.h", null, false);

      // generation du fichier CaseOptions
      if (m_co_generator!=null)
        m_co_generator.writeFile();

      TextWriter file_stream = new StringWriter();
      WriteInfo(file_stream);
      string class_name_upper_case = m_class_name.ToUpper();
      file_stream.Write("#ifndef ARCANE_" + class_name_upper_case + "_H\n");
      file_stream.Write("#define ARCANE_" + class_name_upper_case + "_H\n");
      WriteComments(file_stream);

      file_stream.Write("#include \"arcane/VariableTypes.h\"\n");
      file_stream.Write("#include \"arcane/EntryPoint.h\"\n");
      file_stream.Write("#include \"arcane/ISubDomain.h\"\n");
      file_stream.Write("#include \"arcane/ModuleBuildInfo.h\"\n");
      file_stream.Write("#include \"arcane/ModuleFactory.h\"\n");
      file_stream.Write("#include \"arcane/ServiceRegisterer.h\"\n");
      file_stream.Write("#include \"arcane/BasicModule.h\"\n");
      file_stream.Write("#include \"arcane/ServiceInfo.h\"\n");

      bool has_namespace = m_info.NamespaceMacroName != null || m_info.NamespaceName != null;

      if (version < 1.0) {
        if (has_namespace)
          file_stream.Write("#include \"" + m_path + "/Types" + m_info.Name + ".h\"\n");
      }

      WriteComments(file_stream);
      file_stream.Write("ARCANE_BEGIN_NAMESPACE\n");
      file_stream.Write("class ISubDomain;\n");
      file_stream.Write("class IModule;\n");
      file_stream.Write("ARCANE_END_NAMESPACE\n");
      WriteComments(file_stream);

      if (version < 1.0) {
        WriteComments(file_stream);
        file_stream.Write("ARCANE_BEGIN_NAMESPACE\n");
        WriteComments(file_stream);
      }

      if (has_namespace) {
        if (m_info.NamespaceMacroName != null)
          file_stream.Write(m_info.NamespaceMacroName + "_BEGIN_NAMESPACE\n");
        else
          file_stream.Write("namespace " + m_info.NamespaceName + "{\n");
        WriteComments(file_stream);
      }

      if (m_co_generator!=null) {
        WriteComments(file_stream);
        file_stream.Write("class " + m_co_generator.getClassName() + ";\n");
        WriteComments(file_stream);
      }

      file_stream.Write("//! Generation de la classe de base du Module\n");
      file_stream.Write("class " + m_class_name + "\n");
      file_stream.Write(": public " + m_info.ParentName + "\n");
      file_stream.Write("{\n");
      file_stream.Write(" public:\n");
      file_stream.Write("  static Arcane::IServiceInfo* serviceInfoCreateFunction(const Arcane::String& name)\n");
      file_stream.Write("  {\n");
      file_stream.Write("    Arcane::IServiceInfo* si = new Arcane::ServiceInfo(name,Arcane::VersionInfo(\"0.0\"),\n");
      file_stream.Write("                                       Arcane::IServiceInfo::Dim2|Arcane::IServiceInfo::Dim3);\n");
      file_stream.Write("    return si;\n");
      file_stream.Write("  }\n");
      file_stream.Write(" public:\n");

      // ructeur
      {
        file_stream.Write("  " + m_class_name + "(const " + arcane_scope + "ModuleBuildInfo& mb)\n");
        file_stream.Write(": " + m_info.ParentName + "(mb)\n");

        if (m_co_generator!=null)
          file_stream.Write(", m_options(0)\n");

        _WriteVariablesConstructor(m_info.VariableInfoList,"this",file_stream);

        file_stream.Write("{\n");
        if (m_co_generator!=null) {
          file_stream.Write("  m_options = new " + m_co_generator.getClassName()
                            + "(mb.m_sub_domain->caseMng());\n");
          file_stream.Write("  m_options->setCaseModule(this);\n");
        }

        // points d'entrée
        foreach(EntryPointInfo epi in m_info.EntryPointInfoList){
          file_stream.Write("  addEntryPoint(this");
          file_stream.Write(",\"");
          file_stream.Write(epi.Name + "\",\n");
          file_stream.Write("                &" + m_class_name + "::");
          file_stream.Write(epi.MethodeName);
          file_stream.Write(",\n                Arcane::IEntryPoint::W");
          file_stream.Write(ToClassName(epi.Where));
          file_stream.Write(",\n                Arcane::IEntryPoint::");
          switch (epi.Property) {
            case Property.PNone:
              file_stream.Write("PNone);\n");
              break;
            case Property.PAutoLoadBegin:
              file_stream.Write("PAutoLoadBegin);\n");
              break;
            case Property.PAutoLoadEnd:
              file_stream.Write("PAutoLoadEnd);\n");
              break;
          }
        }
        file_stream.Write("}\n");

      }

      // Destructeur
      {
        file_stream.Write("  virtual ~" + m_class_name + "()\n\n");
        file_stream.Write("{\n");
        if (m_co_generator!=null)
          file_stream.Write("  delete m_options;\n");
        file_stream.Write("}\n");
      }
			
	  // Points d'entrée comme méthodes abstraites
      file_stream.Write(" public:\n");
      file_stream.Write("  // points d'entrée\n");

      foreach( EntryPointInfo epi in m_info.EntryPointInfoList){
        file_stream.Write("  virtual void " + epi.MethodeName + "() = 0;");
        file_stream.Write(" // " + epi.Name + "\n");
      }
      file_stream.Write("\n");

      if (m_co_generator!=null) {
        file_stream.Write(" public:\n");
        file_stream.Write("  //! Options du jeu de données du module\n");
        file_stream.Write("  " + m_co_generator.getClassName() + "* options() const { return m_options; }\n");
        file_stream.Write("\n");
        file_stream.Write(" private:\n");
        file_stream.Write("  //! Options du jeu de données du module\n");
        file_stream.Write("  " + m_co_generator.getClassName() + "* m_options;\n\n");
      }

      _WriteVariablesDeclaration(m_info.VariableInfoList,file_stream);

      file_stream.Write("};\n");
      WriteComments(file_stream);

      // macro de creation du module
      //string module_upper_case = m_info.m_name.clone();
      string module_upper_case = m_info.Name;
      module_upper_case = module_upper_case.ToUpper();
      file_stream.Write("#define ARCANE_REGISTER_MODULE_" + module_upper_case);
      file_stream.Write("(class_name) \\\n");
      file_stream.Write("extern \"C++\" ARCANE_EXPORT {0}IModuleFactoryInfo* \\\n",arcane_scope);
      file_stream.Write("arcaneCreateModuleFactory" + m_info.Name);
      file_stream.Write("("+arcane_scope+"IServiceInfo* si) \\\n");
      file_stream.Write("{ \\\n");
      string is_autoload = (m_info.IsAutoload) ? "true" : "false";
      file_stream.Write("  return new {0}ModuleFactory(si,new {0}ModuleFactory2T< class_name > (\"{1}\"),{2}); \\\n",
        arcane_scope,m_info.Name,is_autoload);
      //file_stream.Write("); \\\n");
      file_stream.Write("} \\\n");
      file_stream.Write(arcane_scope+"ServiceRegisterer ARCANE_EXPORT globalModuleRegisterer");
      file_stream.Write(m_info.Name + "(arcaneCreateModuleFactory" + m_info.Name);
      //file_stream.Write(",&" + m_class_name + "::serviceInfoCreateFunction");
      file_stream.Write(",\"" + m_info.Name + "\")\n");

      WriteComments(file_stream);

      if (has_namespace) {
        if (m_info.NamespaceMacroName != null)
          file_stream.Write(m_info.NamespaceMacroName + "_END_NAMESPACE\n");
        else
          file_stream.Write("}\n");
      }
      if (version < 1.0) {
        file_stream.Write("ARCANE_END_NAMESPACE\n");
      }
      WriteComments(file_stream);
      file_stream.Write("#endif\n");

      // write file
      _writeClassFile(m_info.Name + "_axl.h", file_stream.ToString(), true);
    }
  }
}
