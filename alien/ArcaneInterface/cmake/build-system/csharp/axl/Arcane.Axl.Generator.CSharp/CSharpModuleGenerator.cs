/*---------------------------------------------------------------------------*/
/* ModuleBaseGenerator.cc                                      (C) 2000-2007 */
/*                                                                           */
/* Classe générant le code de la classe de base d'un module.                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
using System.IO;
using Integer = System.Int32;

namespace Arcane.Axl
{
  /**
   * Classe générant le code de la classe C# d'un module. 
   */
  public class CSharpModuleGenerator : CSharpCodeGenerator
  {
    /** Objet stockant les informations de l'élément XML "module". */
    private ModuleInfo m_info;
    /** Générateur des fichiers "CaseOptions.h" et "CaseOptions.cc". */
    private CSharpCaseOptionsGenerator m_co_generator;

    public CSharpModuleGenerator(string path,
                                 string output_path,
                                 ModuleInfo info)
      : base(path, output_path)
    {
      m_info = info;
      m_class_name = "Arcane" + m_info.Name + "Object";
      m_co_generator = new CSharpCaseOptionsGenerator(path,
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
      //double version = m_info.m_version;

      _writeClassFile(m_info.Name + "_axl.cs", null, false);

      // generation du fichier CaseOptions
      if (m_co_generator!=null)
        m_co_generator.writeFile();

      TextWriter file_stream = new StringWriter();
      WriteInfo(file_stream);
      WriteComments(file_stream);

      bool has_namespace = !  string.IsNullOrEmpty(m_info.NamespaceName);

      if (has_namespace) {
        file_stream.Write("namespace " + m_info.NamespaceName + "{\n");
        WriteComments(file_stream);
      }

      // Ajoute un attribut pour indiquer au compilateur Arcane de ne pas generer de version
      // C++ de cette classe puisqu'il existe un axl.h correspondant
      file_stream.Write("[Arcane.Compiler.Directives.CppClassGenerationInfo(NotGenerated=true)]\n");
      file_stream.Write("public abstract class " + m_class_name + "\n");
      file_stream.Write(": " + ConvertNamespace(m_info.ParentName) + "\n");
      file_stream.Write("{\n");

      // Constructeur
      {
        file_stream.Write(" public " + m_class_name + "(" + arcane_scope + "ModuleBuildInfo mb)\n");
        file_stream.Write(": base(mb)\n");
        file_stream.Write("{\n");

        if (m_co_generator!=null)
          file_stream.Write(" m_options = null;\n");

        _WriteVariablesConstructor(m_info.VariableInfoList,"this",file_stream);

        if (m_co_generator!=null) {
          file_stream.Write("  m_options = new " + m_co_generator.getClassName()
                            + "(mb.m_sub_domain.caseMng());\n");
          file_stream.Write("  m_options.setCaseModule(this);\n");
        }

        // points d'entrée
        //for (Integer i = 0; i < m_info.EntryPointInfoList; i++) {
        //EntryPointInfo epi = m_info.m_entry_point_info_list[i];
        foreach(EntryPointInfo epi in m_info.EntryPointInfoList){
          file_stream.Write("  _AddEntryPoint(");
          file_stream.Write("\"");
          file_stream.Write(epi.Name + "\",\n");
          file_stream.Write("                this.");
          file_stream.Write(epi.UpperMethodName);
          file_stream.Write(",\n                Arcane.IEntryPoint.W");
          file_stream.Write(ToClassName(epi.Where));
          file_stream.Write(",\n                Arcane.IEntryPoint.");
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

	  // Points d'entrée comme méthodes abstraites
      file_stream.Write("  // points d'entrée\n");
      foreach( EntryPointInfo epi in m_info.EntryPointInfoList){
        file_stream.Write("  public abstract void " + epi.UpperMethodName + "();");
        file_stream.Write(" // " + epi.Name + "\n");
      }
      file_stream.Write("\n");

      if (m_co_generator!=null) {
        file_stream.Write("  //! Options du jeu de données du module\n");
        file_stream.Write("  public " + m_co_generator.getClassName() + " Options { get { return m_options; } }\n");
        file_stream.Write("\n");
        file_stream.Write("  //! Options du jeu de données du module\n");
        file_stream.Write("  private " + m_co_generator.getClassName() + " m_options;\n\n");
      }

      _WriteVariablesDeclaration(m_info.VariableInfoList,file_stream);
      file_stream.Write("};\n");
      WriteComments(file_stream);

      if (has_namespace) {
        file_stream.Write("}\n");
      }
      WriteComments(file_stream);

      // write file
      _writeClassFile(m_info.Name + "_axl.cs", file_stream.ToString(), true);
    }
  }
}
