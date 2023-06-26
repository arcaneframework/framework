/*---------------------------------------------------------------------------*/
/* CodeGenerator.cc                                            (C) 2000-2008 */
/*                                                                           */
/* Classe de base des classes de génération de code en C#.                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace Arcane.Axl
{
  /**
   * Classe de base des classes de génération de code en C#. 
   */
  public abstract class CSharpCodeGenerator : CodeGenerator
  {
    protected static string arcane_scope = "Arcane.";

    protected CSharpCodeGenerator(string path,string output_path) : base(path,output_path)
    {
    }

    /** Ecrit les traductions */
    public static void writeNameTranslations(TextWriter ts,
                                             NameTranslationsInfo info,
                                             string variable)
    {
      foreach (DictionaryEntry de in info.m_names) {
        string key = (string)de.Key;
        string value = (string)de.Value;
        ts.Write("  ");
        if (variable!=null && variable!="")
          ts.Write("m_"+variable + ".");
        ts.Write("addAlternativeNodeName(\""
                 + key + "\",\"" + value + "\");\n"); ;
      }
    }

    static public string ToCSharpType(string name)
    {
      return name.Replace("::",".");
    }

    protected void _writeClassFile(string file_name, string content, bool append)
    {
      string full_file_name = Path.Combine(m_output_path, file_name);
      TextWriter ts = null;
      if (append)
        ts = File.AppendText(full_file_name);
      else
        ts = File.CreateText(full_file_name);

      ts.Write(content);
      ts.Close();
    }

    private void _WriteVariableProperty(TextWriter writer,bool value,
                                        string name,ref bool is_first_attribute)
    {
      if (value){
        if (is_first_attribute)
          writer.Write(", ");
        else
          writer.Write("|");
        is_first_attribute = false;
        writer.Write(arcane_scope + "IVariable." +name);
      }
    }

    private string _VariableTypeName(VariableInfo var_info)
    {
      string var_class_name;
      string var_namespace_name;
      var_info.GetGeneratedClassName(out var_class_name,out var_namespace_name);
      var_namespace_name = ConvertNamespace(var_namespace_name);
      return var_namespace_name + "." + var_class_name;
    }

    protected void _WriteVariablesConstructor(IList<VariableInfo> variables,
                                              string first_argument,
                                              TextWriter writer)
    {
      foreach(VariableInfo variable_info in variables){
        bool is_first_attribute = true;
        writer.Write(" m_" + variable_info.FieldName);
        writer.Write(" = new " + _VariableTypeName(variable_info) + "(new "+arcane_scope+"VariableBuildInfo("+first_argument+", \"");
        writer.Write(variable_info.Name + "\"");
        _WriteVariableProperty(writer,variable_info.IsNoDump,
                               "PNoDump",ref is_first_attribute);
        _WriteVariableProperty(writer,variable_info.IsNoNeedSync,
                               "PNoNeedSync",ref is_first_attribute);
        _WriteVariableProperty(writer,variable_info.IsExecutionDepend,
                               "PExecutionDepend",ref is_first_attribute);
        _WriteVariableProperty(writer,variable_info.IsSubDomainDepend,
                               "PSubDomainDepend",ref is_first_attribute);
        _WriteVariableProperty(writer,variable_info.IsSubDomainPrivate,
                               "PSubDomainPrivate",ref is_first_attribute);
        _WriteVariableProperty(writer,variable_info.IsNoRestore,
                               "PNoRestore",ref is_first_attribute);
        writer.Write("));\n");
      }
    }

    protected void _WriteVariablesDeclaration(IList<VariableInfo> variables,
                                              TextWriter writer)
    {
      writer.Write("  // variables\n");
      foreach(VariableInfo variable_info in variables){
        writer.Write("  protected readonly " + _VariableTypeName(variable_info));
        writer.Write(" m_" + variable_info.FieldName + ";\n");
      }
    }
    
    static public string ConvertNamespace(string name)
    {
      return name.Replace("::",".");
    }
 
  }
}
