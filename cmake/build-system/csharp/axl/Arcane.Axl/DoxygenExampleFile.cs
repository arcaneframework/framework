/*---------------------------------------------------------------------------*/
/* DoxygenExampleFile.cs                                       (C) 2000-2007 */
/*                                                                           */
/* Génération de la documentation au format Doxygen.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Xml;

namespace Arcane.Axl
{
  public class DoxygenExampleFile
  {
    static TextWriter m_aliases_stream = new StringWriter();
    static string m_output_path;
    TextWriter m_example_stream = new StringWriter();
    string m_base_name;

    public DoxygenExampleFile(ServiceOrModuleInfo base_info, params string[] suffixes)
    {
      if (base_info.IsModule) {
        m_base_name = "axldoc_module_" + base_info.FileBaseName;
      }
      else {
        m_base_name = "axldoc_service_" + base_info.FileBaseName;
      }
      foreach (string suffixe in suffixes) m_base_name += suffixe;
    }

    public DoxygenExampleFile(Option option)
    {
      m_base_name = DoxygenDocumentationUtils.AnchorName(option);
    }

    public TextWriter ExampleStream { get { return m_example_stream; } }
    static public TextWriter AliasesStream { get { return m_aliases_stream; } }
    static public string OutputPath { get { return m_output_path   ; }
                                      set { m_output_path = value  ; } }
    public string BaseName { get { return m_base_name; } }

    static public string getFilePath(string base_name)
    {
      string file_name = String.Format("examples/out_{0}_ex.dox", base_name);
      return Path.Combine(m_output_path, file_name);
    }      
    
    static private void _Write(string file_path, TextWriter stream)
    {
      using (TextWriter tw = new StreamWriter(file_path,false,Utils.WriteEncoding)) {
        tw.Write(stream.ToString());
      }      
    }

    public void Write()
    {
      string file_path = getFilePath(m_base_name);
      _Write(file_path, m_example_stream);
    }

    static public void WriteAliases()
    {
      string file_path = Path.Combine(m_output_path,
                                      "alias/out_axldoc_aliases.doxyfile");
      _Write(file_path, m_aliases_stream);
    }
  }
}
