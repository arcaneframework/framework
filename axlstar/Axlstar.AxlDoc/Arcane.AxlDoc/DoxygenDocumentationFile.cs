//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DoxygenDocumentationFile.cs                                 (C) 2000-2019 */
/*                                                                           */
/* Génération de la documentation au format Doxygen.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
using System;
using System.IO;
using Arcane.Axl;

namespace Arcane.AxlDoc
{
  public class DoxygenDocumentationFile
  {
    TextWriter m_full_stream = new StringWriter ();
    TextWriter m_brief_stream = new StringWriter ();
    TextWriter m_main_desc_stream = new StringWriter ();
    string m_output_path;

    public string OutputPath { get { return m_output_path; } }

    string m_page_name;
    const string left_mark = "'"; // left and 
    const string right_mark = "'";//   right marks around module and service names
    string m_page_title;
    string m_sub_title;

    public string SubTitle { get { return m_sub_title; } set { m_sub_title = value; } }

    static int m_id = 0;
    string m_language;

    public DoxygenDocumentationFile (ServiceOrModuleInfo base_info, string output_path, string language)
    {
      m_language = language;
      m_output_path = output_path;
      if (base_info.IsModule) {
        m_page_name = "axldoc_module_" + base_info.FileBaseName;

        m_page_title = "Module " + left_mark + base_info.GetTranslatedName(m_language) + right_mark;
        Console.WriteLine ("New module pageref={0}", m_page_name);
      } else {
        m_page_name = "axldoc_service_" + base_info.FileBaseName;
        m_page_title = "Service " + left_mark + base_info.Name + right_mark;
        Console.WriteLine ("New service pageref={0}", m_page_name);
      }
      ++m_id;
    }

    public DoxygenDocumentationFile (ComplexOptionInfo option_info, string output_path, string language)
    {
      m_language = language;
      m_output_path = output_path;
      m_page_name = DoxygenDocumentationUtils.AnchorName (option_info);
      ServiceOrModuleInfo base_info = option_info.ServiceOrModule;

      m_page_title = left_mark + base_info.GetTranslatedName (m_language) + right_mark +
        " option <" + option_info.GetTranslatedFullName (m_language) + ">";
      ++m_id;
    }

    public TextWriter FullDescStream { get { return m_full_stream; } }

    public TextWriter BriefDescStream { get { return m_brief_stream; } }

    public TextWriter MainDescStream { get { return m_main_desc_stream; } }

    public void Write ()
    {
      _WriteMarkdown ();
    }

    void _WriteMarkdown ()
    {
      string file_name = String.Format ("{0}.md", m_page_name);
      string full_name = Path.Combine (m_output_path, file_name);

      string snippet_name = String.Format ("snippet_{0}", m_page_name);
      string snippet_file_name = String.Format ("{0}.md", snippet_name);
      string snippet_full_name = Path.Combine (m_output_path, snippet_file_name);

      using (TextWriter tw = new StreamWriter (full_name, false, Utils.WriteEncoding)) {
        tw.WriteLine ("# {1} {{#{0}}}\n", m_page_name, m_page_title);
        string main_desc_string = m_main_desc_stream.ToString ();
        if (!string.IsNullOrEmpty (main_desc_string)) {
          tw.Write (main_desc_string);
        }
        if (!String.IsNullOrEmpty(m_sub_title))
          tw.WriteLine(m_sub_title);
        if (!string.IsNullOrEmpty (m_brief_stream.ToString ())) {
          if(m_language == "fr"){
            tw.WriteLine ("\n## Liste des options\n");
          }
          else{
            tw.WriteLine ("\n## Summary of options\n");
          }

          // if(false){
          //   tw.WriteLine ("\\snippet{{doc}} {0} {1}", snippet_file_name, snippet_name);
          // }
          // else{
          tw.WriteLine("<ul>");
          tw.Write (m_brief_stream.ToString ());
          tw.WriteLine ("</ul>");
          // }
        }
        if (!string.IsNullOrEmpty (m_full_stream.ToString ())) {
          if(m_language == "fr"){
            tw.WriteLine("\n## Documentation des options\n");
          }
          else{
            tw.WriteLine("\n## Detailed list of options\n");
          }
          tw.Write(m_full_stream.ToString());
        }
      }

      // Génération des fichiers snippets avec les options des services/modules.
      // Attention : Pour que ça fonctionne, il faut ajouter le dossier de sortie dans
      // la partie "EXAMPLE_PATH" du .doxyfile.
      using (TextWriter tw = new StreamWriter (snippet_full_name, false, Utils.WriteEncoding)) {
        tw.WriteLine ("//![{0}]", snippet_name);
        tw.WriteLine("<ul>");
        tw.Write (m_brief_stream.ToString ());
        tw.WriteLine ("</ul>");
        tw.WriteLine ("//![{0}]", snippet_name);
      }
    }

    public string PageName ()
    {
      return m_page_name;
    }
  }
}
