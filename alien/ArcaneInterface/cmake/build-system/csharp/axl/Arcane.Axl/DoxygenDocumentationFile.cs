/*---------------------------------------------------------------------------*/
/* DoxygenDocumentationFile.cs                                 (C) 2000-2007 */
/*                                                                           */
/* GÃ©nÃ©ration de la documentation au format Doxygen.                         */
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
    public class DoxygenDocumentationFile
    {
        TextWriter m_full_stream = new StringWriter();
        TextWriter m_brief_stream = new StringWriter();
        TextWriter m_main_desc_stream = new StringWriter();

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

        public DoxygenDocumentationFile(ServiceOrModuleInfo base_info, string output_path, string language)
        {
            m_language = language;
            m_output_path = output_path;
            if (base_info.IsModule)
            {
                m_page_name = "axldoc_module_" + base_info.FileBaseName;

                //        m_page_title = "module \\<" + base_info.GetTranslatedName("fr") + "\\>";
                m_page_title = left_mark + base_info.GetTranslatedName(m_language) + right_mark + " Module";
                Console.WriteLine("New module pageref={0}", m_page_name);
            }
            else
            {
                m_page_name = "axldoc_service_" + base_info.FileBaseName;
                m_page_title = left_mark + base_info.Name + right_mark + " service";
                Console.WriteLine("New service pageref={0}", m_page_name);
            }
            ++m_id;
        }

        public DoxygenDocumentationFile(ComplexOptionInfo option_info, string output_path, string language)
        {
            m_language = language;
            m_output_path = output_path;
            m_page_name = DoxygenDocumentationUtils.AnchorName(option_info);
            ServiceOrModuleInfo base_info = option_info.ServiceOrModule;

            m_page_title = " " + left_mark + base_info.GetTranslatedName(m_language) + right_mark +
            " option <" + option_info.GetTranslatedFullName(m_language) + ">\n";
            if (base_info.IsModule)
                m_page_title += " module";
            else
                m_page_title += " service";

            ++m_id;
        }

        public TextWriter FullDescStream { get { return m_full_stream; } }
        public TextWriter BriefDescStream { get { return m_brief_stream; } }
        public TextWriter MainDescStream { get { return m_main_desc_stream; } }

        public void Write()
        {
            string file_name = String.Format("out_{0}.dox", m_page_name);
            string full_name = Path.Combine(m_output_path, file_name);
            using (TextWriter tw = new StreamWriter(full_name, false, Utils.WriteEncoding))
            {
                tw.WriteLine("/*!");
                tw.WriteLine("\n\\page {0} {1}\n", m_page_name, m_page_title);
                if (!String.IsNullOrEmpty(m_sub_title))
                    tw.WriteLine(m_sub_title);
                string main_desc_string = m_main_desc_stream.ToString();
                if (!string.IsNullOrEmpty(main_desc_string))
                {
                    tw.Write(main_desc_string);
                }
                if (!string.IsNullOrEmpty(m_brief_stream.ToString()))
                {
                    tw.WriteLine("<ul>");
                    tw.Write(m_brief_stream.ToString());
                    tw.WriteLine("</ul>");
                }
                tw.Write(m_full_stream.ToString());
                tw.WriteLine("*/");
            }
        }
        public string PageName()
        {
            return m_page_name;
        }
    }
}