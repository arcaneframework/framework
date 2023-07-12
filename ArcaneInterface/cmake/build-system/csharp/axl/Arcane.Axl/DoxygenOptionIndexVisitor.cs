/*
 DoxygenOptionIndex.cs (C) 2000-2011

 GÃ©nÃ©ration de la documentation au format Doxygen.
*/
using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Xml;

namespace Arcane.Axl
{
    /// <summary>
    /// Visiteur pour generer la page d'index de toutes les options
    /// </summary>
    public class DoxygenOptionIndexVisitor : IOptionInfoVisitor
    {
        class IndexName : IComparable<IndexName>
        {
            internal string name;
            internal string full_name;
            internal string main_parent_name;
            internal IndexName(string _name, string _full_name, string _main_parent_name)
            {
                name = _name;
                full_name = _full_name;
                main_parent_name = _main_parent_name;
            }
            public int CompareTo(IndexName other)
            {
                int v = name.CompareTo(other.name);
                if (v != 0)
                    return v;
                v = full_name.CompareTo(other.full_name);
                if (v != 0)
                    return v;
                return main_parent_name.CompareTo(other.main_parent_name);
            }
        }
        private TextWriter m_stream;
        SortedList<IndexName, Option> m_options;
        List<char> m_first_chars;
        private string m_lang;
        CodeInfo m_code_info;

        public DoxygenOptionIndexVisitor(CodeInfo code_info, string lang)
        {
            m_lang = lang;
            m_code_info = code_info;
            m_options = new SortedList<IndexName, Option>();
            m_first_chars = new List<char>();
            m_stream = new StringWriter();
        }

        // Genere la page dans le fichier \a file_path
        public void Generate(string file_path)
        {
            Console.WriteLine("GENERATE INDEX");
            TextWriter tw = m_stream;
            //tw.WriteLine("<h2>data file option index</h2>");
            //tw.WriteLine("<ul>");
            char last_char = '\0';
            foreach (KeyValuePair<IndexName, Option> o in m_options)
            {
                IndexName iname = o.Key;
                Option opt = o.Value;
                Option parent = opt.ParentOption;
                string anchor_name = DoxygenDocumentationUtils.AnchorName(o.Value);
                char first_char = Char.ToLower(iname.name[0]);
                if (first_char != last_char)
                {
                    m_first_chars.Add(first_char);
                    if (last_char != '\0')
                        tw.Write("\n</ul>\n");
                    tw.Write("<p>" + first_char + " :</p>\n");
                    tw.Write("\\anchor axldoc_fullindex_letter_" + ((int)first_char).ToString());
                    tw.Write("\n<ul>\n");
                    last_char = first_char;
                }
                tw.Write("<li>");
                tw.Write("\\ref {0} \"{1}\"", anchor_name, iname.name);
                ServiceOrModuleInfo main_info = opt.ServiceOrModule;
                string main_type_name = "service";
                if (main_info.IsModule)
                    main_type_name = "module";
                tw.Write(" (in {0} '{1}'", main_info.GetTranslatedName(m_code_info.Language), main_type_name);
                if (parent != null)
                {
                    string parent_name = parent.GetTranslatedFullName(m_lang);
                    tw.Write(" option &lt;{0}&gt;", parent_name);
                }
                tw.Write(")\n</li>");
            }
            tw.WriteLine("</ul>");
            using (TextWriter file_tw = new StreamWriter(file_path, false, Utils.WriteEncoding))
            {
                file_tw.WriteLine("/*!");
                file_tw.WriteLine("\n\\page axldoc_all_option_index Keywords index\n");
                file_tw.Write("<p>");
                foreach (char c in m_first_chars)
                {
                    file_tw.Write("\\ref axldoc_fullindex_letter_{0} \"{1}\" ", ((int)c).ToString(), c);
                }
                file_tw.Write("</p>");
                file_tw.Write(m_stream.ToString());
                file_tw.WriteLine("*/");
            }
        }

        public void VisitServiceOrModule(ServiceOrModuleInfo info)
        {
            foreach (Option option in info.Options)
            {
                option.Accept(this);
            }
        }

        public void VisitComplex(ComplexOptionInfo option)
        {
            _AddOption(option);
            option.AcceptChildren(this);
        }

        public void VisitEnumeration(EnumerationOptionInfo option)
        {
            _AddOption(option);
        }

        public void VisitExtended(ExtendedOptionInfo option)
        {
            _AddOption(option);
        }

        public void VisitSimple(SimpleOptionInfo option)
        {
            _AddOption(option);
        }

        public void VisitScript(ScriptOptionInfo option)
        {
            _AddOption(option);
        }

        public void VisitServiceInstance(ServiceInstanceOptionInfo option)
        {
            _AddOption(option);
        }

        void _AddOption(Option opt)
        {
            string name = opt.GetTranslatedName(m_lang);
            string full_name = opt.GetTranslatedFullName(m_lang);
            string main_parent_name = opt.ServiceOrModule.GetTranslatedName(m_lang);
            IndexName n = new IndexName(name, full_name, main_parent_name);
            if (m_options.ContainsKey(n))
            {
                Console.WriteLine("ALREAY OPTION name={0} {1}", n.name, n.full_name);
            }
            else
                m_options.Add(n, opt);
        }
    }
}