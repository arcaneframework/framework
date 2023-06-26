/*---------------------------------------------------------------------------*/
/* DoxygenDocumentationGenerator.cc                            (C) 2000-2007 */
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
    public class DoxygenDocumentationGenerator
    {
        private CodeInfo m_code_info;
        private string m_output_path;
        private string m_user_class;
        private bool m_do_examples;
        private StreamWriter m_dico_writer = null;

        private DoxygenOptionIndexVisitor m_option_index_visitor;

        public DoxygenDocumentationGenerator(string output_path, string user_class,
                                             CodeInfo code_info, bool do_examples, bool do_dico)
        {
            m_output_path = output_path;
            m_user_class = user_class;
            m_code_info = code_info;
            m_do_examples = do_examples;

            if (do_dico)
            {
                FileStream dico_file = new FileStream("dicosearch", FileMode.OpenOrCreate, FileAccess.Write);
                m_dico_writer = new StreamWriter(dico_file);
            }
        }

        public void Generate(List<string> axl_files)
        {
            Console.WriteLine("CREATE DIRECTORY {0}", m_output_path);

            Directory.CreateDirectory(m_output_path);

            SortedList<string, ModuleInfo> modules = new SortedList<string, ModuleInfo>();
            SortedList<string, ServiceInfo> services = new SortedList<string, ServiceInfo>();
            string out_language = m_code_info.Language;
            foreach (string axl_file in axl_files)
            {
                AXLParser parser = AXLParserFactory.CreateParser(axl_file, m_user_class);
                parser.ParseAXLFileForDocumentation();
                ServiceOrModuleInfo info = parser.ServiceOrModule;
                ModuleInfo module_info = parser.Module;
                string service_or_module_name;
                string axl_base_name = Path.GetFileNameWithoutExtension(axl_file);
                Console.WriteLine("** ** ** AXL BASE NAME {0} userclass={1}", axl_base_name, m_user_class);
                if (!_IsValidClass(info.UserClasses))
                {
                    Console.WriteLine("BAD CLASS. REMOVE {0}", axl_base_name);
                    continue;
                }
                if (module_info == null)
                {
                    ServiceInfo service_info = parser.Service;
                    service_or_module_name = service_info.GetTranslatedName(out_language) + service_info.FileBaseName;
                    services[service_or_module_name] = service_info;
                }
                else
                {
                    service_or_module_name = module_info.GetTranslatedName(out_language) + module_info.FileBaseName;
                    modules[service_or_module_name] = module_info;
                }
            }

            m_option_index_visitor = new DoxygenOptionIndexVisitor(m_code_info, out_language);

            if (m_do_examples)
            {
                DoxygenExampleFile.OutputPath = m_output_path;
                Directory.CreateDirectory(DoxygenExampleFile.OutputPath);
            }

            // On traite les services avant les modules
            foreach (ServiceOrModuleInfo service_info in services.Values)
            {
                _generateServiceOrModule(service_info);
            }
            foreach (ServiceOrModuleInfo module_info in modules.Values)
            {
                _generateServiceOrModule(module_info);
            }

            if (m_do_examples)
            {
                DoxygenExampleFile.WriteAliases();
            }
            m_option_index_visitor.Generate(Path.Combine(m_output_path, "_full_index.dox"));
            _WriteCaseMainPage(modules.Values, services.Values);

            if (m_dico_writer != null)
            {
                m_dico_writer.Close();
            }

        }

        private void _generateServiceOrModule(ServiceOrModuleInfo info)
        {
            DoxygenDocumentationFile doc_file = new DoxygenDocumentationFile(info, m_output_path, m_code_info.Language);
            DoxygenDocumentationVisitor dbdv = null;

            if (m_dico_writer != null)
            {
                dbdv = new DoxygenDocumentationVisitor(doc_file, m_code_info, m_dico_writer);
            }
            else
            {
                dbdv = new DoxygenDocumentationVisitor(doc_file, m_code_info);
            }

            if (m_user_class == null)
                dbdv.PrintUserClass = true;
            dbdv.VisitServiceOrModule(info);
            doc_file.Write();
            if (m_do_examples)
            {
                DoxygenExampleFile example_file = new DoxygenExampleFile(info);
                DoxygenExampleVisitor dev = new DoxygenExampleVisitor(example_file, m_code_info, "fr");
                dev.VisitServiceOrModule(info);
                example_file.Write();
            }
            m_option_index_visitor.VisitServiceOrModule(info);
        }

        private bool _IsValidClass(string[] values)
        {
            if (m_user_class == null)
                return true;
            foreach (string s in values)
            {
                if (s == m_user_class)
                    return true;
            }
            return false;
        }
        /*!
         * \brief Génère le chemin relative d'un service ou d'un module.
         */
        string _GetServiceOrModulePath(ServiceOrModuleInfo smi)
        {
            // Supprime le nom du service ou module du chemin.
            // Par exemple, si le service s'appelle Toto et que
            // le chemin est Toto_titi_tata, retourne titi_tata.
            string name = smi.Name;
            string rel_path = smi.FileBaseName;
            int r = rel_path.IndexOf(name, 0);
            if (r >= 0)
            {
                Console.WriteLine("REL PATH INDEX={0} {1}", r, rel_path);
                rel_path = rel_path.Substring(r + name.Length);
                if (rel_path.Length >= 1 && rel_path[0] == '_')
                    rel_path = rel_path.Substring(1);
            }
            return rel_path;
        }
        private void _WriteCaseMainPage(IList<ModuleInfo> modules, IList<ServiceInfo> services)
        {
            string out_lang = m_code_info.Language;
            string full_name = Path.Combine(m_output_path, "out_casemainpage.dox");
            using (TextWriter tw = new StreamWriter(full_name, false, Utils.WriteEncoding))
            {
                tw.WriteLine("/*!");
                tw.WriteLine("\n\\page axldoc_casemainpage.dox Modules and services\n");
                tw.WriteLine("<h2>List of modules</h2>");
                tw.WriteLine("<ul>");
                foreach (ModuleInfo module in modules)
                {
                    string module_path = _GetServiceOrModulePath(module);
                    tw.WriteLine("<li>\\subpage axldoc_module_{0} \"{1}\" ({2})</li>", module.FileBaseName,
                                 module.GetTranslatedName(out_lang), module_path);
                    //tw.Write("<li>{0}</li>\n",module.GetTranslatedName("fr"));
                    if (m_dico_writer != null)
                    {
                        m_dico_writer.Write("pagename=axldoc_module_" + module.FileBaseName + " frname=" + module.GetTranslatedName(out_lang) + "\n");
                    }
                }
                tw.WriteLine("</ul>");
                tw.WriteLine("<h2>List of services</h2>");
                tw.WriteLine("<ul>");
                foreach (ServiceInfo service in services)
                {
                    string service_path = _GetServiceOrModulePath(service);
                    tw.WriteLine("<li>\\subpage axldoc_service_{0} \"{1}\" ({2}) : implements", service.FileBaseName,
                                 service.GetTranslatedName(out_lang), service_path);
                    foreach (ServiceInfo.Interface ii in service.Interfaces)
                    {
                        tw.WriteLine("#{0} ", ii.Name);
                    }
                    tw.WriteLine("</li>");
                    if (m_dico_writer != null)
                    {
                        m_dico_writer.Write("pagename=axldoc_service_" + service.FileBaseName + " frname=" + service.GetTranslatedName(out_lang) + "\n");
                    }
                }
                tw.WriteLine("</ul>");
                tw.WriteLine("*/");
            }
        }
    }
}