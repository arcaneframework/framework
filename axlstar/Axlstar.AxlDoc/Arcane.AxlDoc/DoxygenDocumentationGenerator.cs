//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DoxygenDocumentationGenerator.cc                            (C) 2000-2023 */
/*                                                                           */
/* Génération de la documentation au format Doxygen.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
using System;
using System.IO;
using System.Collections.Generic;
using System.Xml;
using System.Text;
using Arcane.Axl;

namespace Arcane.AxlDoc
{
  public class DoxygenDocumentationGenerator
  {
    private Config m_config;
    private CodeInfo m_code_info;
    private StreamWriter m_dico_writer = null;
    private XmlSchemaFile m_xml_schema_file;
    private ApplicationPageVisitor m_private_application_visitor = null;
    private DoxygenOptionIndexVisitor m_option_index_visitor;

    public DoxygenDocumentationGenerator (Config config, CodeInfo code_info)
    {
      m_config = config;
      m_code_info = code_info;
      if (m_config.do_dico) {
        string dico_file_name = Path.Combine (m_config.output_path, "dicosearch");
        FileStream dico_file = new FileStream (dico_file_name, FileMode.OpenOrCreate, FileAccess.Write);
        if (config.verbose)
          Console.WriteLine ("Configure for 'dicosearch' generation in path '{0}'", dico_file_name);
        m_dico_writer = new StreamWriter (dico_file);
      }
      m_xml_schema_file = new XmlSchemaFile (m_config, code_info.ApplicationName);
    }

    public void Generate (List<string> axl_files)
    {
      if (m_config.verbose)
        Console.WriteLine ("CREATE DIRECTORY {0}", m_config.output_path);

      Directory.CreateDirectory (m_config.output_path);

      SortedList<string, ModuleInfo> modules = new SortedList<string, ModuleInfo> ();
      SortedList<string, ServiceInfo> services = new SortedList<string, ServiceInfo> ();
      string out_language = m_code_info.Language;
      foreach (string axl_file in axl_files) {
        AXLParser parser = AXLParserFactory.CreateParser (axl_file, m_config.user_class);
        parser.ParseAXLFileForDocumentation ();
        ServiceOrModuleInfo info = parser.ServiceOrModule;
        ModuleInfo module_info = parser.Module;
        string service_or_module_name;
        string axl_base_name = Path.GetFileNameWithoutExtension (axl_file);
        if (m_config.verbose)
          Console.WriteLine ("** ** ** AXL BASE NAME {0} userclass={1}", axl_base_name, m_config.user_class);
        if (!_IsValidClass (info.UserClasses)) {
          Console.WriteLine ("BAD CLASS. REMOVE {0}", axl_base_name);
          continue;
        }

        if (module_info == null) {
          ServiceInfo service_info = parser.Service;
          if (m_config.private_app_pages.Filter (service_info)) {
            service_or_module_name = service_info.GetTranslatedName (out_language) + service_info.FileBaseName;
            services [service_or_module_name] = service_info;
          }
        } else {
          if (m_config.private_app_pages.Filter (module_info)) {
            service_or_module_name = module_info.GetTranslatedName (out_language) + module_info.FileBaseName;
            modules [service_or_module_name] = module_info;
          }
        }
      }

      m_option_index_visitor = new DoxygenOptionIndexVisitor (m_code_info, out_language, m_config.private_app_pages);
      m_private_application_visitor = new ApplicationPageVisitor (m_config);


      if (m_config.do_examples) {
        DoxygenExampleFile.OutputPath = m_config.output_path;
        Directory.CreateDirectory (DoxygenExampleFile.OutputPath);
      }

      // On traite les services avant les modules
      foreach (ServiceOrModuleInfo service_info in services.Values) {
        if (m_config.private_app_pages.Filter (service_info)) {
          _generateServiceOrModule (service_info);
          _generateServiceOrModuleXSD (service_info);
          _generateServiceOrModuleApplicationPage (service_info); // launch private applicative doc for current service
        }
      }
      foreach (ServiceOrModuleInfo module_info in modules.Values) {
        if (m_config.private_app_pages.Filter (module_info)) {
          _generateServiceOrModule (module_info);
          _generateServiceOrModuleXSD (module_info);
          _generateServiceOrModuleApplicationPage (module_info); // launch private applicative doc for current module
        }
      }

      if (m_config.do_examples) {
        DoxygenExampleFile.WriteAliases ();
      }

      if(m_code_info.Legacy){
        m_option_index_visitor.GenerateOldIndex (Path.Combine (m_config.output_path, "axldoc_all_option_index.md"));
      }
      else{
        m_option_index_visitor.Generate (Path.Combine (m_config.output_path, "axldoc_all_option_index.md"));
      }
      
      _WriteCaseMainPageMarkdown(modules.Values, services.Values);
      _WriteInterfaceMainPageMarkdown (services.Values);

      if (m_dico_writer != null) {
        m_dico_writer.Close ();
      }

      m_config.private_app_pages.Summary ();

      // Write xsd
      m_xml_schema_file.write ();

    }

    private void _generateServiceOrModule (ServiceOrModuleInfo info)
    {
      DoxygenDocumentationFile doc_file = new DoxygenDocumentationFile (info, m_config.output_path, m_code_info.Language);

      DoxygenDocumentationVisitor dbdv = new DoxygenDocumentationVisitor (doc_file, m_code_info, m_dico_writer, m_config);
      dbdv.AddExtraDescriptionWriter (m_private_application_visitor); // To add applicative info stored in description sub nodes

      if (m_config.user_class == null)
        dbdv.PrintUserClass = true;
      dbdv.VisitServiceOrModule (info);

      doc_file.Write ();

      if (m_config.do_examples) {
        DoxygenExampleFile example_file = new DoxygenExampleFile (info);
        DoxygenExampleVisitor dev = new DoxygenExampleVisitor (example_file, m_code_info, "fr", m_config.private_app_pages);
        dev.VisitServiceOrModule (info);
        example_file.Write ();
      }
      m_option_index_visitor.VisitServiceOrModule (info);
    }

    private void _generateServiceOrModuleXSD (ServiceOrModuleInfo info)
    {
      XmlSchemaVisitor xml_schema_visitor = new XmlSchemaVisitor (m_xml_schema_file, m_code_info);
      xml_schema_visitor.VisitServiceOrModule (info);
    }

    private void _generateServiceOrModuleApplicationPage (ServiceOrModuleInfo info)
    {
      m_private_application_visitor.VisitServiceOrModuleInfo (info);
    }

    private bool _IsValidClass (string [] values)
    {
      if (m_config.user_class == null)
        return true;
      foreach (string s in values) {
        if (s == m_config.user_class)
          return true;
      }
      return false;
    }
    /*!
           * \brief Génère le chemin relative d'un service ou d'un module.
         */
    string _GetServiceOrModulePath (ServiceOrModuleInfo smi)
    {
      // Supprime le nom du service ou module du chemin.
      // Par exemple, si le service s'appelle Toto et que
      // le chemin est Toto_titi_tata, retourne titi_tata.
      string name = smi.Name;
      string rel_path = smi.FileBaseName;
      int r = rel_path.IndexOf (name, 0);
      if (r >= 0) {
        if (m_config.verbose)
          Console.WriteLine ("REL PATH INDEX={0} {1}", r, rel_path);
        rel_path = rel_path.Substring (r + name.Length);
        if (rel_path.Length >= 1 && rel_path [0] == '_')
          rel_path = rel_path.Substring (1);
      }
      return rel_path;
    }

    string _GetBriefDescription (ServiceOrModuleInfo info)
    {
      // Considère que la première phrase de la description correspond au résumé.
      // On récupère donc la description et on tronque au premier point '.'
      // rencontré. De plus, si la chaîne dépasse 120 caractères, elle est tronquée aussi.
      // Cela permet d'éviter d'avoir une description brève trop longue.
      XmlElement desc_elem = info.DescriptionElement;
      if (desc_elem == null)
        return null;

      // Partie permettant de forcer la fermeture des balises avant de couper
      // la description.
      bool force_close_cmds = false;
      if (desc_elem.HasAttribute("doc-brief-force-close-cmds")){
        string str = desc_elem.GetAttribute("doc-brief-force-close-cmds");
        force_close_cmds = Convert.ToBoolean(str);
      }

      // Partie permettant de définir une limite au nombre de caractères
      // pour la description (plus quelques caractères, on ne découpe pas
      // un mot au milieu).
      // -1 signifie pas de limites.
      int max_nb_char = 120;
      if (desc_elem.HasAttribute("doc-brief-max-nb-of-char")){
        string str = desc_elem.GetAttribute("doc-brief-max-nb-of-char");
        max_nb_char = Convert.ToInt32(str);
        if(max_nb_char == -1){
          max_nb_char = desc_elem.InnerText.Length;
        }
      }

      // Partie permettant de finir la description au premier point trouvé.
      bool stop_at_dot = true;
      if (desc_elem.HasAttribute("doc-brief-stop-at-dot")){
        string str = desc_elem.GetAttribute("doc-brief-stop-at-dot");
        stop_at_dot = Convert.ToBoolean(str);
      }

      Console.WriteLine ("GET BRIEF DESCRIPTION name={0} -- force_close_cmds={1} -- max_nb_char={2} -- stop_at_dot={3}", 
        info.Name, force_close_cmds, max_nb_char, stop_at_dot
      );

      string input_text = desc_elem.InnerText;
      if(stop_at_dot){
        int pos = input_text.IndexOf(".");
        if (pos >= 0){
          input_text = input_text.Substring(0, pos+1);
        }
      }

      int nb_char = 0;

      // Le dico qui contient les balises qui peuvent apparaitre
      // dans les courtes descriptions.
      Dictionary<string, string> balises = new Dictionary<string, string>
      {
          { "\\verbatim", "\\endverbatim" },
          { "\\code", "\\endcode" },
          { "\\ref", "" },
          { "\\subpage", "" }, // TODO : Verifier qu'on ne découpe pas les 2 mots d'après
          { "\\f$", "\\f$" },
          { "\\f(", "\\f)" },
          { "\\f[", "\\f]" },
          { "\\f{", "\\f}" }
      };

      // La stack qui contiendra les balises fermantes restantes.
      // Si, à la fin de la lecture, une ou plusieurs balises
      // ne sont pas refermées, on a plus qu'a copier les balises de la stack
      // à la fin de la chaine de caractères.
      Stack<string> stack_balises = new Stack<string>();

      StringBuilder output_string = new StringBuilder();

      string[] lines = input_text.Split(new[] { "\n", "\r\n" }, StringSplitOptions.None);
      foreach (string line in lines)
      {
        // Divise la ligne en mots.
        string[] words = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (string word in words){

          // Vérifie si le mot est une balise ouvrante.
          if (balises.ContainsKey(word)){

            // Pas besoin d'ajouter les balises qui n'ont pas de balises fermantes.
            if(!string.IsNullOrEmpty(balises[word])){
              stack_balises.Push(balises[word]);
            }
            output_string.Append(word).Append(" ");
          }

          // Vérifie si le mot est une balise fermante.
          else if (balises.ContainsValue(word)){

            // Dépile et écrit les balises jusqu'à trouver la bonne balise fermante.
            while (stack_balises.Count > 0 && stack_balises.Peek() != word){
              output_string.Append(stack_balises.Pop()).Append(" ");
            }

            // Retire la balise fermante de la pile si elle a été trouvée.
            if (stack_balises.Count > 0 && stack_balises.Peek() == word){
              output_string.Append(stack_balises.Pop()).Append(" ");
            }
          }

          // Si le mot commence par un backslash, n'est pas une balise connue et si la stack
          // est vide, on retire le backslash.
          else if (word.StartsWith("\\") && stack_balises.Count == 0){
            output_string.Append(word.Substring(1)).Append(" ");
            nb_char += word.Length - 1;
          }

          // Si ce n'est pas une balise et ne commence pas par un backslash, on ajoute tel quel le mot.
          else{
            output_string.Append(word).Append(" ");
            nb_char += word.Length;
          }

          // Si on a atteint la limite de caractère, on break.
          // Si force_close_attributes == True, alors il faut que la stack soit vide pour finir.
          if(nb_char >= max_nb_char && (!force_close_cmds || stack_balises.Count == 0)){
            break;
          }
        }

        // Si on a atteint la limite de caractère, on break.
        // Si force_close_attributes == True, alors il faut que la stack soit vide pour finir.
        if(nb_char >= max_nb_char && (!force_close_cmds || stack_balises.Count == 0)){
          output_string.Append("...");
          break;
        }
        output_string.Append(Environment.NewLine);
      }

      // Ajoute les balises restantes de la stack à la fin du texte.
      while (stack_balises.Count > 0){
        output_string.Append(stack_balises.Pop()).Append(" ");
      }

      // Récupère le string et supprime le retour à la ligne final.
      return output_string.ToString().TrimEnd(Environment.NewLine.ToCharArray());
    }

    /*!
     * \brief Génère la page d'accueil contenant la liste des modules et services.
     */
    private void _WriteCaseMainPageMarkdown (IList<ModuleInfo> modules, IList<ServiceInfo> services)
    {
      string out_lang = m_code_info.Language;
      string full_name = Path.Combine (m_config.output_path, "axldoc_casemainpage.md");

      // Affiche le chemin des sources du service ou module si on demande le mode développeur.
      bool want_path = String.IsNullOrEmpty (m_config.user_class);

      using (TextWriter tw = new StreamWriter (full_name, false, Utils.WriteEncoding)) {

        if(modules.Count != 0 && services.Count != 0){
          tw.WriteLine ("# {0} {{#axldoc_casemainpage}}\n", m_code_info.Translation.ModuleAndServices);
        }
        else if(modules.Count != 0){
          tw.WriteLine ("# {0} {{#axldoc_casemainpage}}\n", m_code_info.Translation.ListOfModules);
        }
        else if(services.Count != 0){
          tw.WriteLine ("# {0} {{#axldoc_casemainpage}}\n", m_code_info.Translation.ListOfServices);
        }
        else{
          tw.WriteLine ("# {0} {{#axldoc_casemainpage}}\n", m_code_info.Translation.ModuleAndServices);
        }

        if(modules.Count != 0){
          tw.WriteLine ("## {0}\n", m_code_info.Translation.ListOfModules);
          tw.WriteLine ("<table>");
          tw.Write ("<tr><th>Balise XML</th><th>Nom</th>");
          if (want_path)
            tw.Write ("<th>Chemin</th>");
          tw.WriteLine ("<th>Description</th></tr>");
          foreach (ModuleInfo module in modules) {
            tw.Write ("<tr><td>{1}</td><td>\\subpage axldoc_module_{0} \"{2}\"</td>", module.FileBaseName,
              module.GetTranslatedName (out_lang), module.Name);
            if (want_path) {
              string module_path = _GetServiceOrModulePath (module);
              tw.Write ("<td>{0}</td>", module_path);
            }
            tw.Write ("<td>{0}</td>", _GetBriefDescription (module));
            tw.Write ("</tr>\n");
            if (m_dico_writer != null) {
              m_dico_writer.Write ("pagename=axldoc_module_" + module.FileBaseName + " frname=" + module.GetTranslatedName (out_lang) + "\n");
            }
          }
          tw.WriteLine ("</table>");
        }

        if(services.Count != 0){
          tw.WriteLine ("\n## {0}\n", m_code_info.Translation.ListOfServices);
          tw.WriteLine ("<table>");
          tw.WriteLine ("<tr><th>Nom</th><th>Interface</th>");
          if (want_path)
            tw.Write ("<th>Chemin</th>");
          tw.Write ("<th>Description</th>");
          tw.WriteLine ("</tr>");
          foreach (ServiceInfo service in services) {
            tw.Write ("<tr><td>\\subpage axldoc_service_{0} \"{1}\"</td>", service.FileBaseName,
              service.GetTranslatedName (out_lang));
            tw.Write ("<td>");
            foreach (ServiceInfo.Interface ii in service.Interfaces) {
              tw.WriteLine ("%{0} ", ii.Name);
            }
            tw.Write ("</td>");
            if (want_path) {
              string service_path = _GetServiceOrModulePath (service);
              tw.Write ("<td>&space;{0}</td>", service_path);
            }
            tw.Write ("<td>{0}</td>", _GetBriefDescription (service));
            tw.WriteLine ("</td></tr>");
            if (m_dico_writer != null) {
              m_dico_writer.Write ("pagename=axldoc_service_" + service.FileBaseName + " frname=" + service.GetTranslatedName (out_lang) + "\n");
            }
          }
          tw.WriteLine ("</table>");
        }
      }
    }


    void _WriteInterfaceMainPageMarkdown (IList<ServiceInfo> services)
    {
      string out_lang = m_code_info.Language;
      string full_name = Path.Combine (m_config.output_path, "axldoc_interfacemainpage.md");
      SortedList<String, SortedList<String, ServiceInfo>> interfaces = new SortedList<String, SortedList<String, ServiceInfo>> ();

      foreach (ServiceInfo service in services) {
        foreach (ServiceInfo.Interface intface in service.Interfaces) {
          SortedList<String, ServiceInfo> related_services = null;
          if (!interfaces.TryGetValue (intface.Name, out related_services)) {
            related_services = new SortedList<String, ServiceInfo> ();
            interfaces.Add (intface.Name, related_services);
          }
          if (related_services.ContainsKey (service.Name)) {
            Console.WriteLine ("WARNING: Several services with same name (name='{0}')", service.Name);
          } else
            related_services.Add (service.Name, service);
        }
      }

      using (TextWriter tw = new StreamWriter (full_name, false, Utils.WriteEncoding)) {
        tw.WriteLine("# " + m_code_info.Translation.ServiceInterfaces + " {#axldoc_interfacemainpage}\n");
        if (interfaces.Count > 0) {
          tw.WriteLine ("## {0}\n", m_code_info.Translation.ListOfInterfaces);
          tw.WriteLine ("<table class=\"doxtable\">");
          tw.WriteLine ("<tr><th>Interface</th><th>{0}</th></tr>", m_code_info.Translation.Implementations);

          foreach (var intface in interfaces) {
            tw.WriteLine("<tr>");
            tw.WriteLine("<td>\\anchor axldoc_interface_{1} {0}</td>", intface.Key, DoxygenDocumentationUtils.AnchorName(intface.Key));
            tw.WriteLine("<td>");
            foreach (var service in intface.Value.Values) {
              string service_path = _GetServiceOrModulePath (service);
              if(String.IsNullOrEmpty(service_path)){
                tw.WriteLine ("- \\ref axldoc_service_{0} \"{1}\"",
                              service.FileBaseName, service.GetTranslatedName (out_lang));
              }
              else{
                tw.WriteLine ("- \\ref axldoc_service_{0} \"{1}\" ({2})",
                              service.FileBaseName, service.GetTranslatedName (out_lang), service_path);
              }
            }
            tw.WriteLine("</td></tr>");
          }
          tw.WriteLine("</table>");
        }
      }
    }
  }
}
