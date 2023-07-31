//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*
 MainAXLDOC.cs (C) 2000-2023

 Générateur de documentation à partir de fichiers AXL.
*/

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Reflection;
using Arcane.AxlDoc.UserInterfaces;
using Arcane.Axl;

namespace Arcane.AxlDoc
{
  public enum XmlGenerationMode { None, WithInfo, WithoutInfo };
  public enum SortMode { None, Alpha, IndexAlpha };

  public interface Config
  {
    bool verbose { get; }
    bool debug { get; }
    string arcane_db_file { get; }
    string output_path { get; }
    string user_class { get; }
    bool do_examples { get; }
    bool do_dico { get; }
    bool show_details { get; }
    SortMode do_sort { get; }
    int max_display_size { get; }
    XmlGenerationMode do_generate_xsd { get; }
    IApplicationPages private_app_pages { get; }
  };

  public class ConfigWriter : Config
  {
    public ConfigWriter() {
      output_path = Directory.GetCurrentDirectory();
      do_examples = false;
      do_dico = false;
      do_generate_xsd = XmlGenerationMode.None;
      private_app_pages = null;
      do_sort = SortMode.IndexAlpha;
      max_display_size = 30;
    }
    public bool verbose { get; set; }
    public bool debug { get; set; }
    public string arcane_db_file { get; set; }
    public string output_path { get; set; }
    public string user_class { get; set; }
    public bool do_examples { get; set; }
    public bool do_dico { get; set; }
    public bool show_details { get; set; }
    public SortMode do_sort { get; set; }
    public int max_display_size { get; set; }
    public XmlGenerationMode do_generate_xsd { get; set; }
    public IApplicationPages private_app_pages { get; set; }
  };

  public class MainAXLDOC
  {
    #region CONFIGURATIONS
    private static string codename = "axldoc";
    private static int RTCODE_OK = 0;
    private static int RTCODE_ERROR = 1;
    private static string default_dll_filename = "ApplicationPages.dll";
    #endregion

    static private bool m_debug = false; // TODO merge this with Config

    public static int MainExec(string[] args)
    {
      try {
        MainAXLDOC v = new MainAXLDOC();
        v.Execute(args);
        return RTCODE_OK;
      } catch (AxlToolExit e) {
        Console.WriteLine (e.Message);
        return RTCODE_OK;
      } catch (Exception e) {
        Console.Error.WriteLine (String.Format ("AXLDOC: Error during generation: {0}", e.Message));
        if (m_debug)
          Console.WriteLine("Stacktrace: {0}", e.StackTrace);
        return RTCODE_ERROR;
      }
    }

    public void Execute(string[] args)
    {
      m_config.private_app_pages = _loadAppPlugin ();
      m_options = new Mono.Options.OptionSet ();

      // Set default values
      bool show_help = false;
      bool do_generate_final = false;
      string doc_language = "fr";
      List<string> files = new List<string>();
      string wanted_encoding = null;
      bool doc_legacy = false;

      // Configure options
      m_options.Add ("h|help", "Help page", v => { show_help = true; });
      m_options.Add ("v|verbose", "Enable verbose mode", v => { m_config.verbose = true; });
      m_options.Add ("debug", "Enable debug mode", v => { m_config.debug = true; m_debug = true; });
      m_options.Add ("a|arcane-file=", "Arcane file", (string s) => { m_config.arcane_db_file = s; });
      m_options.Add ("u|user-class=", "User class", (string s) => { m_config.user_class = s; });
      m_options.Add ("l|language=", "Language", (string s) => { doc_language = s; });
      m_options.Add ("e|examples", "Generate examples", v => { m_config.do_examples = true; });
      m_options.Add ("encoding=", "Define encoding", (string s) => { wanted_encoding = s; });
      m_options.Add ("d|dico", "Dictionary", v => { m_config.do_dico = true; });
      m_options.Add ("show-details", "Add details", v => { m_config.show_details = true; });
      m_options.Add ("max-display-size=", 
                     String.Format("Maximal option number by page (default is {0}; value <= 0 means unlimited)", m_config.max_display_size),
                     (int i) => { m_config.max_display_size = i; });
      m_options.Add ("sort=", "Sort algorithm for options; \nallowed values are None (disabled), IndexAlpha (default), Alpha", 
                     (string s) => {  SortMode mode = SortMode.IndexAlpha;
                                      if (!String.IsNullOrEmpty (s))
                                      if (!Enum.TryParse (s, out mode))
                                        _showHelpAndExit (String.Format ("Invalid sort option value '{0}'", s));
                                      m_config.do_sort = mode;
                                    });
      m_options.Add ("generate-final", "Generate final", v => { do_generate_final = true; });
      m_options.Add ("generate-xsd:", "Generate XSD with info; \nallowed values are None (disabled), WithInfo (default when enabled), WithoutInfo", 
                     (string s) => {  XmlGenerationMode mode = XmlGenerationMode.WithInfo;
                                      if (!String.IsNullOrEmpty (s))
                                      if (!Enum.TryParse (s, out mode))
                                        _showHelpAndExit (String.Format ("Invalid generate-xsd option value '{0}'", s));
                                      m_config.do_generate_xsd = mode;
                                   });
      m_options.Add ("f|axl-list-file=", "Axl list file", (string s) => { var axl_list = File.ReadAllLines(s); files = new List<string>(axl_list); });
      m_options.Add ("o|output-path=", "Output path for generated code\n", (string s) => { m_config.output_path = s; }); // final \n to separate plugin options
      m_options.Add ("legacy", "Generate legacy documentation", v => { doc_legacy = true; });

      // For plugins
      m_config.private_app_pages.Configure (m_options);

      try {
        List<string> remaining_args = m_options.Parse (args);

        // Check if remaining args looks like options
        for (int i=0; i<remaining_args.Count; ++i) {
          String arg = remaining_args [i];
          if (arg.StartsWith("-"))
            throw new Mono.Options.OptionException (String.Format ("Invalid option {0}", arg), arg);
          else 
            files.Add (arg);
        }
        if (show_help) {
          _showHelpAndExit ();
        }
      } catch (Mono.Options.OptionException e) {
        _showHelpAndExit (e.Message);
      }

      if (files.Count == 0) {
        _showHelpAndExit("Missing arg files; use -f or add filenames");
      }
      if (m_config.debug)
        Console.WriteLine("ARGS[1] OUTPUT PATH " + m_config.output_path);

      if (!String.IsNullOrEmpty(wanted_encoding))
        Utils.WriteEncoding = Encoding.GetEncoding(wanted_encoding);

      CodeInfo code_info = new CodeInfo(m_config.arcane_db_file);
      code_info.Language = doc_language;
      code_info.Legacy = doc_legacy;

      if (do_generate_final) {
        if (files.Count > 1) 
          throw new AxlToolException (String.Format ("Too many directories ({0}) specified", files.Count));
        FinalAxlGenerator axl_gen = new FinalAxlGenerator (code_info, files [0]);
        axl_gen.Generate (m_config.output_path);
      } else {
        DoxygenDocumentationGenerator doc_gen = new DoxygenDocumentationGenerator (m_config, code_info);
        doc_gen.Generate (files);
      }
    }

    private void _showHelpAndExit (String message = null)
    {
      StringWriter writer = new StringWriter ();
      if (message == null)
        writer.WriteLine ("Requested Help page");
      else
        writer.WriteLine (message);

      writer.WriteLine ("Usage: {0} [options] [args]", codename);
      writer.WriteLine ("Options : ");
      m_options.WriteOptionDescriptions (writer);

      writer.WriteLine ("Environment variable:");
      writer.WriteLine ("\tAXLDOC_CUSTOM_PLUGIN : define directory where {0} can be found or direct filename", default_dll_filename);

      writer.WriteLine ("\nUsage hints");
      writer.WriteLine ("\t{0} axlfiles...", codename);
      writer.WriteLine ("\t{0} --generate-final axldirectory", codename);
      writer.WriteLine ("\t{0} -f axldb.txt [extra files]", codename); // usage @ifpen

      if (message == null)
        throw new AxlToolExit (writer.ToString ());
      else
        throw new AxlToolException (writer.ToString ());
    }

    private static IApplicationPages _loadAppPlugin()
    {

      string dll_env_path = Environment.GetEnvironmentVariable ("AXLDOC_CUSTOM_PLUGIN");
      string dll_filename = dll_env_path ?? ".";
      if (Directory.Exists (dll_filename)) {
        dll_filename = Path.Combine (dll_filename, default_dll_filename);
      }

      IApplicationPages private_app_pages = null;

      if (File.Exists (dll_filename)) {
        // Console.WriteLine("Checking file {0}", dll_filename);
        AssemblyName assembly_name = AssemblyName.GetAssemblyName (dll_filename);
        Assembly assembly = Assembly.Load (assembly_name); // must be present in run dir...

        if (assembly != null) {
          Type[] types = assembly.GetTypes ();
          foreach (Type type in types) {
            // Console.WriteLine ("Try to find custom interface with {0}", type);
            if (type.GetInterface (typeof(IApplicationPages).FullName) != null) {
              private_app_pages = (IApplicationPages)Activator.CreateInstance (type);
            }
          }
        }

        if (private_app_pages != null) {
          Console.WriteLine ("Custom plugin found and loaded in from {0}", dll_filename);
        } else {
          Console.WriteLine ("WARNING: Skipping not a compatible plugin {0}.", dll_filename);
        }
      } else {
        if (dll_env_path != null)
          Console.WriteLine ("WARNING: No plugin file found in {0}", dll_env_path);
      }

      return private_app_pages ?? new DefaultApplicationPages ();
    }

    #region MEMBERS
    private ConfigWriter m_config = new ConfigWriter();
    private Mono.Options.OptionSet m_options = null;
    #endregion
  }
}
