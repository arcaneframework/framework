/*
 MainAXLDOC.cs (C) 2000-2012

 Générateur de documentation à partir de fichiers AXL.
*/

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Arcane.Axl
{
  public class MainAXLDOC
  {
   public MainAXLDOC()
    {
    }

    public static int Main(string[] args)
    {
      MainAXLDOC v = new MainAXLDOC();
      return v.Execute(args);
    }

    public int Execute(string[] args)
    {
      int nb_arg = args.Length;
      if (nb_arg==0) {
        _PrintUsage();
        return 1;
      }
      bool do_generate_final = false;
      string doc_language = "fr";
      m_output_path = Directory.GetCurrentDirectory();
      List<string> files = new List<string>();
      string wanted_encoding = null;
      for( int i=0; i<nb_arg; ++i ){
        string arg = args[i];
        if (arg.StartsWith("-") || arg.StartsWith("--")){
          if (arg=="-a" || arg=="--arcane-file"){
            if (i==nb_arg)
              throw new ArgumentException("Bad argument for option '--arcane-file'");
            else
              m_arcane_db_file = args[++i];
          }
          else if (arg=="-o" || arg=="--output-path"){
            if (i==nb_arg)
              throw new ArgumentException("Bad argument for option '--output-path'");
            else
              m_output_path = args[++i];
          }
          else if (arg=="-u" || arg=="--user-class"){
            if (i==nb_arg)
              throw new ArgumentException("Bad argument for option '--user-class'");
            else
              m_user_class = args[++i];
          }
          else if (arg=="-l" || arg=="--language"){
            if (i==nb_arg)
              throw new ArgumentException("Bad argument for option '--language'");
            else
              doc_language = args[++i];
          }
          else if (arg=="-e" || arg=="--examples"){
            m_do_examples = true;
          }
          else if (arg=="--encoding"){
            if (i==nb_arg)
              throw new ArgumentException("Bad argument for option '--encoding'");
            else
              wanted_encoding = args[++i];
          }
          else if (arg=="-d" || arg=="--dico"){
            m_do_dico = true;
          }
          else if (arg=="--generate-final"){
            do_generate_final = true;
          }
          else
            throw new ArgumentException("Unknown argument '"+arg+"'");
        }
        else 
          files.Add(arg);
      }
      Console.WriteLine("ARGS[1] OUTPUT PATH " + m_output_path);
      if (!String.IsNullOrEmpty(wanted_encoding))
        Utils.WriteEncoding = Encoding.GetEncoding(wanted_encoding);

      CodeInfo code_info = new CodeInfo(m_arcane_db_file);
      code_info.Language = doc_language;
      if (do_generate_final){
        if (files.Count==0){
          Console.WriteLine("No directory specified");
          return 1;
        }
        if (files.Count>1){
          Console.WriteLine("Too many directories ({0}) specified",files.Count);
          return 1;
        }
        FinalAxlGenerator axl_gen = new FinalAxlGenerator(code_info,files[0]);
        axl_gen.Generate(m_output_path);
        return 0;
      }
      DoxygenDocumentationGenerator doc_gen = new DoxygenDocumentationGenerator(m_output_path,m_user_class,
                                                                                code_info,m_do_examples,m_do_dico);
      doc_gen.Generate(files);

      return 0;
    }
    private void _PrintUsage()
    {
      Console.WriteLine("Usage: axldoc [-o|--output output_path] [-a|--arcane-db-file arcane_file] [-u|--user-class class] [-l|--language doc_language] [-e|--examples] [-d|--dico] axlfiles");
      Console.WriteLine("Usage: axldoc --generate-final [-l|--language doc_language] -o|--output output_path -a|--arcane-db-file arcane_file axldirectory");
    }

    private string m_arcane_db_file;
    private string m_output_path;
    private string m_user_class;
    private bool m_do_examples;
    private bool m_do_dico;
  }
}
