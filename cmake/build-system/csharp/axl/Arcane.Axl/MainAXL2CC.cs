/*---------------------------------------------------------------------------*/
/* MainAXL2CC.cs                                               (C) 2000-2012 */
/*                                                                           */
/* Programme principal du script de génération de code à partir de           */
/* fichiers AXL.                                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using System;
using System.IO;

namespace Arcane.Axl
{  
  public class MainAXL2CC
  {
    public MainAXL2CC()
    {
    }

    public static int Main(string[] args)
    {
      MainAXL2CC v = new MainAXL2CC();
      return v.Execute(args);
    }

    public int Execute(string[] args)
    {
      int nb_arg = args.Length;
      if (nb_arg == 0) {
        _PrintUsage();
        return 1;
      }
      string axl_file_name = args[nb_arg - 1];
      m_full_file_name = axl_file_name;
      m_output_path = Directory.GetCurrentDirectory();
      m_include_path = ".";
      string copy_outfile = null;
      string language = "c++"; // Sortie en C++ par défaut.
      for (int i = 0; i < nb_arg - 1; ++i) {
        string arg = args[i];
        if (arg == "-i" || arg == "--header-path") {
          if (i == nb_arg)
            throw new ArgumentException("Bad argument for option '--header-path'");
          m_include_path = args[++i];
        }
        else if (arg == "-o" || arg == "--output-path") {
          if (i == nb_arg)
            throw new ArgumentException("Bad argument for option '--output-path'");
          m_output_path = args[++i];
        }
        else if (arg == "-l" || arg == "--lang") {
          if (i == nb_arg)
            throw new ArgumentException("Bad argument for option '--lang'");
          language = args[++i];
        } 
        else if (arg == "--copy") {
          if (i == nb_arg)
            throw new ArgumentException("Bad argument for option '--copy'");
          copy_outfile = args[++i];
        } 
        else if (arg == "--no-restore") {
          GlobalContext.Instance.NoRestore = true;
        }
        else if (arg.Contains("--verbose=") == true) {
          String verbose = arg.Replace("--verbose=","");
          if(verbose == "1")
            GlobalContext.Instance.Verbose = true;
          else if(verbose == "0" || verbose == "")
            GlobalContext.Instance.Verbose = false;
          else
            throw new ArgumentException(String.Format("Invalide verbose option '{0}'",verbose));
        }
        else
          throw new ArgumentException("Unknown argument '" + arg + "'");
      }

      string file_extension = Path.GetExtension(axl_file_name);
      if(GlobalContext.Instance.Verbose){
        Console.WriteLine("EXTENSION = {0}", file_extension);
      }

      if (Path.GetExtension(axl_file_name) != ".axl")
        throw new ArgumentException("axlfile has to have extension '.axl'");

      if(GlobalContext.Instance.Verbose){
        Console.WriteLine("OUTPUT PATH " + m_output_path);
        Console.WriteLine("ARGS[3] INCLUDE PATH " + m_include_path);
        Console.WriteLine("ARGS[4] FILE NAME " + m_full_file_name);
      }
      
      AXLParser parser = AXLParserFactory.CreateParser(m_full_file_name,null);

      parser.ParseAXLFile();
      if (language=="c++"){
        CodeGenerator generator = null;
        ModuleInfo module_info = parser.Module;
        if (module_info != null)
          generator = new CppModuleBaseGenerator(m_include_path, m_output_path, module_info);
        else
          generator = new CppServiceBaseGenerator(m_include_path, m_output_path, parser.Service);
        generator.writeFile();
      }
      else if (language=="c#"){
        Console.WriteLine("WARNING:  USE CSHARP !!!!!!!!!!!!!!");
        CodeGenerator generator = null;
        ModuleInfo module_info = parser.Module;
        if (module_info != null)
          generator = new CSharpModuleGenerator(m_include_path, m_output_path, module_info);
        else
          generator = new CSharpServiceGenerator(m_include_path,m_output_path,parser.Service);
        generator.writeFile();
      }
      else
        throw new ArgumentException(String.Format("Invalide language '{0}'",language));
      if (!String.IsNullOrEmpty(copy_outfile)){
        if(GlobalContext.Instance.Verbose) {
          Console.WriteLine("Copy '{0}' to '{1}'",m_full_file_name,copy_outfile);
        }
        File.Copy(m_full_file_name,copy_outfile,true);
      }

      return 0;
    }

    private void _PrintUsage()
    {
      Console.WriteLine("Usage: axl.exe [-i|--header header] [-o|--output output_path]"+
                        " [-l|--lang c++|c#] [--no-restore] [--copy outfile] [--verbose={ |0|1}] axlfile");
    }

    private string m_output_path;
    private string m_include_path;
    private string m_full_file_name;
  }
}
