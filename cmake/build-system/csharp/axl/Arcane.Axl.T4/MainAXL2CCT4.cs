/*---------------------------------------------------------------------------*/
/* Main.cs                                                     (C) 2000-2015 */
/*                                                                           */
/* Programme principal du script de génération de code à partir de           */
/* fichiers AXL.                                                             */
/* Version 2.0 utilisant la génération par template T4                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;
using System.Xml;
using System.Xml.Serialization;


namespace Arcane.Axl
{
  class MainAXL2CCT4
  {
    public static int Main(string[] args)
    {
      var v = new MainAXL2CCT4();

      return v.Execute(args);
    }

    private enum AxlKind
    {
      Module,
      Service,
      Unknown 
    }

    private void PrintUsage()
    {
      Console.WriteLine("Usage: axl.exe [-i|--header header] [-o|--output output_path]"+
                        " [-l|--lang c++|c#] [--no-restore] [--copy outfile] [--verbose={ |0|1}] axlfile");
    }

    static AxlKind axlKind (string axl)
    {
      using (var reader = XmlTextReader.Create(axl)) {
        while (reader.Read ()) {
          if ((reader.Name == "module") && reader.IsStartElement ()) {
            return AxlKind.Module;
          } 
          if ((reader.Name == "service") && reader.IsStartElement ()) {
            return AxlKind.Service;
          }
        }
      }
      return AxlKind.Unknown;
    }

    static private Arcane.Axl.Xsd.Service DeserializeService (string axl)
    {
      Arcane.Axl.Xsd.Service service = null;
      
      try {
        var serializer = new XmlSerializer (typeof(Arcane.Axl.Xsd.Service));
        using (var reader = XmlReader.Create(axl)) {
          service = (Arcane.Axl.Xsd.Service)serializer.Deserialize (reader);
        } 
      } catch(Exception e) {
        Console.WriteLine("XML parsing error: {0}\n{1}", e.Message, e.InnerException.Message);
        Environment.Exit(1);
      }
      return service;
    }

    static private Arcane.Axl.Xsd.Module DeserializeModule (string axl)
    {
      Arcane.Axl.Xsd.Module module = null;
      try {
        var serializer = new XmlSerializer (typeof(Arcane.Axl.Xsd.Module));
        using (var reader = XmlReader.Create(axl)) {
          module = (Arcane.Axl.Xsd.Module)serializer.Deserialize (reader);
        }
      } catch (Exception e) {
        Console.WriteLine ("XML parsing error: {0}\nInner exception : {1}", 
                        e.Message, e.InnerException.Message);
        Environment.Exit (1);
      }
      return module;
    }

    static private void GenerateModuleFile_Axl(Arcane.Axl.Xsd.Module module, string output_path)
    {
      var output_case = Path.Combine (output_path, module.Name + "_axl.h");
      using (var file = new StreamWriter(output_case)) {
        var case_options = new CaseOptionsT4 (module, version: "0.8");
        file.WriteLine (case_options.TransformText ());
        var base_class = new ModuleT4 (module, version: "0.8");
        file.WriteLine (base_class.TransformText ());
      }
    }
   
    // TODO mutualiser plus factory
    static private void GenerateModule (string axl, string output_path) 
    {
      Arcane.Axl.Xsd.Module module = DeserializeModule (axl);
      GenerateModuleFile_Axl (module, output_path);
    }

    public int Execute(string[] args)
    {
//      string[] args = { 
//        "--copy", 
//        "/work/IRLIN276_1/desrozis/IFPEN/working/arcane-axl/arcane/debug/share/axl/ArcaneCasePartitioner_arcane_std.axl",
//        "-o",
//        "/work/IRLIN276_1/desrozis/IFPEN/working/arcane-axl/arcane/debug/arcane/std",
//        "/work/IRLIN276_1/desrozis/IFPEN/working/arcane-axl/arcane/src/arcane/std/ArcaneCasePartitioner.axl"
//      };

      int nb_arg = args.Length;
      if (nb_arg == 0) {
        PrintUsage();
        return 1;
      }
      string generation_mode = null;
      bool with_arcane = true;
      bool with_mesh = true;
      bool with_parameter_factory = false;
      string axl_file_name = args[nb_arg - 1];
      string full_file_name = axl_file_name;
      string output_path = Directory.GetCurrentDirectory();
      string include_path = ".";
      string copy_outfile = null;
      string language = "c++"; // Sortie en C++ par défaut.
      for (int i = 0; i < nb_arg - 1; ++i) {
        string arg = args[i];
        if (arg == "-i" || arg == "--header-path") {
          if (i == nb_arg)
            throw new ArgumentException("Bad argument for option '--header-path'");
          include_path = args[++i];
        }
        else if (arg == "-o" || arg == "--output-path") {
          if (i == nb_arg)
            throw new ArgumentException("Bad argument for option '--output-path'");
          output_path = args[++i];
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
        else if (arg == "-g" || arg == "--gen-target") {
          if (i == nb_arg)
            throw new ArgumentException("Bad argument for option '--gen-target'");
          generation_mode = args[++i];
        }
        else if (arg == "--no-restore") {
          GlobalContext.Instance.NoRestore = true;
        }
        else if (arg == "--with-arcane") {
          if (i == nb_arg)
            throw new ArgumentException("Bad argument for option '--with-arcane'");
          if(args[++i]=="no")
            with_arcane = false;
        }
        else if (arg == "--with-mesh") {
          if (i == nb_arg)
            throw new ArgumentException("Bad argument for option '--with-mesh'" +
              "'");
          if(args[++i]=="no")
            with_mesh = false;
        }
        else if (arg == "--with-parameter-factory") {
          if (i == nb_arg)
            throw new ArgumentException("Bad argument for option '--with-parameter-factory'" +
                                        "'");
          if(args[++i]=="yes")
            with_parameter_factory = true;
        }
        else if (arg == "--namespace-simple-types") {
          if (i == nb_arg)
            throw new ArgumentException("Bad argument for option '--simple-type-namespace'");
          // a revoir
          SimpleTypeExtensions.namespaceT = args[++i];
        }
        else if (arg == "--export") {
          if (i == nb_arg)
            throw new ArgumentException("Bad argument for option '--export'");
          // a revoir
          SimpleTypeExtensions.exportT = args[++i];
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

      // Le c# n'est pas encore supporté
      if (language == "c#") {
        Console.WriteLine ("axl2ccT4 for C# not yet implemented, switch to axl2cc");
        // tmp used by cea
        Console.WriteLine("WARNING:  USE CSHARP !!!!!!!!!!!!!!");
        AXLParser parser = AXLParserFactory.CreateParser(full_file_name,null);
        parser.ParseAXLFile();
        CodeGenerator generator = null;
        ModuleInfo module_info = parser.Module;
        if(module_info!=null)
            generator = new CSharpModuleGenerator(include_path, output_path, module_info);
        else
            generator = new CSharpServiceGenerator(include_path, output_path,parser.Service);
        generator.writeFile();
      }

      string file_extension = Path.GetExtension(axl_file_name);
      if(GlobalContext.Instance.Verbose){
        Console.WriteLine("EXTENSION = {0}", file_extension);
      }
      
      if (Path.GetExtension(axl_file_name) != ".axl")
        throw new ArgumentException("axlfile has to have extension '.axl'");
      
      if(GlobalContext.Instance.Verbose){
        Console.WriteLine("OUTPUT PATH " + output_path);
        Console.WriteLine("ARGS[3] INCLUDE PATH " + include_path);
        Console.WriteLine("ARGS[4] FILE NAME " + full_file_name);
      }

      // Génération du module ou service
      // On pourra à terme ajouter une option pour spécifier ce que l'on souhaite
      // générer (--module, --service ou extension *.sxl, *.mxl par exemple)
      try {

        switch(axlKind(full_file_name)) {
        case AxlKind.Module :
          GenerateModule (full_file_name, output_path); break;
        case AxlKind.Service :
          // do specific class for this
          Arcane.Axl.Xsd.Service service = DeserializeService (full_file_name);
          ServiceGenerator generator = new ServiceGenerator(service, output_path, include_path, "O.8");
          if(service.IsNotCaseOption)
            generator.GenerateStandardServiceFile_Axl();
          else
            generator.GenerateCaseOptionService(generation_mode, with_mesh, with_arcane, with_parameter_factory); 
          break;
        case AxlKind.Unknown :
          Console.WriteLine("Axl file error : no module or service tag defined");
          return 1;
        }
      
      } catch (Exception e) {
        if(e.InnerException != null) {
          Console.WriteLine ("Generation error : {0}\nInner exception : {1}",
                             e.Message, e.InnerException.Message);
        } else {
          Console.WriteLine ("Generation error (No inner exception) : {0}", e.Message);
        }
        return 1;
      }

      if (!String.IsNullOrEmpty(copy_outfile)){
        if(GlobalContext.Instance.Verbose){
          Console.WriteLine("Copy '{0}' to '{1}'", full_file_name, copy_outfile);
        }
        File.Copy(full_file_name, copy_outfile, true);
      }

      return 0;
    }
  }
}
