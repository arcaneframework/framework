//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Generator.cs                                                (C) 2000-2018 */
/*                                                                           */
/* Programme principal du script de génération de code à partir de           */
/* fichiers AXL.                                                             */
/* Version 2.0 utilisant la génération par template T4                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
using System;
using System.IO;
using System.Xml;
using System.Xml.Serialization;
using Arcane.Axl;

namespace Axlstar.Axl
{
  public class Generator
  {
    const string GENERATOR_VERSION = "1.0";

    private enum AxlKind
    {
      Module,
      Service,
      Unknown
    }

    private void PrintUsage ()
    {
      Console.WriteLine ("Usage: axl.exe [-i|--header header] [-o|--output output_path]" +
                        " [-l|--lang c++|c#] [--no-restore] [--copy outfile] [--verbose={ |0|1}] axlfile");
    }

    static AxlKind axlKind (string axl)
    {
      using (var reader = XmlTextReader.Create (axl)) {
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

    static XmlReader _CreateXmlReader (string file_name)
    {
      XmlReaderSettings settings = new XmlReaderSettings ();
      settings.DtdProcessing = DtdProcessing.Parse;
      XmlNameTable name_table = new NameTable ();
      string subset = CommonEntities.Build ();
      XmlParserContext context = new XmlParserContext (name_table, null, "axl", null, null, subset, "", "", XmlSpace.Preserve);

      XmlReader reader = XmlReader.Create (file_name, settings, context);
      return reader;
    }

    static private Arcane.Axl.Xsd.Service DeserializeService (string axl)
    {
      Arcane.Axl.Xsd.Service service = null;

      try {
        var serializer = new XmlSerializer (typeof (Arcane.Axl.Xsd.Service));
        using (var reader = _CreateXmlReader (axl)) {
          service = (Arcane.Axl.Xsd.Service)serializer.Deserialize (reader);
        }
      } catch (Exception e) {
        _WriteException (e, "XML Parsing error");
        Environment.Exit (1);
      }
      return service;
    }

    static private Arcane.Axl.Xsd.Module DeserializeModule (string axl)
    {
      Arcane.Axl.Xsd.Module module = null;
      try {
        var serializer = new XmlSerializer (typeof (Arcane.Axl.Xsd.Module));
        using (var reader = _CreateXmlReader (axl)) {
          module = (Arcane.Axl.Xsd.Module)serializer.Deserialize (reader);
        }
      } catch (Exception e) {
        _WriteException (e, "XML parsing error");
        Environment.Exit (1);
      }
      return module;
    }
    static void _WriteException (Exception e, string message, int level = 0)
    {
      Console.WriteLine ("{3}: Exception type={0}, message={1}, stack_trace={2}",
                         e.GetType (), e.Message, e.StackTrace, message);
      Exception ie = e.InnerException;
      if (ie != null)
        _WriteException (ie, message, level + 1);
    }
    static private void GenerateModuleFile_Axl (Arcane.Axl.Xsd.Module module, string include_path, string output_path)
    {
      var output_case = Path.Combine (output_path, module.Name + "_axl.h");
      using (var file = new StreamWriter (output_case)) {
        var case_options = new CaseOptionsT4 (module, include_path: include_path, version: GENERATOR_VERSION);
        file.WriteLine (case_options.TransformText ());
        var base_class = new ModuleT4 (module, path: include_path, version: GENERATOR_VERSION);
        file.WriteLine (base_class.TransformText ());
      }
    }

    // TODO mutualiser plus factory
    static Arcane.Axl.Xsd.Module GenerateModule (string axl_file_name, string include_path, string output_path)
    {
      Arcane.Axl.Xsd.Module module = DeserializeModule (axl_file_name);
      module.CheckValid ();
      GenerateModuleFile_Axl (module, include_path, output_path);
      return module;
    }

    static void _SetAxlContent (string axl_file_name, Arcane.Axl.Xsd.Base xsd_base)
    {
      // Positionne la xsd_base.Content avec le contenu du fichier AXL.
      // On positionne une première fois AxlContent avant de faire la sérialisation JSON.
      // Une fois ceci fait, on repositionne AxlContent avec le contenu
      // de la sérialisation JSON.
      byte[] axl_content = File.ReadAllBytes (axl_file_name);
      //var compressed_content = LZ4.LZ4Codec.Wrap (axl_content);
      var compressed_content = axl_content;
      xsd_base.AxlContent = new FileContent (compressed_content,"LZ4");
      Newtonsoft.Json.JsonSerializer x = new Newtonsoft.Json.JsonSerializer ();
      using (StringWriter tw = new StringWriter()) {
        x.Serialize(tw, xsd_base);
        string s = tw.ToString();
        axl_content = System.Text.Encoding.UTF8.GetBytes(s);
      }
      //compressed_content = LZ4.LZ4Codec.Wrap(axl_content);
      compressed_content = axl_content;
      xsd_base.AxlContent = new FileContent(compressed_content, "LZ4");
    }
    public int Execute(string[] args)
    {
      int nb_arg = args.Length;
      if (nb_arg == 0) {
        PrintUsage();
        return 1;
      }
      string generation_mode = null;
      bool with_arcane = true;
      bool with_mesh = true;
      bool with_parameter_factory = false;
      bool with_content = false;
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
        } else if (arg == "--with-content") {
          with_content = true;
        } else if (arg == "-g" || arg == "--gen-target") {
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
        Console.WriteLine ("GENERATOR_VERSION " + GENERATOR_VERSION);
        Console.WriteLine("OUTPUT PATH " + output_path);
        Console.WriteLine("INCLUDE PATH " + include_path);
        Console.WriteLine("FILE NAME " + full_file_name);
      }

      // Génération du module ou service
      // On pourra à terme ajouter une option pour spécifier ce que l'on souhaite
      // générer (--module, --service ou extension *.sxl, *.mxl par exemple)
      try {

        switch(axlKind(full_file_name)) {
        case AxlKind.Module :
          Arcane.Axl.Xsd.Module module = DeserializeModule (axl_file_name);
          module.CheckValid ();
          if (with_content)
            _SetAxlContent (full_file_name, module);
          GenerateModuleFile_Axl (module, include_path, output_path);
          break;
        case AxlKind.Service :
          // do specific class for this
          Arcane.Axl.Xsd.Service service = DeserializeService (full_file_name);
          service.CheckValid ();
          if (with_content)
            _SetAxlContent (full_file_name, service);
          ServiceGenerator generator = new ServiceGenerator(service, include_path, output_path, GENERATOR_VERSION);
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
        _WriteException (e, "Generation error");
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
