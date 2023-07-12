using System;
using System.Xml;
using System.Xml.Serialization;
using System.IO;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using System.Linq;

namespace LawCompiler
{

  public enum MultiThreadMode
  {
    Sequential,
    ArcaneTBB,
    Kokkos
  }
  
  public enum InferenceMode
  {
    None,
    ONNX
  }

  class MainClass
  {
    public static void Main (string[] args)
    {
      bool help = false;
      bool verbose = false;
      bool debug = false;
      string xml = null;
      string path = null;
      string multiThreading = "Sequential";
      string inference = "None";
			
      var p = new NDesk.Options.OptionSet () {
				{ "h|?|help", "Show help message and exit", (v) => help = v != null },
                { "verbose", "Verbose mode", (v) => verbose = v != null },
                { "debug", "Debug mode", (v) => debug = v != null },
				{ "law=", "XML describing law file", (v) => xml = v },
				{ "path=", "Path where C++ law files are generated", (v) => path = v },
				{ "multi-threading=", "multi threading mode", (v) => multiThreading = v },
				{ "inference=", "inference mode", (v) => inference = v },
			};

      p.Parse (args);

      MultiThreadMode multiThreadModeValue;
      if(Enum.TryParse(multiThreading, out multiThreadModeValue))
        Console.WriteLine ("Generating for: {0}", multiThreadModeValue);          
      else
        Console.WriteLine ("Mode {0} is not a valid value, generating for Sequential", multiThreading);
        
      InferenceMode inferenceModeValue;
      if(Enum.TryParse(inference, out inferenceModeValue))
        Console.WriteLine ("Generating for inference with target: {0}", inferenceModeValue);

      if (xml == null) {
        Console.WriteLine ("xml option is mandatory");
        help = true;
      }
      if (path == null) {
        Console.WriteLine ("path option is mandatory");
        help = true;
      }
			
      if (help) {
        Console.WriteLine ("usage: law.exe [OPTION]+\n");
        Console.WriteLine ("Valid options include:");
        p.WriteOptionDescriptions (Console.Out);
        Environment.Exit (-1);
      }
			
      if (verbose) {
        Console.WriteLine ("Law generator xml -> c++");
        if(debug) {
           Console.WriteLine ("debug mode: activated");
        }
        Console.WriteLine ("law file : {0}", xml);
        Console.WriteLine ("law directory : {0}", path);
      }

      Directory.CreateDirectory (path);

			path += "/";

      law law = null;
      var serializer = new XmlSerializer (typeof(law));
      if (verbose) {
        Console.ForegroundColor = ConsoleColor.Blue;
        Console.WriteLine ("parse data-model from xml file");
        Console.ResetColor ();
      }
      try {
        using (StreamReader stream = File.OpenText(xml)) {
          try {
            using (XmlReader reader = XmlReader.Create(stream, Xml.CreateXmlSettings (xml))) {
              law = serializer.Deserialize (reader) as law;
            }
          } catch(Exception e) {
            Console.Error.WriteLine("XmlReader Unexpected Exception : {0}", e.Message);
            Console.Error.WriteLine("Exception : {0}", e.GetType().FullName);
            var inner = e.InnerException;
            if(inner != null) {
              Console.Error.WriteLine("XmlReader Unexpected Exception : {0}", inner.Message);
              Console.Error.WriteLine("Exception : {0}", inner.GetType().FullName);
            }
            Environment.Exit(-1);
          } 
        }
      } catch(Exception e) {
        Console.Error.WriteLine("StreamReader Unexpected Exception : {0}", e.Message);
        Environment.Exit(-1);
      } 

      {
        var lawname = Path.GetFileNameWithoutExtension(xml);
        if (law.name != lawname) {
          Console.Error.WriteLine("Filename [xxx.law] should be the name of law in xml : {0} != {1}", law.name, lawname);
          Environment.Exit(-1);
        }
      }

      // generation du fichier *_law.h
      var gen = new LawT4 (law, debug, multiThreadModeValue, inferenceModeValue);
      var file = path + "/" + law.name + "_law.h";
      using (StreamWriter writer = new StreamWriter (file)) {
        writer.Write (gen.TransformText ());
      }
    }
	}
}