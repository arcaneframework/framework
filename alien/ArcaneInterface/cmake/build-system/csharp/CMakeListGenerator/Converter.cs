using System;
using System.IO;
using System.Xml;
using System.Xml.Linq;
using System.Xml.Serialization;
using System.Xml.Schema;

namespace CMakeListGenerator
{
  public class Converter
  {
    public Converter (bool force_overwrite, String libname, String path, String outpath, bool recurse)
    {
      System.IO.Directory.CreateDirectory (outpath);

      Makefile makefile = Initialize<Makefile> (Path.Combine(path,"config.xml"), "CMakeListGenerator.Makefile.xsd");

      String cmake_file = Path.Combine (outpath, "CMakeLists.txt");

      if (System.IO.File.Exists (cmake_file)) {
        if (force_overwrite) {
          if(!recurse) {
            Console.WriteLine("WARNING: target '{0}' is going to be overwritten as requested by user", cmake_file);
          }
        } else {
          throw new CodeException(String.Format("Cannot overwrite existing '{0}' file; use -f to force overwrite", cmake_file));  
        }
      }

      using (StreamWriter writer = new StreamWriter(cmake_file)) {

        var cmake = new CMakeList (makefile, libname);

        if (path != outpath) {
          cmake.SrcPath = path + "/";
          cmake.OutSrcPath = outpath + "/";
        }

        writer.Write (cmake.TransformText ());
      }

      if (recurse && makefile.directories != null) {
        foreach(var directory in makefile.directories) {
          var new_path = Path.Combine(path,directory.Value).Replace("\\","/");
          var new_outpath = Path.Combine(outpath, directory.Value).Replace("\\", "/");
          new Converter(force_overwrite, libname, new_path, new_outpath, true);
        }
      }
    }

    private T Initialize<T> (string file, string xsd)
      where T : class, new()
    {
      T model = null;
      
      var serializer = new XmlSerializer (typeof(T));
      
      try {
        using (StreamReader stream = System.IO.File.OpenText(file)) {
          using (XmlReader reader = XmlReader.Create(stream, Xml.CreateXmlSettings (xsd,file))) {
            model = serializer.Deserialize (reader) as T;
          }
        }
      } catch(Exception e) {
        if(e.InnerException == null)
          Console.Error.WriteLine("Unexpected Exception : {0}", e.Message);
        else {
          var exception = e.InnerException as XmlException;
          Console.Error.WriteLine("\n* VALIDATION ERROR :");
          Console.Error.WriteLine("  File    : {0}", file);
          Console.Error.WriteLine("  Line    : {0}", exception.LineNumber);
          Console.Error.WriteLine("  Column  : {0}", exception.LinePosition);
          if(exception.Message.Contains("Expected >")) {
            Console.Error.WriteLine("  Message : Invalid closing xml node <\\>");
          } else {
            Console.Error.WriteLine("  Unexpected Exception");
          }
          Console.Error.WriteLine(" [Internal exception catched : {0}]", exception.Message);
          Console.Error.WriteLine();
          Console.Error.WriteLine("Please, check your xml files and reconfigure!");
        }
      }
      
      return model;
    }
  }
}

