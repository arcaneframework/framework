using System;
using System.Xml;
using System.Xml.Serialization;
using System.Xml.Schema;
using System.IO;
using System.Text;
using System.Reflection;

namespace LawCompiler
{
  public class Xml
  {
    public Xml ()
    {
    }

    static public string Beautify(XmlDocument doc)
    {
      StringBuilder sb = new StringBuilder();
      XmlWriterSettings settings = new XmlWriterSettings
      {
        Indent = true,
        IndentChars = "  ",
        NewLineChars = "\r\n",
        NewLineHandling = NewLineHandling.Replace
      };
      using (XmlWriter writer = XmlWriter.Create(sb, settings)) {
        doc.Save(writer);
      }
      return sb.ToString();
    }

    public static XmlReaderSettings CreateXmlSettings (string file)
    {
      var settings = new XmlReaderSettings ();
      Assembly assembly = Assembly.GetExecutingAssembly();
      using (Stream stream = assembly.GetManifestResourceStream("Law.xsd")) {
        XmlSchema schema = XmlSchema.Read(stream, null);
        settings.Schemas.Add(schema);
      }
      settings.ValidationType = ValidationType.Schema;
      settings.ValidationFlags = 
        XmlSchemaValidationFlags.ProcessIdentityConstraints |
        XmlSchemaValidationFlags.ReportValidationWarnings;
      settings.ValidationEventHandler += delegate(object sender, ValidationEventArgs args) {
        var reader = sender as XmlTextReader;
        Console.Error.WriteLine ("Error detected during validation :");
        Console.Error.WriteLine ("  File     : {0}", file);
        Console.Error.WriteLine ("  Line     : {0}", args.Exception.LineNumber);
        Console.Error.WriteLine ("  Position : {0}", args.Exception.LinePosition);
        if(args.Exception.Message.Contains("Invalid start element")) {
          Console.Error.WriteLine ("  Message  : Invalid xml node '{0}'", reader.Name);
        } else if(args.Exception.Message.Contains("Attribute declaration was not found")) {
          Console.Error.WriteLine ("  Message  : Invalid xml attribute '{0}'", reader.Name);
        } else if(args.Exception.Message.Contains("Element declaration for") &&
                  args.Exception.Message.Contains("is missing")) {
          Console.Error.WriteLine ("  Message  : Invalid xml node '{0}'", reader.Name);
        } else if(args.Exception.Message.Contains("Required attribute") &&
                  args.Exception.Message.Contains("was not found")) {
          Console.Error.WriteLine ("  Message  : Missing xml attribute of element '{0}'", reader.Name);
        } else {
          Console.Error.WriteLine ("  Message  : Xml node '{0}' has failed value '{1}'", 
                                   reader.Name, reader.Value);
          if(args.Exception.InnerException != null)
            Console.Error.WriteLine ("  Details  : {0}", args.Exception.InnerException.Message);
        }
        Console.Error.WriteLine ();
        Console.Error.WriteLine ();
        Console.Error.WriteLine("[Internal exception catched : {0}]", args.Exception.Message);
        Console.Error.WriteLine ();
        Console.Error.WriteLine("Please, check your xml files and reconfigure!");
        Console.Error.WriteLine ();
        throw new Exception ();
      };
      return settings;
    }
  }
}

