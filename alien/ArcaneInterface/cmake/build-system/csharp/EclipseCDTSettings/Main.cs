using System;
using System.Text;
using System.Collections.Generic;
using System.IO;
using System.Xml;

namespace EclipseCDTSettings
{
  class MainClass
  {
    static private void showHelp(Mono.Options.OptionSet opt_set)
    {
      Console.WriteLine("Options :");
      opt_set.WriteOptionDescriptions(Console.Out);
      Environment.Exit(0);
    }
    
    public static string ToUpperCaseName(string s)
    {
      return s.Replace("-", "_").ToUpper();
    }
    
    public static void Main(string[] args)
    {
      bool show_help = false;
      bool verbose = false;
      Mono.Options.OptionSet opt_set = new Mono.Options.OptionSet();
      opt_set.Add("h|help|?", "help message", v => { show_help = true; });
      opt_set.Add("v|verbose", "verbose mode", v => { verbose = true; });
      string[] remaining_args = opt_set.Parse(args).ToArray();
      
      if (show_help || remaining_args.Length != 2) showHelp(opt_set);
      string pkglist_file = remaining_args[0];
      string output_file = remaining_args[1];
      
      if (verbose)
        Console.WriteLine("Loading {0} into {1}", pkglist_file, output_file);

      var document = new XmlDocument();
      document.Load(pkglist_file);

      var packages = document.SelectNodes(string.Format("//packages/package[@available='true']"));
 
      var paths = new List<String> ();
      var macros = new List<String> ();

      foreach (XmlNode package in packages)
      {
        var includes = package.SelectNodes(string.Format("include"));

        foreach (XmlNode include in includes)
        {
          paths.Add(include.InnerText);
        }

        var defines = package.SelectNodes(string.Format("flags"));

        foreach (XmlNode define in defines)
        {
          macros.Add(define.InnerText);
        }
      }

      var doc = new XmlDocument();

      doc.LoadXml (
        "<?xml version=\"1.0\" encoding=\"iso-8859-1\"?>" +
        "<cdtprojectproperties>" +
        "<section name=\"org.eclipse.cdt.internal.ui.wizards.settingswizards.IncludePaths\">" +
        "<language name=\"C++ Source File\" />" +
        "</section>" +
        "<section name=\"org.eclipse.cdt.internal.ui.wizards.settingswizards.Macros\">" +
        "<language name=\"C++ Source File\" />" +
        "</section>" +
        "</cdtprojectproperties>"
        );

      XmlNode include_node = doc.SelectSingleNode(string.Format("//cdtprojectproperties/section[@name='org.eclipse.cdt.internal.ui.wizards.settingswizards.IncludePaths']/language"));

      foreach (var path in paths) {
        XmlElement p = doc.CreateElement("includepath");
        p.InnerText = path;
        include_node.AppendChild(p);
      }

      XmlNode define_node = doc.SelectSingleNode(string.Format("//cdtprojectproperties/section[@name='org.eclipse.cdt.internal.ui.wizards.settingswizards.Macros']/language"));

      foreach (var macro in macros) {
        XmlElement p = doc.CreateElement("macro");
        XmlElement n = doc.CreateElement("name");
        XmlElement v = doc.CreateElement("value");
        n.InnerText = macro;
        p.AppendChild(n);
        p.AppendChild(v);
        define_node.AppendChild(p);
      }

      doc.Save(output_file);
    }
  }
}
