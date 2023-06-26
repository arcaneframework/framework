
using System;
using System.Collections.Generic;
using System.IO;
using System.Xml;

namespace Arcane.Axl
{
  //! Copie un fichier axl en intégrant les inclusions.
  public class MainAXLCOPY
  {
    public MainAXLCOPY()
    {
    }

    public static int Main(string[] args)
    {
      MainAXLCOPY v = new MainAXLCOPY();
      return v.Execute(args);
    }

    public int Execute(string[] args)
    {
      int nb_arg = args.Length;
      if (nb_arg==0){
        _PrintUsage();
        return 1;
      }
      if (args[0]=="install"){
        List<string> largs = new List<string>();
        for( int i=1; i<nb_arg; ++i )
          largs.Add(args[i]);
        MainAXLInstall rc = MainAXLInstall.Create(largs.ToArray());
        if (rc!=null)
          rc.Execute();
        return 0;
      }

      if (nb_arg!=2) {
        _PrintUsage();
        return 1;
      }
      string axl_file_name = args[0];
      string output_file_name = args[1];
      Console.WriteLine("COPY: axl_file_name='{0}' output_file_name='{1}'",axl_file_name,output_file_name);

      AXLParser parser = AXLParserFactory.CreateParser(axl_file_name,null);
      parser.ParseAXLFileForDocumentation();
      using(XmlWriter writer = XmlWriter.Create(output_file_name)){
        parser.Document.WriteTo(writer);
      }

      return 0;
    }

    private void _PrintUsage()
    {
      Console.WriteLine("Usage: axlcopy axlfile outputfile");
      Console.WriteLine("Usage: axlcopy install ...");
    }
  }

}
