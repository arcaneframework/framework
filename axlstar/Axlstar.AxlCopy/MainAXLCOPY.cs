//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Xml;

namespace Arcane.Axl
{
  //! Copie un fichier axl en intégrant les inclusions.
  public class MainAXLCOPY
  {
    public MainAXLCOPY()
    {
    }

    public static int MainExec(string[] args)
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
      Console.WriteLine("COPY (V2): axl_file_name='{0}' output_file_name='{1}'",axl_file_name,output_file_name);

      AXLParser parser = AXLParserFactory.CreateParser(axl_file_name,null);
      parser.ParseAXLFileForDocumentation();

      // Avec Mono 5.0 qui utilise la version Microsoft du framework .NET
      // il n'est pas possible de faire parser.WriteTo() sans lever une exception
      // (erreur de l'automate interne). Pour éviter cela, on créé un nouveau
      // document et on clone l'élément racine de 'parser.Document' dans ce
      // nouveau document.
      // Cela permet en outre de forcer l'encodage du document de sortie à utf-8.
      XmlDocument doc = parser.Document;
      Encoding encoding = Encoding.UTF8;
      XmlWriterSettings settings = new XmlWriterSettings ();
      settings.Encoding = encoding;
      settings.NewLineHandling = NewLineHandling.None;
      XmlDocument new_doc = new XmlDocument ();
      XmlNode new_root = new_doc.ImportNode (doc.DocumentElement, true);
      new_doc.AppendChild (new_root);
      using (StreamWriter sw = new StreamWriter (new FileStream (output_file_name, FileMode.Create), encoding)) {
        using (XmlWriter xw = XmlWriter.Create (sw, settings)) {
          new_doc.Save (xw);
          xw.Close ();
        }
        sw.Close ();
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
