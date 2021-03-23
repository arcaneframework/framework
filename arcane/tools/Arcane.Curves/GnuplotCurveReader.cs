//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.IO;

namespace Arcane.Curves
{
  public class GnuplotCurveReader
  {
    public static ICurve ReadCurve(string curve_name,string filename)
    {
      using(TextReader reader = File.OpenText(filename)){
        return ReadCurve(curve_name,reader);
      }
    }

    public static ICurve ReadCurve(string curve_name,TextReader reader)
    {
      //TODO: verifier que le fichier n'est pas trop gros.
      //TODO: avec .NET 4.0, utiliser les enumerator sur les lignes.
      //string[] lines = File.ReadAllLines(filename);
      //TextReader reader = File.OpenText(filename);
      char[] whitespace_char = new char[]{' ','\t'};
      RealArray cx = new RealArray();
      RealArray cy = new RealArray();
      int current_line = 0;
      string line;
      while( (line=reader.ReadLine())!=null){
        ++current_line;
        if (String.IsNullOrEmpty(line))
          continue;
        line = line.Trim();
        if (line[0] == '#')
          continue;

        // convertion de la chaine en une liste de sous-chaines
        // (recherche du caractere separateur :espace ou virgule)
        string[] strs = line.Split(whitespace_char);
        if (strs.Length < 2)
          throw new ApplicationException(String.Format("Can not parse line '{0}'", line));
        string xstr = strs[0];
        string ystr = strs[1];
        double xlu = 0.0;
        double ylu = 0.0;
        // fichier a au moins deux colonnes
        bool x_ok = Double.TryParse(xstr,out xlu);
        if (!x_ok)
          throw new ApplicationException(String.Format("Can not parse '{0}' to Double",xstr));
        bool y_ok = Double.TryParse(ystr,out ylu);
        if (!y_ok)
          throw new ApplicationException(String.Format("Can not parse '{0}' to Double",ystr));
        cx.Add(xlu);
        cy.Add(ylu);
      }

      BasicCurve curve = new BasicCurve(curve_name,cx,cy);
      return curve;
    }
  }
}
