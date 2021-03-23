//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Xml;
using System.Xml.Linq;
using System.IO;
using System.Collections.Generic;

namespace Arcane.Curves
{
  /// <summary>
  /// Lecture des courbes d'un cas au format Gnuplot
  /// </summary>
  public class GnuplotCaseReader
  {
    class CurveReader : ICaseCurveReader
    {
      string m_name;
      string m_path;
      public CurveReader(string name,string path)
      {
        m_name = name;
        m_path = path;
      }
      public ICurve Read()
      {
        ICurve curve = GnuplotCurveReader.ReadCurve(m_name,m_path);
        return curve;
      }
    }
    
    CaseCurves m_case_curves;
    public CaseCurves CaseCurves { get { return m_case_curves; } }

    bool m_read_all;
    public GnuplotCaseReader ()
    {
      m_case_curves = new CaseCurves();
      m_read_all = false;
    }

    /// <summary>
    /// Lecture des courbes decrites dans le fichier 'time_history.xml' du repertoire \a path
    /// </summary>
    /// <param name="path">
    /// A <see cref="System.String"/>
    /// </param>
    public void ReadPath(string path)
    {
      Console.WriteLine("CHECK GNUPLOT PATH path={0}",path);
      string timehistory_file = Path.Combine(path,"time_history.xml");
      XDocument doc = XDocument.Load(timehistory_file);
      XElement root_elem = doc.Root;
      DirectoryInfo base_dir = new DirectoryInfo(path);//Directory.GetParent(path);
      DirectoryInfo gnuplot_dir = new DirectoryInfo(Path.Combine(base_dir.FullName,"gnuplot"));
      Console.WriteLine("BASE_DIR={0}",base_dir.FullName);
      int nb_curve = 0;
      List<string> curves_name = new List<string>();
      foreach(XElement ce in root_elem.Elements("curve")){
        string name = ce.Attribute("name").Value;
        curves_name.Add(name);
      }
      //GnuplotCurveReader curve_reader = new GnuplotCurveReader();
      foreach(string name in curves_name){
        //Console.WriteLine("NAME={0}",name);
        if (m_read_all){
          ICurve curve = GnuplotCurveReader.ReadCurve(name,Path.Combine(gnuplot_dir.FullName,name));
          //Console.WriteLine("Nb_Point={0}",curve.NbPoint);
          m_case_curves.AddCurve(new BasicCaseCurve(curve));
        }
        else{
          ICaseCurveReader curve_reader = new CurveReader(name,Path.Combine(gnuplot_dir.FullName,name));
          m_case_curves.AddCurve(new BasicCaseCurve(name,curve_reader));
        }
        ++nb_curve;
      }

      Console.WriteLine("Nb curve={0}",nb_curve);
    }
  }
}
