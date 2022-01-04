//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.IO;

namespace Arcane.Curves
{

  public class GnuplotCurveWriter : ICurveWriter
  {
    string m_path;
    public GnuplotCurveWriter (string path)
    {
      m_path = path;
    }

    public void Write(ICurve curve)
    {
      string name = curve.Name;

      using(StreamWriter tw = File.CreateText(Path.Combine(m_path,name))){
        Console.WriteLine("CREATE FILE {0}",name);
        Write(curve,tw);
        //RealConstArrayView c_x = curve.X;
        //RealConstArrayView c_y = curve.Y;
        //int nb_value = c_x.Length;
        //for( int i=0; i<nb_value; ++i )
          //tw.WriteLine(c_x[i].ToString("E16")+" "+c_y[i].ToString("E16"));
      }
    }

    public void Write(ICurve curve,StreamWriter writer)
    {
      StreamWriter tw = writer;
      RealConstArrayView c_x = curve.X;
      RealConstArrayView c_y = curve.Y;
      int nb_value = c_x.Length;
      for( int i=0; i<nb_value; ++i )
        tw.WriteLine(c_x[i].ToString("E16")+" "+c_y[i].ToString("E16"));
    }

  }
}
