//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.IO;
using System.Collections.Generic;
using System.Xml.Linq;

namespace Arcane.Curves
{
  /// <summary>
  /// Ecrit les courbes d'un cas au format Arcane.
  /// </summary>
  public class ArcaneCaseWriter
  {
    public static void WriteCurves(CaseCurves case_curves,string path)
    {
      ArcaneCaseWriter acw = new ArcaneCaseWriter(case_curves);
      acw._WriteAll(path);
    }
    
    FileStream m_file_stream;
    BinaryWriter m_binary_writer;
    List<ICurve> m_curves;
    XElement m_curves_element;
    
    ArcaneCaseWriter(CaseCurves case_curves)
    {
      m_curves = new List<ICurve>();
      foreach(ICaseCurve cc in case_curves.Curves)
        m_curves.Add(cc.Read());
    }
    
    void _WriteAll(string path)
    {
      m_curves_element = new XElement("curves",
                                      new XAttribute("times-size",0),
                                      new XAttribute("times-offset",0)
                                      );

      Console.WriteLine("Write arcane curve file format path={0}",path);
      string file_name = Path.Combine(path,"curves.acv");
      //TODO: Verifier troncature du fichier s'il existe deja.
      using(FileStream fs = File.OpenWrite(file_name)){
        m_file_stream = fs;
        m_binary_writer = new BinaryWriter(fs);
        _WriteHeader();
        foreach(ICurve cc in m_curves){
          _WriteCurve(cc);
        }
        Console.WriteLine("XML1={0}",m_curves_element.ToString());
        Console.WriteLine("XML2={0}",m_curves_element.Value);
        string xml_string = m_curves_element.ToString();
        //string xml_text = System.Text.UTF8Encoding.UTF8.GetString(xml_string.Array,xml_string.Offset,xml_length);
        Byte[] xml_bytes  = System.Text.Encoding.UTF8.GetBytes(xml_string);
        Int64 xml_bytes_pos = _WriteArray(xml_bytes);
        Int64 xml_bytes_length = xml_bytes.Length;
        // Ecrit a la fin du fichier l'offset des donnees XML
        // et ensuite la longueur de xml_bytes.
        // Ces deux valeurs doivent etre de type Int64 pour la version 2 du fichier
        m_binary_writer.Write(xml_bytes_pos);
        m_binary_writer.Write(xml_bytes_length);
      }
    }
    
    // Ecrit l'en tete (12 octets)
    void _WriteHeader()
    {
      // Le numero magique
      Byte[] magic_bytes = new Byte[] { (Byte)'A', (Byte)'C', (Byte)'V', 122 };
      m_binary_writer.Write(magic_bytes);
      
      // La version
      // Actuellement on utilise la version 2.0 qui permet de le support des fichiers
      // dont la taille dépasse 2Go (32 bits).
      Byte[] version_bytes = new Byte[] { 2, 0, 0, 0 };
      m_binary_writer.Write(version_bytes);

      // L'indianness
      Int32 indianness = 0x01020304;
      m_binary_writer.Write(indianness);
    }
    
    
    void _WriteCurve(ICurve curve)
    {
      Int64 x_offset = _WriteArray(curve.X);
      Int64 y_offset = _WriteArray(curve.Y);
      XElement celem = new XElement("curve",
                                    new XAttribute("name",curve.Name),
                                    new XAttribute("sub-size",1),
                                    new XAttribute("values-offset",y_offset),
                                    new XAttribute("values-size",curve.NbPoint),
                                    new XAttribute("x-offset",x_offset)
                                    );
      Console.WriteLine("CELEM={0}",celem);
      m_curves_element.Add(celem);             
    }
    
    Int64 _WriteArray(RealConstArrayView values)
    {
      long offset = m_file_stream.Position;
      foreach(double v in values)
        m_binary_writer.Write(v);
      return _FromOffset(offset);
    }
    
    Int64 _WriteArray(Byte[] values)
    {
      long offset = m_file_stream.Position;
      m_file_stream.Write(values,0,values.Length);
      return _FromOffset(offset);
    }

    Int64 _FromOffset(long offset)
    {
      return offset;
    }
  }
}

