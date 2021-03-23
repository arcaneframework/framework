//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.IO;
using System.Xml;
using System.Collections.Generic;

interface IDeflater
{
  void Decompress(Byte[] input_buffer,Byte[] output_buffer);
}

/// <summary>
/// Decompresseur utilisant l'algorithme 'Bzip2'
/// </summary>
class Bzip2Deflater : IDeflater
{
  public void Decompress(Byte[] input_buffer,Byte[] output_buffer)
  {
    Stream instream = new MemoryStream(input_buffer);
    //StringWriter outstream = new StringWriter();
    Console.WriteLine("Using 'Bzip2' deflater len={0} outlen={1}",input_buffer.Length,output_buffer.Length);
    System.IO.Stream outstream = new MemoryStream(output_buffer,true);
    ICSharpCode.SharpZipLib.BZip2.BZip2.Decompress(instream,outstream,true);
  }
}

/// <summary>
/// Decompresseur utilisant l'algorithme 'LZ4'
/// </summary>
class LZ4Deflater : IDeflater
{
  public void Decompress(Byte[] input_buffer, Byte[] output_buffer)
  {
    Stream instream = new MemoryStream(input_buffer);
    //StringWriter outstream = new StringWriter();
    Console.WriteLine("Using 'LZ4' deflater len={0} outlen={1}", input_buffer.Length, output_buffer.Length);
    System.IO.Stream outstream = new MemoryStream(output_buffer, true);
    K4os.Compression.LZ4.LZ4Codec.Decode(input_buffer, 0, input_buffer.Length, output_buffer, 0, output_buffer.Length);
  }
}

namespace Arcane.VariableComparer
{
  
  /// <summary>
  /// Valeurs d'un bloc de la base de resultats
  /// </summary>
  public class ResultDatabasePart
  {
    // Cette valeur doit etre cohérente avec celle dans BasicReaderWriter.cc
    const int DEFLATE_MIN_SIZE = 512;
    ResultDatabase m_database;
    int m_part;
    public int Part { get { return m_part; } }
    
    MetaData m_metadata;
    public MetaData MetaData { get { return m_metadata; } }
    
    Dictionary<string,VariableDataInfo> m_variables_info;
    public IDictionary<string,VariableDataInfo> VariablesDataInfo { get { return m_variables_info; } }
    
    FileStream m_value_stream;
    IDeflater m_deflater;
    int m_version = 1;
    
    /// <summary>
    /// Cree le bloc numero \a part de la base \a database
    /// </summary>
    public ResultDatabasePart(ResultDatabase database,int part)
    {
      m_database = database;
      m_part = part;
    }
    void _CheckValueStream()
    {
      if (m_value_stream!=null)
        return;
       string base_path = m_database.BasePath;
     string path = Path.Combine(base_path,String.Format("var___MAIN___{0}.txt",m_part));
      m_value_stream = new FileStream(path,FileMode.Open,FileAccess.Read);
    }
    public void ReadVariableAsRealArray(string varname,double[] values,int array_index)
    {
      VariableDataInfo vdi = m_variables_info[varname];
      _CheckValueStream();
      // Il faut ajouter 4 a l'offset pour avoir la bonne valeur.
      // Par contre, je ne sais pas pourquoi. Cela fonctionne sur mono mais c'est peut-etre
      // un bug. Il faudrait faire le test sous Win32.
      Int64 file_offset = vdi.FileOffset;
      // La première valeur indique le nombre d'éléments (car il s'agit d'un tableau 1D).
      // Avec la version 1, ce nombre est sur 32 bits. Avec les version suivantes, il est sur 64 bits.
      if (m_version <= 2)
        file_offset += 4;
      else
        file_offset += 8;
      m_value_stream.Seek(file_offset,SeekOrigin.Begin);
      int nb_value = vdi.NbBaseElement;
      //Console.WriteLine("SET FILE OFFSET part={0} var={1} offset={2} n={3} name={4} pos={5}",m_part,varname,file_offset,nb_value,m_value_stream.Name,m_value_stream.Position);
#if false
      if (nb_value>1){
        for( int i=0; i<16; ++i ){
        int b = m_value_stream.ReadByte();
        Console.WriteLine("BYTE I={0} V={1}",i,b);
        }
      }
#endif
      m_value_stream.Seek(file_offset,SeekOrigin.Begin);
      BinaryReader reader = new BinaryReader(m_value_stream);
      int out_len = nb_value * sizeof(double);
      // Il n'y a compression que si la taille est superieure a DEFLATE_MIN_SIZE
      // En cas de compression, il faut d'abord lire un Int64 qui contiend
      // la taille en byte des donnees compressees. On lit ces valeurs
      // qu'on envoie au decompresseur.
      if (m_deflater!=null && out_len>DEFLATE_MIN_SIZE){
        Int64 binary_len = reader.ReadInt64();
        Byte[] input_buffer = new Byte[binary_len];
        Byte[] output_buffer = new Byte[out_len];
        m_value_stream.Read(input_buffer,0,input_buffer.Length);
        m_deflater.Decompress(input_buffer,output_buffer);
        reader = new BinaryReader(new MemoryStream(output_buffer));
      }
      for( int i=0; i<nb_value; ++i ){
        double z = reader.ReadDouble();
        //if (i<5)
        //Console.WriteLine("Z={0}",z);
        values[array_index+i] = z; 
      }
    }
    
    public void Read()
    {
      string base_path = m_database.BasePath;
      string metadata_path = Path.Combine(base_path,String.Format("metadata-{0}.txt",m_part));
      string metadata_str = File.ReadAllText(metadata_path);
      m_metadata = new MetaData();
      m_metadata.ParseString(metadata_str);
      Console.WriteLine("NB_VARIABLE={0} part={1}",m_metadata.Variables.Count,m_part);
      _ReadVariablesDataInfo();
    }

    void _ReadVariablesDataInfo()
    {
      string base_path = m_database.BasePath;
      string metadata_path = Path.Combine(base_path,String.Format("own_metadata_{0}.txt",m_part));
      //string metadata_str = File.ReadAllText(metadata_path);
      XmlDocument doc = new XmlDocument();
      doc.Load(metadata_path);
      XmlElement doc_element = doc.DocumentElement;
      string deflater_service = doc_element.GetAttribute("deflater-service");
      if (!String.IsNullOrEmpty(deflater_service)){
        if (deflater_service=="Bzip2"){
          m_deflater = new Bzip2Deflater();
        }
        else if (deflater_service == "LZ4") {
          m_deflater = new LZ4Deflater();
        }
        else
          throw new ApplicationException("Can only handle 'Bzip2' or 'LZ4' deflater-service");
      }
      // Récupère le numéro de version. Si absent, il s'agit de la version 1.
      string version_str = doc_element.GetAttribute("version");
      if (!String.IsNullOrEmpty(version_str))
        m_version = int.Parse(version_str);
      Console.WriteLine("MetaDataVersion = {0}", m_version);
      m_variables_info = new Dictionary<string,VariableDataInfo>();

      foreach (XmlNode node in doc_element){
        if (node.Name != "variable-data")
          continue;
        XmlElement element = node as XmlElement;
        string full_name = element.GetAttribute("full-name");
        VariableDataInfo vdi = new VariableDataInfo(full_name,element);
        m_variables_info.Add(full_name,vdi);
      }
      Console.WriteLine("NB_VAR_DATA_INFO={0}",m_variables_info.Count);
    }
  }
}
