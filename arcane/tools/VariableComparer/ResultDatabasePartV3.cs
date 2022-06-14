//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.IO;
using System.Xml;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Newtonsoft.Json;

namespace Arcane.VariableComparer
{
  [StructLayout(LayoutKind.Sequential)]
  struct BasicReaderWriterDatabaseEpilogFormat
  {
    // La taille de cette structure ne doit pas être modifiée sous peine
    // de rendre le format incompatible. Pour supporter des évolutions, on fixe
    // une taille de 128 octets, soit 16 'Int64'
    public const int STRUCT_SIZE = 128;
    // Version de l'epilogue. A ne pas confondre avec la version du fichier
    // qui est dans l'en-tête.
    Int32 m_version;
    Int32 m_padding0;
    Int64 m_padding1;
    Int64 m_padding2;
    Int64 m_padding3;
    Int64 m_json_data_info_file_offset;
    public Int64 JSONDataInfoFileOffset { get { return m_json_data_info_file_offset; } }
    Int64 m_json_data_info_size;
    public Int64 JSONDataInfoSize { get { return m_json_data_info_size; } }
    Int64 m_remaining_padding0;
    Int64 m_remaining_padding1;
    Int64 m_remaining_padding2;
    Int64 m_remaining_padding3;
    Int64 m_remaining_padding4;
    Int64 m_remaining_padding5;
    Int64 m_remaining_padding6;
    Int64 m_remaining_padding7;
    Int64 m_remaining_padding8;
    Int64 m_remaining_padding9;
    public void ReadFromStream(BinaryReader reader)
    {
      m_version = reader.ReadInt32();
      m_padding0 = reader.ReadInt32();
      m_padding1 = reader.ReadInt64();
      m_padding2 = reader.ReadInt64();
      m_padding3 = reader.ReadInt64();
      m_json_data_info_file_offset = reader.ReadInt64();
      m_json_data_info_size = reader.ReadInt64();
      Console.WriteLine($"JSON Info offset={m_json_data_info_file_offset} size={m_json_data_info_size}");
    }
  }

  public class ACRDataBaseVariableInfo
  {
    public string Name;
    public Int64 FileOffset;
    public Int64[] Extents;
  }
  class ACRDataBase
  {
    public class ACRDataBaseJSON
    {
      // Ce champs est désérialisé depuis le JSON.
      #pragma warning disable 0649
      public ACRDataBaseVariableInfo[] Data;
      #pragma warning restore 0649

      public Dictionary<string, ACRDataBaseVariableInfo> Dict = new Dictionary<string, ACRDataBaseVariableInfo>();

      public void FillDictionary()
      {
        Dict.Clear();
        foreach (var x in Data) {
          Dict.Add(x.Name, x);
        }
      }
    }
    string m_filename;
    Int64 m_file_length = 0;
    FileStream m_file_stream;
    BinaryReader m_file_binary_reader;
    BasicReaderWriterDatabaseEpilogFormat m_epilog;
    ACRDataBaseJSON m_json_database;
    internal ACRDataBaseJSON JSONDataBase { get { return m_json_database; } }
    internal IDeflater Deflater { get; set; }
    public ACRDataBase(string filename)
    {
      m_filename = filename;
    }
    public void OpenRead()
    {
      // TODO: vérifier l'en-tête.
      FileInfo fi = new FileInfo(m_filename);
      m_file_length = fi.Length;
      const int epilog_size = BasicReaderWriterDatabaseEpilogFormat.STRUCT_SIZE;
      if (m_file_length < epilog_size)
        throw new IOException($"File '{m_filename}' is too short");
      // Lit l'épilogue
      m_file_stream = new FileStream(m_filename, FileMode.Open, FileAccess.Read);
      m_file_stream.Seek(m_file_length - epilog_size, SeekOrigin.Begin);
      m_file_binary_reader = new BinaryReader(m_file_stream);
      m_epilog.ReadFromStream(m_file_binary_reader);

      // Lit les informations JSON de la base de données.
      m_file_stream.Seek(m_epilog.JSONDataInfoFileOffset, SeekOrigin.Begin);
      byte[] bytes = new byte[m_epilog.JSONDataInfoSize];
      m_file_stream.Read(bytes, 0, bytes.Length);
      string json_string = System.Text.Encoding.UTF8.GetString(bytes);
      //Console.WriteLine("JSON STRING={0}", json_string);
      ACRDataBaseJSON o = JsonConvert.DeserializeObject<ACRDataBaseJSON>(json_string);
      if (o == null)
        throw new VariableComparerException("Can not convert json infos from database to json object");
      m_json_database = o;
      m_json_database.FillDictionary();
    }
    ACRDataBaseVariableInfo _GetAndCheckDimension1(string key_name)
    {
      ACRDataBaseVariableInfo vinfo = m_json_database.Dict[key_name];
      if (vinfo.Extents == null)
        throw new VariableComparerException($"extents for '{key_name}' is null");
      if (vinfo.Extents.Length == 0)
        throw new VariableComparerException($"extents for '{key_name}' has 0 length");
      int ex_length = vinfo.Extents.Length;
      if (ex_length != 1)
        throw new VariableComparerException($"extents for '{key_name}' should be of dimension 1 (v={ex_length}");
      return vinfo;
    }
    void _SeekFile(ACRDataBaseVariableInfo vinfo)
    {
      if (vinfo == null)
        throw new ArgumentException("Null 'vinfo'");
      if (vinfo.FileOffset < 0)
        throw new ArgumentException($"Invalid negative 'FileOffset={vinfo.FileOffset}' in 'vinfo'");
      m_file_stream.Seek(vinfo.FileOffset, SeekOrigin.Begin);
    }

    //! Lit les octests associés à la clé 'key_name' sans prendre en compte une éventuelle compression
    byte[] _ReadDirectPartAsBytes(string key_name)
    {
      ACRDataBaseVariableInfo vinfo = _GetAndCheckDimension1(key_name);
      Int64 length = vinfo.Extents[0];
      byte[] bytes = new byte[length];
      if (length < 0) {
        _SeekFile(vinfo);
        m_file_stream.Read(bytes, 0, (int)length);
      }
      return bytes;
    }
    public byte[] ReadPartAsBytes(string key_name)
    {
      ACRDataBaseVariableInfo vinfo = _GetAndCheckDimension1(key_name);
      Int64 length = vinfo.Extents[0];
      byte[] out_bytes = new byte[length];
      if (length > 0) {
        _SeekFile(vinfo);
        if (Deflater != null && length > Deflater.MinCompressSize) {
          Console.WriteLine($"Reading compressed byte array for key {key_name}");
          Int64 binary_len = m_file_binary_reader.ReadInt64();
          Byte[] input_buffer = new Byte[binary_len];
          m_file_stream.Read(input_buffer, 0, input_buffer.Length);
          Deflater.Decompress(input_buffer, out_bytes);
        }
        else
          m_file_stream.Read(out_bytes, 0, (int)length);
      }
      return out_bytes;
    }

    public double[] ReadPartAsReal(string key_name)
    {
      ACRDataBaseVariableInfo vinfo = _GetAndCheckDimension1(key_name);
      Int64 length = vinfo.Extents[0];
      double[] v = new double[length];
      if (length > 0) {
        _SeekFile(vinfo);
        Int64 buf_out_len = length * sizeof(double);
        BinaryReader reader = m_file_binary_reader;
        if (Deflater != null && length > Deflater.MinCompressSize) {
          Int64 binary_len = m_file_binary_reader.ReadInt64();
          Byte[] input_buffer = new Byte[binary_len];
          Byte[] output_buffer = new Byte[buf_out_len];
          m_file_stream.Read(input_buffer, 0, input_buffer.Length);
          Deflater.Decompress(input_buffer, output_buffer);
          reader = new BinaryReader(new MemoryStream(output_buffer));
        }

        for (int i = 0; i < length; ++i)
          v[i] = reader.ReadDouble();
      }
      return v;
    }
  }

  /// <summary>
  /// Valeurs d'un bloc de la base de resultats au format V3+
  /// </summary>
  public class ResultDatabasePartV3 : IResultDatabasePart
  {
    ResultDatabase m_database;

    public int Part { get; private set; }

    public MetaData MetaData { get; private set; }

    Dictionary<string, VariableDataInfo> m_variables_info;
    public IDictionary<string, VariableDataInfo> VariablesDataInfo { get { return m_variables_info; } }

    int m_version = 1;

    ACRDataBase m_acr_database;
    ArcaneJSONDataBaseInfo m_arcane_main_db_info;
    /// <summary>
    /// Cree le bloc numero \a part de la base \a database
    /// </summary>
    public ResultDatabasePartV3(ResultDatabase database, int part, ArcaneJSONDataBaseInfo arcane_db_info)
    {
      m_database = database;
      Part = part;
      m_arcane_main_db_info = arcane_db_info;
    }
    public void ReadVariableAsRealArray(string varname, double[] values, int array_index)
    {
      double[] file_values = m_acr_database.ReadPartAsReal(varname);
      file_values.CopyTo(values, array_index);
    }

    public void Read()
    {
      string base_path = m_database.BasePath;
      string acr_file_path = Path.Combine(base_path, String.Format("arcane_db_n{0}.acr", Part));
      m_acr_database = new ACRDataBase(acr_file_path);
      m_acr_database.OpenRead();
      if (!String.IsNullOrEmpty(m_arcane_main_db_info.DataCompressor)) {
        IDeflater d = Utils.CreateDeflater(m_arcane_main_db_info.DataCompressor);
        d.MinCompressSize = m_arcane_main_db_info.DataCompressorMinSize;
        m_acr_database.Deflater = d;
      }
      byte[] metadata_bytes = m_acr_database.ReadPartAsBytes("Global:CheckpointMetadata");
      string metadata_string = System.Text.Encoding.UTF8.GetString(metadata_bytes);
      // Supprime un éventuel '\0' terminal.
      // TODO regarder pourquoi il y a un zéro terminal
      int string_length = metadata_string.Length;
      if (string_length > 0 && metadata_string[string_length - 1] == 0)
        metadata_string = metadata_string.Substring(0, string_length - 1);
      //Console.WriteLine("CHECKPOINT_METADATA='{0}'", metadata_string);
      MetaData = new MetaData();
      MetaData.ParseString(metadata_string);
      Console.WriteLine("NB_VARIABLE={0} part={1}", MetaData.Variables.Count, Part);
      _ReadVariablesDataInfo();
    }

    void _ReadVariablesDataInfo()
    {
      byte[] metadata_bytes = m_acr_database.ReadPartAsBytes("Global:OwnMetadata");
      string metadata_string = System.Text.Encoding.UTF8.GetString(metadata_bytes);
      XmlDocument doc = new XmlDocument();
      doc.LoadXml(metadata_string);
      XmlElement doc_element = doc.DocumentElement;

      // Récupère le numéro de version. Si absent, il s'agit de la version 1.
      string version_str = doc_element.GetAttribute("version");
      if (!String.IsNullOrEmpty(version_str))
        m_version = int.Parse(version_str);
      Console.WriteLine("MetaDataVersion = {0}", m_version);
      m_variables_info = new Dictionary<string, VariableDataInfo>();
      if (m_version != 3)
        throw new VariableComparerException($"Unsupported version '{m_version}'. Valid values are '3'");

      foreach (XmlNode node in doc_element) {
        if (node.Name != "variable-data")
          continue;
        XmlElement element = node as XmlElement;
        string full_name = element.GetAttribute("full-name");
        VariableDataInfo vdi = new VariableDataInfo(full_name, element);
        m_variables_info.Add(full_name, vdi);
      }
      Console.WriteLine("NB_VAR_DATA_INFO={0}", m_variables_info.Count);
    }
  }
}
