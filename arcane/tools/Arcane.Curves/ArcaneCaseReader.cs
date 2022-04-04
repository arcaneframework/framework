//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.IO;
using System.Collections.Generic;
using System.Xml.Linq;

//TODO
// - Faire un reader pour toutes les courbes du cas, et charger les temps uniquement au premier
// appel (faire cela thread-safe).
// - Gerer correctement la fermeture du flux contenant les courbes.
// - Avec .NET 4.0, utiliser si possible les fichiers mappees (MemoryMappedFile je crois)
namespace Arcane.Curves
{
  /// <summary>
  /// Lecture des courbes d'un cas au format Arcane (fichier .acv)
  /// </summary>
  public class ArcaneCaseReader
  {
    /// <summary>
    /// Infos sur l'emplacement de la courbe dans le fichier.
    /// </summary>
    class CurveInfo
    {
      string m_curve_name;
      public string CurveName { get { return m_curve_name; } }

      Int32 m_iteration_len;
      public Int32 IterationLength { get { return m_iteration_len; } }

      Int64 m_iteration_offset;
      public Int64 IterationOffset { get { return m_iteration_offset; } }

      Int64 m_data_offset;
      public Int64 DataOffset { get { return m_data_offset; } }

      Int32 m_data_len;
      public Int32 DataLength { get { return m_data_len; } }

      Int64 m_xoffset;
      public Int64 XOffset { get { return m_xoffset; } }

      Int32 m_sub_size;
      public Int32 SubSize { get { return m_sub_size; } }

      public void SerializeFromXml(XElement elem)
      {
        m_curve_name = elem.Attribute("name").Value;
        m_data_offset = Int64.Parse(elem.Attribute("values-offset").Value);
        m_data_len = Int32.Parse(elem.Attribute("values-size").Value);
        m_sub_size = Int32.Parse(elem.Attribute("sub-size").Value);

        XAttribute xattr = elem.Attribute("x-offset");
        if (xattr != null) {
          m_xoffset = Int64.Parse(xattr.Value);
        } else {
          m_iteration_len = Int32.Parse(elem.Attribute("iterations-size").Value);
          m_iteration_offset = Int64.Parse(elem.Attribute("iterations-offset").Value);
        }
      }
    }

    /// <summary>
    /// Informations concernant la partie XML d'un cas.
    /// La partie XML est celle qui contient les méta-données du cas
    /// (liste et nom des courbes, type des courbes, ...)
    /// </summary>
    class CaseXmlInfos
    {
      XDocument m_document;
      /// <summary>
      /// Document XML
      /// </summary>
      /// <value>The document.</value>
      public XDocument Document { get { return m_document; } }

      Int64 m_offset;
      /// <summary>
      /// Offset dans le fichier du début du document.
      /// </summary>
      public Int64 XOffset { get { return m_offset; } }

      Int32 m_length;
      /// <summary>
      /// Longueur du document
      /// </summary>
      /// <value>The length.</value>
      public Int32 Length { get { return m_length; } }

      public CaseXmlInfos(XDocument doc, Int64 xoffset, Int32 length)
      {
        m_document = doc;
        m_offset = xoffset;
        m_length = length;
      }
    }

    class ArcaneCaseCurves : CaseCurves
    {
      ReaderAdapter m_adapter;
      internal ArcaneCaseCurves(ReaderAdapter adapter)
      {
        m_adapter = adapter;
      }
      public override void Dispose()
      {
        base.Dispose();
        m_adapter.Close();
      }
    }

    /// <summary>
    /// Adapteur pour lire un fichier soit via un tableau en memoire, soit un flux
    /// </summary>
    class ReaderAdapter
    {
      // Si m_all_bytes est non nul, on considere que le fichier
      // est en memoire et m_stream n'est pas utilise.
      Byte [] m_all_bytes;
      Stream m_stream;
      Int64 m_length;
      BinaryReader m_binary_reader;
      object m_lock;

      public Int64 Length { get { return m_length; } }

      public ReaderAdapter(Byte [] all_bytes)
      {
        m_all_bytes = all_bytes;
        m_length = all_bytes.Length;
      }

      public ReaderAdapter(Stream stream, long length)
      {
        m_stream = stream;
        m_binary_reader = new BinaryReader(stream);
        m_length = length;
        m_lock = new object();
        //m_all_bytes = all_bytes;
      }

      public ArraySegment<Byte> GetView(long offset, int count)
      {
        //Console.WriteLine("GetView offset={0} count={1}",offset,count);
        if (m_all_bytes != null) {
          // Si presence d'un tableau, on est sur que l'offset est sur 32 bits
          return new ArraySegment<Byte>(m_all_bytes, (int)offset, count);
        }
        Byte [] bytes = null;
        lock (m_lock) {
          m_stream.Seek(offset, SeekOrigin.Begin);
          bytes = m_binary_reader.ReadBytes(count);
        }
        return new ArraySegment<Byte>(bytes, 0, count);
      }

      public void Close()
      {
        if (m_stream != null) {
          m_stream.Close();
          m_stream = null;
        }
      }
    }

    /// <summary>
    /// Lecteur pour une courbe dans le format 'acv'.
    /// </summary>
    class CurveReader : ICaseCurveReader
    {
      // Si m_times n'est pas nul, on considere que les abscisses de la courbes
      // sont des iterations et correspondantes aux indexes dans ce tableau \a times.

      CurveInfo m_curve_info;
      ReaderAdapter m_reader;
      RealArray m_times;
      ICurve m_curve;
      int m_sub_index;
      int m_sub_size;
      public CurveReader(CurveInfo curve_info, ReaderAdapter reader, RealArray times, int sub_index, int sub_size)
      {
        m_curve_info = curve_info;
        m_reader = reader;
        m_times = times;
        m_sub_index = sub_index;
        m_sub_size = sub_size;
      }
      public ICurve Read()
      {
        if (m_curve != null)
          return m_curve;
        m_curve = _Read();
        m_reader = null;
        m_times = null;
        return m_curve;
      }

      ICurve _Read()
      {
        CurveInfo ci = m_curve_info;
        string curve_name = ci.CurveName;

        // TODO Dans le cas ou m_sub_size est supérieur à 1, optimiser le reader
        // pour ne pas relire à chaque sous-courbe les données ni les itérations.

        // Lecture des valeurs
        Int64 data_offset = ci.DataOffset;
        Int32 data_len = ci.DataLength;
        RealArray data = new RealArray(data_len);
        ArraySegment<Byte> data_segment = m_reader.GetView(data_offset, data_len * sizeof(double));
        //Console.WriteLine("CURVE name={0} offset={1} len={2} xoffset={3}",curve_name,data_offset,data_len,ci.XOffset);
        data.View.ReadBytes(data_segment.Array, data_segment.Offset);
        // Dans le cas d'une courbe avec plusieurs valeurs, 'data' contient les valeurs
        // de toutes les courbes sous la forme data[i][j] avec 'i' l'itération et 'j' l'index de la courbe.
        if (m_sub_size > 1) {
          int data2_len = data_len / m_sub_size;
          RealArray data2 = new RealArray(data2_len);
          int sub_index = m_sub_index;
          int sub_size = m_sub_size;
          for (int i = 0; i < data2_len; ++i)
            data2 [i] = data [(i * sub_size) + sub_index];
          data = data2;
          curve_name = curve_name + "_" + m_sub_index.ToString();
        }
        // Lecture des temps physiques des iterations
        Int64 x_offset = ci.XOffset;
        if (x_offset != 0) {
          // Contient directement la liste des valeurs des abscisses
          RealArray x_data = new RealArray(data_len);
          ArraySegment<Byte> x_data_segment = m_reader.GetView(x_offset, data_len * sizeof(double));
          x_data.View.ReadBytes(x_data_segment.Array, x_data_segment.Offset);
          return new BasicCurve(curve_name, x_data, data);
        }

        Int32 iteration_len = ci.IterationLength;
        Int64 iteration_offset = ci.IterationOffset;
        Int32Array iterations = new Int32Array(iteration_len);
        ArraySegment<Byte> iteration_segment = m_reader.GetView(iteration_offset, iteration_len * sizeof(int));
        iterations.View.ReadBytes(iteration_segment.Array, iteration_segment.Offset);

        ICurve cv = Utils.CreateCurve(curve_name, iterations.ConstView, data, m_times.ConstView);
        iterations.Dispose();
        return cv;
      }
    }

    CaseCurves m_case_curves;
    public CaseCurves CaseCurves { get { return m_case_curves; } }

    int m_file_version;
    int FileVersion { get { return m_file_version; } }

    CaseReaderSettings m_settings;

    /// <summary>
    /// Constructeur
    /// </summary>
    private ArcaneCaseReader(CaseReaderSettings settings)
    {
      m_settings = settings;
      m_case_curves = new CaseCurves();
    }

    /// <summary>
    /// Créé et lit les courbes au format Arcane
    /// </summary>
    /// <param name="path">Répertoire où se trouve le fichier de courbe</param>
    public static ArcaneCaseReader CreateFromFile(string path)
    {
      return CreateFromFile(path, new CaseReaderSettings());
    }

    /// <summary>
    /// Créé et lit les courbes au format Arcane
    /// </summary>
    /// <param name="path">Répertoire où se trouve le fichier de courbe</param>
    public static ArcaneCaseReader CreateFromFile(string path, CaseReaderSettings settings)
    {
      ArcaneCaseReader reader = new ArcaneCaseReader(settings);
      reader._ReadFile(path);
      return reader;
    }
    /// <summary>
    /// Renomme les courbes contenues dans le fichier de courbe \a path et réécrit
    /// le fichier 'curves.acv' correspondant.
    /// </summary>
    /// <param name="path">Path.</param>
    /// <param name="renamer">Renamer.</param>
    public static void RenameCurves(string path,ICurveRenamer renamer)
    {
      ArcaneCaseReader case_reader = new ArcaneCaseReader(new CaseReaderSettings());
      ReaderAdapter reader = case_reader._CreateReaderAdapter(path);
      CaseXmlInfos doc_info = case_reader._ReadXmlInfos(reader);
      if (case_reader.FileVersion != 2)
        throw new ArgumentException("Bad file version for rename curves (only version 2 is allowed)");
      XDocument doc = doc_info.Document;
      XElement curve_elem = doc.Element("curves");
      bool has_change = false;
      List<XElement> elements_to_remove = new List<XElement>();
      foreach(var e in curve_elem.Elements("curve")) {
        XAttribute name_attr = e.Attribute("name");
        string current_name = (string)name_attr;
        string new_name = renamer.Rename(current_name);
        if (new_name == "(null)") {
          // Cas spécial indiquant qu'on souhaite supprimer la courbe.
          elements_to_remove.Add(e);
          Console.WriteLine("CURVE Remove name name={0}", current_name);
          has_change = true;
        }
        else if (new_name != current_name) {
          name_attr.Value = new_name;
          Console.WriteLine("CURVE Change name old_name={0} new_name={1}", current_name, new_name);
          has_change = true;
        }
      }
      if (!has_change)
        return;
      // Supprime les courbes demandées.
      foreach (XElement e in elements_to_remove)
        e.Remove();

      // Écrit le document XML à la fin du fichiers de courbes.
      {
        string xml_string = doc.ToString();
        Byte [] xml_bytes = System.Text.Encoding.UTF8.GetBytes(xml_string);
        FileStream new_file = new FileStream(path, FileMode.Open, FileAccess.Write, FileShare.ReadWrite);
        new_file.Seek(doc_info.XOffset, SeekOrigin.Begin);
        new_file.Write(xml_bytes, 0, xml_bytes.Length);
        long buf_length = xml_bytes.Length;
        byte[] xoffset_bytes = BitConverter.GetBytes((Int64)doc_info.XOffset);
        byte[] length_bytes = BitConverter.GetBytes((Int64)buf_length);
        new_file.Write(xoffset_bytes, 0, xoffset_bytes.Length);
        new_file.Write(length_bytes, 0, length_bytes.Length);
        new_file.SetLength(new_file.Position);
        new_file.Close();
      }

    }

    /// <summary>
    /// Lecture des courbes du fichier \a path.
    /// Les courbes existantes sont supprimées avant lecture.
    /// </summary>
    /// <param name="path">
    /// A <see cref="System.String"/> Nom du fichier contenant les courbes du cas
    /// </param>
    void ReadFile(string path)
    {
      DateTime t1 = DateTime.Now;
      _ReadFile(path);
      DateTime t2 = DateTime.Now;
      TimeSpan diff = t2 - t1;
      Console.WriteLine("TIME_TO_READ={0} (in s)", diff.TotalSeconds);
    }

    public static ArcaneCaseReader CreateFromMemory(Byte [] bytes)
    {
      return CreateFromMemory(bytes, new CaseReaderSettings());
    }

    public static ArcaneCaseReader CreateFromMemory(Byte [] bytes, CaseReaderSettings settings)
    {
      ArcaneCaseReader reader = new ArcaneCaseReader(settings);
      reader._ReadBytes(bytes);
      return reader;
    }

    /// <summary>
    /// Lecture des courbes contenues dans le tableau \a bytes.
    /// Ce tableau doit avoir ete lu a partir d'un fichier .acv.
    /// Les courbes existantes sont supprimees avant lecture.
    /// </summary>
    /// <param name="bytes">
    /// A <see cref="Byte[]"/>
    /// </param>
    void _ReadBytes(Byte [] bytes)
    {
      ReaderAdapter reader = new ReaderAdapter(bytes);
      _Read(reader);
    }

    void _ReadFile(string path)
    {
			ReaderAdapter reader = _CreateReaderAdapter(path);
			m_case_curves = new ArcaneCaseCurves(reader);
      _Read(reader);
    }

    ReaderAdapter _CreateReaderAdapter(string path)
    {
      FileInfo file_info = new FileInfo(path);
      long file_length = file_info.Length;
      // Si le fichier est en dessous d'une certaine taille, le lit entierement
      // (pour l'instant, pour test, lit partiellement)
      bool use_file = true;
      if (file_length > 10000000)
        use_file = true;
      //use_file = false;
      ReaderAdapter reader = null;
      if (m_case_curves != null)
        m_case_curves.Dispose();
      if (use_file) {
        FileStream s = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        reader = new ReaderAdapter(s, file_length);
      } else {
        Byte [] all_bytes = File.ReadAllBytes(path);
        reader = new ReaderAdapter(all_bytes);
      }
      return reader;
    }

    void _CheckHeader(ArraySegment<Byte> header_segment)
    {
      Byte [] header = header_segment.Array;
      int offset = header_segment.Offset;
      // Nombre magique pour etre sur qu'il s'agit d'un fichier de courbe arcane
      if (header [offset] != 'A' || header [offset + 1] != 'C' || header [offset + 2] != 'V' || header [offset + 3] != 122)
        throw new ApplicationException("Bad header");
      byte v0 = header [offset + 4];
      m_file_version = (int)v0;
      // Verifie que la conversion little-endian/big-endian est correcte
      Int32 indianness = BitConverter.ToInt32(header, offset + 8);
      Int32 wanted_indianness = 0x01020304;
      if (indianness != wanted_indianness) {
        Console.WriteLine("READ={0} wanted={1}", indianness, wanted_indianness);
        throw new ApplicationException("Bad correspondance little endian - big endian");
      }
    }
    CaseXmlInfos _ReadXmlInfos(ReaderAdapter reader)
    {
      // Verifie qu'il s'agit bien d'un fichier de courbes Arcane.
      _CheckHeader(reader.GetView(0, 12));
      // Les deux derniers champs du fichier contiennent la position et la longueur
      // de la chaîne de caractères contenant le descripteur XML. Avec la version 1 du fichier,
      // ces champs sont de type Int32. Avec la version 2, ce sont des Int64.
      // La version 2 permet aux fichiers de courbes de faire plus de 2Go. Ce format
      // est celui utilisé par Arcane par défaut depuis 2010.

      // Les deux derniers Int32 du fichier contiennent la position et la longueur
      // de la chaine de caractere contenant le descripteur XML.
      Int64 xml_offset = 0;
      Int32 xml_length = 0;
      if (m_file_version == 2) {
        Int64 xml_info_pos = reader.Length - 16;
        ArraySegment<Byte> xml_info = reader.GetView(xml_info_pos, 16);
        xml_offset = BitConverter.ToInt64(xml_info.Array, xml_info.Offset);
        xml_length = (int)BitConverter.ToInt64(xml_info.Array, xml_info.Offset + 8);
      } else if (m_file_version == 1) {
        Int64 xml_info_pos = reader.Length - 8;
        ArraySegment<Byte> xml_info = reader.GetView(xml_info_pos, 8);
        xml_offset = BitConverter.ToInt32(xml_info.Array, xml_info.Offset);
        xml_length = BitConverter.ToInt32(xml_info.Array, xml_info.Offset + 4);
      } else
        throw new ApplicationException(String.Format("Bad 'curves.acv' version '{0}", m_file_version));
      if (Utils.VerboseLevel > 1)
        Console.WriteLine("XML_INFOS={0} {1}", xml_offset, xml_length);

      // Lecture de la partie XML contenant la description des courbes
      ArraySegment<Byte> xml_string = reader.GetView(xml_offset, xml_length);

      // Le fichier généré par Arcane ajoute un '\0' terminal à la chaîne contenant
      // le XML. A partir de Mono 5.0 et pour le framework .NET, cela n'est pas
      // toléré. On l'enlève donc si on le trouve.
      if (xml_length > 0) {
        byte last_byte = xml_string.Array [xml_string.Offset + xml_length - 1];
        if (Utils.VerboseLevel > 2)
          Console.WriteLine("LAST_BYTE={0}", (int)last_byte);
        if (last_byte == 0)
          --xml_length;
      }
      string xml_text = System.Text.UTF8Encoding.UTF8.GetString(xml_string.Array, xml_string.Offset, xml_length);
      if (Utils.VerboseLevel > 2) {
        Console.WriteLine("XML_TEXT='{0}'", xml_text);
      }
      XDocument doc = XDocument.Parse(xml_text);
      var doc_info = new CaseXmlInfos(doc, xml_offset, xml_length);
      return doc_info;
    }
    void _Read(ReaderAdapter reader)
    {
      XDocument doc = _ReadXmlInfos(reader).Document;
      XElement curve_elem = doc.Element("curves");

      // Lectures des temps de chaque iteration
      Int32 times_len = Int32.Parse(curve_elem.Attribute("times-size").Value);
      Int64 times_offset = Int64.Parse(curve_elem.Attribute("times-offset").Value);
      //Console.WriteLine("CURVE_FILE: time_offset={0} time_len={1}",times_offset,times_len);
      RealArray times = null;
      if (times_len>0){
        times = new RealArray((int)times_len);
        ArraySegment<Byte> times_segment = reader.GetView(times_offset,times_len*sizeof(double));
        times.View.ReadBytes(times_segment.Array,times_segment.Offset);
      }

      m_case_curves = new CaseCurves();
      bool do_print = Utils.VerboseLevel > 0;
      foreach(XElement ce in curve_elem.Elements("curve")){
        CurveInfo ci = new CurveInfo();
        ci.SerializeFromXml(ce);
        // Ne traite pas les courbes vides
        if (ci.DataLength<=0)
          continue;
        if (do_print)
          Console.WriteLine("CHECK_CURVE name={0} sub_size={1}",ci.CurveName,ci.SubSize);
        if (ci.SubSize == 1) {
          CurveReader cr = new CurveReader (ci, reader, times, 0, 1);
          BasicCaseCurve bcc = new BasicCaseCurve (ci.CurveName, cr);
          if (do_print)
            Console.WriteLine("ADD_CURVE name={0}",ci.CurveName);
          m_case_curves.AddCurve (bcc);
        }
        else if (ci.SubSize > 1) {
          switch(m_settings.MultiCurveBehavior){
          case MultiCurveBehaviour.Ignore:
            break;
          case MultiCurveBehaviour.ReadAsSeveralMonoValue:            
          // Les courbes multi-valeurs sont gérées en considérant qu'il s'agit de N courbes scalaires.
            int nb_sub_size = ci.SubSize;
            for (int i = 0; i < nb_sub_size; ++i) {
              CurveReader cr = new CurveReader (ci, reader, times, i, nb_sub_size);
              BasicCaseCurve bcc = new BasicCaseCurve (ci.CurveName + "_" + i, cr);
              if (do_print)
                Console.WriteLine("ADD_MULTI_CURVE name={0} index={1}",ci.CurveName,i);
              m_case_curves.AddCurve (bcc);
            }
            break;
          }
        }
      }

      if (do_print)
        Console.WriteLine("NB_CURVE={0}",m_case_curves.Curves.Count);
    }
  }
}
