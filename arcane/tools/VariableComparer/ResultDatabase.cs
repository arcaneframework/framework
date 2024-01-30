//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.IO;
using System.Collections.Generic;
using Newtonsoft.Json;
namespace Arcane.VariableComparer
{
  class CommonVariableInfo
  {
    public VariableMetaData MetaData;
    public int NbOccurence;
    internal CommonVariableInfo(VariableMetaData metadata)
    {
      MetaData = metadata;
      NbOccurence = 1;
    }
  }
  public class ArcaneJSONDataBaseInfo
  {
    public int Version;
    public int NbPart;
    public string DataCompressor;
    public int DataCompressorMinSize = 512;
  }

  public class ResultDatabase
  {
    const string DB_FILENAME = "arcane_acr_db.json";

    string m_base_path;
    public string BasePath { get { return m_base_path; } }

    int m_nb_part;
    IResultDatabasePart[] m_parts;

    Dictionary<string, VariableMetaData> m_item_variables;
    public IDictionary<string, VariableMetaData> Variables { get { return m_item_variables; } }

    //! Non nul si on utilise la version des comparaisons utilisant DB_FILENAME
    ArcaneJSONDataBaseInfo m_arcane_db_info;

    /// <summary>
    /// Cree une reference sur une base de resultat stockee sur le chemin \a base_path
    /// </summary>
    public ResultDatabase(string base_path)
    {
      m_base_path = base_path;
      m_item_variables = new Dictionary<string, VariableMetaData>();
    }

    public void ReadDatabase()
    {
      _CheckAndReadJSONDataBase();
      bool use_v3 = false;
      if (m_arcane_db_info != null) {
        m_nb_part = m_arcane_db_info.NbPart;
        use_v3 = true;
      }
      else
        _ReadNbPart();
      m_parts = new IResultDatabasePart[m_nb_part];
      for (int i = 0; i < m_nb_part; ++i) {
        IResultDatabasePart rpart = null;
        if (use_v3)
          rpart = new ResultDatabasePartV3(this, i, m_arcane_db_info);
        else
          rpart = new ResultDatabasePart(this, i);
        rpart.Read();
        m_parts[i] = rpart;
      }
      _ComputeItemVariables();
    }

    public double[] ReadVariableAsRealArray(string varname)
    {
      int total_size = 0;
      foreach (var part in m_parts) {
        VariableDataInfo vdi = part.VariablesDataInfo[varname];
        total_size += vdi.NbBaseElement;
      }

      double[] values = new double[total_size];

      int array_index = 0;
      foreach (var part in m_parts) {
        VariableDataInfo vdi = part.VariablesDataInfo[varname];
        part.ReadVariableAsRealArray(varname, values, array_index);
        array_index += vdi.NbBaseElement;
      }

      return values;
    }

    //! Valeur du hash des valeurs de la variable \a varname pour la comparaison
    public string GetComparisonHashValue(string varname)
    {
      // Seule la partie 0 contient la valeur du hash.
      VariableDataInfo vdi = m_parts[0].VariablesDataInfo[varname];
      return vdi.ComparisonHashValue;
    }

    class JSONDataBaseObject
    {
#pragma warning disable 0649
      public ArcaneJSONDataBaseInfo ArcaneCheckpointRestartDataBase;
#pragma warning restore 0649

    }
    void _CheckAndReadJSONDataBase()
    {
      m_arcane_db_info = null;
      // Regarde si le fichier décrivant la base de données est présent.
      // Si c'est le cas, cela signifie qu'on utilise la version 3+ du format
      // de 'BasicReaderWriter'.
      string db_filename = Path.Combine(m_base_path, DB_FILENAME);
      //Console.WriteLine($"Check read JSON Database {db_filename} base_path={m_base_path}");
      if (!File.Exists(db_filename))
        return;
      Console.WriteLine($"Reading database file {db_filename}");
      string s = File.ReadAllText(db_filename);
      var o = JsonConvert.DeserializeObject<JSONDataBaseObject>(s);
      if (o == null)
        throw new VariableComparerException($"Can not parse JSON database '{db_filename}'");
      var oinfo = o.ArcaneCheckpointRestartDataBase;
      if (oinfo == null)
        throw new VariableComparerException($"Can not parse JSON database (second part) '{db_filename}'");
      Console.WriteLine($"Version={oinfo.Version} NbPart={oinfo.NbPart} DataCompressor={oinfo.DataCompressor}");
      m_arcane_db_info = oinfo;
    }
    private void _ReadNbPart()
    {
      string info_path = Path.Combine(m_base_path, "infos.txt");
      string values = File.ReadAllText(info_path);
      int nb_part = int.Parse(values);
      if (nb_part <= 0)
        throw new VariableComparerException("Bad values for number of part");
      //Console.WriteLine("NB_PART={0}",nb_part);
      m_nb_part = nb_part;
    }

    private void _ComputeItemVariables()
    {
      Dictionary<string, CommonVariableInfo> nb_variables_occurence = new Dictionary<string, CommonVariableInfo>();
      foreach (IResultDatabasePart data_part in m_parts) {
        Console.WriteLine("PARSING PART = {0}", data_part.Part);
        IDictionary<string, VariableDataInfo> saved_infos = data_part.VariablesDataInfo;
        foreach (VariableMetaData var_meta_data in data_part.MetaData.Variables) {
          // Il faut que ce soit une variable du maillage
          if (String.IsNullOrEmpty(var_meta_data.ItemFamilyName))
            continue;
          // Il ne faut pas que ce soit une variable partielle
          if (!String.IsNullOrEmpty(var_meta_data.ItemGroupName))
            continue;
          // Il faut que ce soit une variable scalaire sur le maillage
          if (var_meta_data.Dimension != 1)
            continue;
          // Il faut que ce soit une variable dont la donnée est 'Real'
          if (var_meta_data.DataType != "Real")
            continue;
          string fname = var_meta_data.FullName;
          bool is_written = saved_infos.ContainsKey(fname);
          if (!is_written) {
            //Console.WriteLine("Variable '{0}' is not used",fname);
            continue;
          }
          //Console.WriteLine("Adding in part name={0}",fname);
          if (nb_variables_occurence.ContainsKey(fname))
            ++nb_variables_occurence[fname].NbOccurence;
          else
            nb_variables_occurence.Add(fname, new CommonVariableInfo(var_meta_data));
        }
      }
      int total = 0;
      foreach (KeyValuePair<string, CommonVariableInfo> pair in nb_variables_occurence) {
        if (pair.Value.NbOccurence == m_nb_part) {
          m_item_variables.Add(pair.Key, pair.Value.MetaData);
          //Console.WriteLine("ADD VARIABLE: name='{0}'",pair.Value.MetaData.FullName);
        }
        ++total;
      }
      Console.WriteLine("TOTAL='{0}' '{1}'", total, m_item_variables.Count);
    }


  }
}
