//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.IO;
using System.Collections.Generic;

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
  
  public class ResultDatabase
  {
    string m_base_path;
    public string BasePath { get { return m_base_path; } }
    
    int m_nb_part;
    ResultDatabasePart[] m_parts;
    
    Dictionary<string,VariableMetaData> m_item_variables;
    public IDictionary<string,VariableMetaData> Variables { get { return m_item_variables; } }
    
    /// <summary>
    /// Cree une reference sur une base de resultat stockee sur le chemin \a base_path
    /// </summary>
    public ResultDatabase(string base_path)
    {
      m_base_path = base_path;
      m_item_variables = new Dictionary<string,VariableMetaData>();
    }
    
    public void ReadDatabase()
    {
      _ReadNbPart();
      m_parts = new ResultDatabasePart[m_nb_part];
      for( int i=0; i<m_nb_part; ++i ){
        m_parts[i] = new ResultDatabasePart(this,i);
        m_parts[i].Read();
      }
      _ComputeItemVariables();
    }

    public double[] ReadVariableAsRealArray(string varname)
    {
      int total_size = 0;
      foreach(ResultDatabasePart part in m_parts){
        VariableDataInfo vdi = part.VariablesDataInfo[varname];
        total_size += vdi.NbBaseElement;
      }
      
      double[] values = new double[total_size];
      
      int array_index = 0;
      foreach(ResultDatabasePart part in m_parts){
        VariableDataInfo vdi = part.VariablesDataInfo[varname];
        part.ReadVariableAsRealArray(varname,values,array_index);
        array_index += vdi.NbBaseElement;
      }

      return values;
    }
    
    private void _ReadNbPart()
    {
      string info_path = Path.Combine(m_base_path,"infos.txt");
      string values = File.ReadAllText(info_path);
      int nb_part = int.Parse(values);
      if (nb_part<=0)
        throw new ApplicationException("Bad values for number of part");
      //Console.WriteLine("NB_PART={0}",nb_part);
      m_nb_part = nb_part;
    }
    
    private void _ComputeItemVariables()
    {
      Dictionary<string,CommonVariableInfo> nb_variables_occurence = new Dictionary<string,CommonVariableInfo>();
      foreach(ResultDatabasePart data_part in m_parts){
        Console.WriteLine("PARSING PART = {0}",data_part.Part);
        IDictionary<string,VariableDataInfo> saved_infos = data_part.VariablesDataInfo;
        foreach(VariableMetaData var_meta_data in data_part.MetaData.Variables){
          // Il faut que ce soit une variable du maillage
          if (String.IsNullOrEmpty(var_meta_data.ItemFamilyName))
            continue;
          // Il ne faut pas que ce soit une variable partielle
          if (!String.IsNullOrEmpty(var_meta_data.ItemGroupName))
            continue;
          // Il faut que ce soit une variable scalaire sur le maillage
          if (var_meta_data.Dimension!=1)
            continue;
          // Il faut que ce soit une variable dont la donnée est 'Real'
          if (var_meta_data.DataType!="Real")
            continue;
          string fname = var_meta_data.FullName;
          bool is_written = saved_infos.ContainsKey(fname);
          if (!is_written){
            //Console.WriteLine("Variable '{0}' is not used",fname);
            continue;
          }
          //Console.WriteLine("Adding in part name={0}",fname);
          if (nb_variables_occurence.ContainsKey(fname))
            ++nb_variables_occurence[fname].NbOccurence;
          else
            nb_variables_occurence.Add(fname,new CommonVariableInfo(var_meta_data));
        }
      }
      int total = 0;
      foreach(KeyValuePair<string,CommonVariableInfo> pair in nb_variables_occurence){
        if (pair.Value.NbOccurence==m_nb_part){
          m_item_variables.Add(pair.Key,pair.Value.MetaData);
          //Console.WriteLine("ADD VARIABLE: name='{0}'",pair.Value.MetaData.FullName);
        }
        ++total;
      }
      Console.WriteLine("TOTAL='{0}' '{1}'",total,m_item_variables.Count);
    }
  }
}
