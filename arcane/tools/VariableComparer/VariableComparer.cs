//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System.Collections.Generic;

namespace Arcane.VariableComparer
{
  public class VariableComparer
  {
    string m_reference_path;
    string m_target_path;

    List<VariableMetaData> m_common_variables;
    /// <summary>
    /// Liste des variables scalaires du maillage qui existent sur la reference et la cible
    /// </summary>
    public List<VariableMetaData> CommonVariables { get { return m_common_variables; } }

    ResultDatabase m_reference_base;
     ResultDatabase m_target_base;

    public VariableComparer(string reference_path,string target_path)
    {
      m_reference_path = reference_path;
      m_target_path = target_path;
      m_common_variables = new List<VariableMetaData>();
    }

    public void ReadDatabase()
    {
      m_reference_base = new ResultDatabase(m_reference_path);
      m_reference_base.ReadDatabase();

      m_target_base = new ResultDatabase(m_target_path);
      m_target_base.ReadDatabase();

      foreach(KeyValuePair<string,VariableMetaData> var in m_target_base.Variables){
        if (m_reference_base.Variables.ContainsKey(var.Key))
          m_common_variables.Add(var.Value);
      }
    }

    public void ReadVariableAsRealArray(VariableMetaData var,out double[] target_values,out double[] reference_values)
    {
      target_values = m_target_base.ReadVariableAsRealArray(var.FullName);
      reference_values = m_reference_base.ReadVariableAsRealArray(var.FullName);
    }
    //! Retourne un tuple contenant les hash de comparaison de la variable \a var pour la référence et la cible
    public (string,string) GetComparisonHashValue(VariableMetaData var)
    {
      string ref_hash = m_reference_base.GetComparisonHashValue(var.FullName);
      string target_hash = m_target_base.GetComparisonHashValue(var.FullName);
      return (ref_hash,target_hash);
    }
  }
}
