//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Xml;
using System.Collections.Generic;

namespace Arcane.VariableComparer
{
  /*!
   * \brief Contient les informations sur les données contenues dans le fichier
   */
  public class MetaData
  {
    private XmlDocument m_document = new XmlDocument();
    private IList<VariableMetaData> m_variables = new List<VariableMetaData>();
    public IList<VariableMetaData> Variables { get { return m_variables; } }

    public bool Contains(string variableName)
    {
      Console.WriteLine("Contains");
      foreach (VariableMetaData mdv in m_variables)
        if (mdv.FullName.Equals(variableName))
         return true;
      return false;
    }

    public void ParseString(string metadata)
    {
      //Console.WriteLine("META_DATA="+metadata);
      m_document.LoadXml(metadata);
      XmlNode variables_elem = m_document.DocumentElement.SelectSingleNode("variables");
      if (variables_elem == null) {
        Console.WriteLine("WARNING: no <variables> element in metadata");
        return;
      }

      foreach (XmlNode node in variables_elem){
        if (node.Name != "variable")
          continue;
        XmlElement element = node as XmlElement;
        string name = element.GetAttribute("base-name");
        int dimension = int.Parse(element.GetAttribute("dimension"));
        string data_type = element.GetAttribute("data-type");
        string item_family_name = null;
        string item_group_name = null;
        string mesh_name = null;
        if (element.HasAttribute("item-family-name"))
          item_family_name = element.GetAttribute("item-family-name");
        //if (element.HasAttribute("item-group-name"))
        item_group_name = element.GetAttribute("item-group-name");
        if (element.HasAttribute("mesh-name"))
          mesh_name = element.GetAttribute("mesh-name");
        int property = int.Parse(element.GetAttribute("property"));
        //Console.WriteLine("NAME = {0} {1} {2} mesh={3} family={4}",name,dimension,data_type,mesh_name,item_family_name);
        VariableMetaData mdv = new VariableMetaData(name, dimension, data_type, mesh_name, item_family_name,item_group_name,property);
        m_variables.Add(mdv);
      }
    }
  }
}
