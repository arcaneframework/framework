//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Xml;
using Integer = System.Int32;

namespace Arcane.VariableComparer
{
  public enum eDataType
  {
    DT_Byte = 0, //!< Donnée de type octet
    DT_Real, //!< Donnée de type réel
    DT_Int32, //!< Donnée de type entier 32 bits
    DT_Int64, //!< Donnée de type entier 64 bits
    DT_String, //!< Donnée de type chaîne de caractère unicode
    DT_Real2, //!< Donnée de type vecteur 2
    DT_Real3, //!< Donnée de type vecteur 3
    DT_Real2x2, //!< Donnée de type tenseur 3x3
    DT_Real3x3, //!< Donnée de type tenseur 3x3
    DT_Unknown  //!< Donnée de type inconnu ou non initialisé
  }

  public class VariableDataInfo
  {
    String m_full_name;
    Integer m_nb_dimension;
    Integer m_dim1_size;
    Integer m_dim2_size;
    Integer m_nb_element;
    Integer m_nb_base_element;
    Integer m_dimension_array_size;
    bool m_is_multi_size;
    eDataType m_base_data_type;
    Integer m_memory_size;
    Int64 m_file_offset;
    string m_item_group_name;
    string m_comparison_hash_value;

    public String FullName { get { return m_full_name; }}
    public Integer NbDimension { get { return m_nb_dimension; }}
    public Integer Dim1Size { get { return m_dim1_size; }}
    public Integer Dim2Size { get { return m_dim2_size; }}
    public Integer NbElement { get { return m_nb_element; }}
    public Integer NbBaseElement { get { return m_nb_base_element; }}
    public Integer DimensionArraySize { get { return m_dimension_array_size; }}
    public bool IsMultiSize { get { return m_is_multi_size; }}
    public eDataType BaseDataType { get { return m_base_data_type; }}
    public Integer MemorySize { get { return m_memory_size; }}
    public Int64 FileOffset { get { return m_file_offset; }}
    public string ItemGroupName { get { return m_item_group_name; } }
    public string ComparisonHashValue { get { return m_comparison_hash_value; } }

    public VariableDataInfo(string full_name,XmlElement element)
    {
      m_full_name = full_name;
      m_nb_dimension = _readInteger(element,"nb-dimension");
      m_dim1_size = _readInteger(element,"dim1-size");
      m_dim2_size = _readInteger(element,"dim2-size");
      m_nb_element = _readInteger(element,"nb-element");
      m_nb_base_element = _readInteger(element,"nb-base-element");
      m_dimension_array_size = _readInteger(element,"dimension-array-size");
      m_is_multi_size = _readBool(element,"is-multi-size");
      m_base_data_type = (eDataType)_readInteger(element,"base-data-type");
      m_memory_size = _readInteger(element,"memory-size");
      m_file_offset = _readInt64(element,"file-offset");

      // Peut etre nul si la variable est sur toute la famille
      m_item_group_name = element.GetAttribute("item-group-name");

      // A partir de Arcane 3.12
      // Peut etre nul si le calcul du hash n'est pas activé
      m_comparison_hash_value = element.GetAttribute("comparison-hash");
    }

    private Integer _readInteger(XmlElement node,string attr_name)
    {
      return int.Parse(node.GetAttribute(attr_name));
    }

    Int64 _readInt64(XmlElement node,string attr_name)
    {
      return Int64.Parse(node.GetAttribute(attr_name));
    }

    bool _readBool(XmlElement node,string attr_name)
    {
     string s = node.GetAttribute(attr_name);
      if (s=="0" || s=="false")
        return false;
      return true;
    }
  }
}
