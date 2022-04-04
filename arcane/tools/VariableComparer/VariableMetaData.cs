//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;

namespace Arcane.VariableComparer
{

 public class VariableMetaData
 {
  const int PNoDump = (1 << 0);
  const int PNoNeedSync = (1 << 1);
  const int PHasTrace = (1 << 2);
  const int PSubDomainDepend = (1 << 3);
  const int PSubDomainPrivate = (1 << 4);
  const int PExecutionDepend = (1 << 5);
  const int PPrivate = (1 << 6);
  const int PTemporary = (1 << 7);
  const int PNoRestore= (1 << 8);

    private string m_base_name;
   public string BaseName { get { return m_base_name; } }

   private string m_mesh_name;
   public string MeshName { get { return m_mesh_name; } }

   private string m_item_family_name;
   public string ItemFamilyName { get { return m_item_family_name; } }

   private string m_item_group_name;
   public string ItemGroupName { get { return m_item_group_name; } }

    private int m_dimension;
   public int Dimension { get { return m_dimension; } }
   
   private string m_data_type;
   public string DataType { get { return m_data_type; } }

   private string m_full_name;
   public string FullName { get { return m_full_name; } }

   private int m_property;
   public int Property { get { return m_property; } }

    public bool IsSubDomainDepend { get { return (m_property & PSubDomainDepend)!=0; } }
    public bool IsExecutionDepend { get { return (m_property & PExecutionDepend)!=0; } }
    
    public VariableMetaData(string base_name,int dimension,string data_type,
                           string mesh_name,string item_family_name,string item_group_name,int property)
    {
      m_base_name = base_name;
      m_dimension = dimension;
      m_data_type = data_type;
      m_mesh_name = mesh_name;
      m_item_family_name = item_family_name;
      m_item_group_name = item_group_name;
      m_property = property;
      m_full_name = "";
      if (m_mesh_name!=null)
        m_full_name += m_mesh_name + "_";
      if (m_item_family_name!=null)
        m_full_name += m_item_family_name + "_";
      m_full_name += m_base_name;
      //Console.WriteLine("NEW VAR name={0} property={1} exec={2} sd={3}",base_name,property,IsExecutionDepend,IsSubDomainDepend);
    }
  }
}
