//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections.Generic;

#if ARCANE_64BIT
using Integer = System.Int64;
#else
using Integer = System.Int32;
#endif
using Real = System.Double;

namespace Arcane
{
  public interface ICaseOptionExtentedConverter
  {
    object TryToConvert(string name);
  }

  
  public class ItemGroupExtendedConverter : ICaseOptionExtentedConverter
  {
    public interface IWrapper
    {
      IItemFamily GetFamily(IMesh mesh);
      object ConvertGroup(ItemGroup group);
    }

    ICaseMng m_case_mng;
    protected IWrapper m_wrapper;
    public ItemGroupExtendedConverter(ICaseMng case_mng)
    {
      m_case_mng = case_mng;
    }
    public ItemGroupExtendedConverter(ICaseMng case_mng,IWrapper wrapper)
    {
      m_case_mng = case_mng;
      m_wrapper = wrapper;
    }
    object ICaseOptionExtentedConverter.TryToConvert(string name)
    {
      Console.WriteLine("TRY TO CONVERT FACEGROUP name={0}",name);
      ISubDomain sub_domain = m_case_mng.SubDomain();
      IMesh mesh = sub_domain.DefaultMesh();
      ItemGroup item_group;
      if (m_wrapper!=null){
        item_group = m_wrapper.GetFamily(mesh).FindGroup(name);
        if (item_group.IsNull())
          return null;
        return m_wrapper.ConvertGroup(item_group);
      }
      item_group = mesh.FindGroup(name);
      if (item_group.IsNull())
        return null;
      return item_group;
    }
  }

  public class NodeGroupExtendedConverter : ItemGroupExtendedConverter, ItemGroupExtendedConverter.IWrapper
  {
    public NodeGroupExtendedConverter(ICaseMng case_mng) : base(case_mng)
    {
      m_wrapper = this;
    }
    public IItemFamily GetFamily(IMesh mesh)
    {
      return mesh.NodeFamily();
    }
    public object ConvertGroup(ItemGroup group)
    {
      NodeGroup g = new NodeGroup(group);
      if (g.IsNull())
        return null;
      return g;
    }
  }

  public class FaceGroupExtendedConverter : ItemGroupExtendedConverter, ItemGroupExtendedConverter.IWrapper
  {
    public FaceGroupExtendedConverter(ICaseMng case_mng) : base(case_mng)
    {
      m_wrapper = this;
    }
    public IItemFamily GetFamily(IMesh mesh)
    {
      return mesh.FaceFamily();
    }
    public object ConvertGroup(ItemGroup group)
    {
      FaceGroup g = new FaceGroup(group);
      if (g.IsNull())
        return null;
      return g;
    }
  }

  public class CellGroupExtendedConverter : ItemGroupExtendedConverter, ItemGroupExtendedConverter.IWrapper
  {
    public CellGroupExtendedConverter(ICaseMng case_mng) : base(case_mng)
    {
      m_wrapper = this;
    }
    public IItemFamily GetFamily(IMesh mesh)
    {
      return mesh.CellFamily();
    }
    public object ConvertGroup(ItemGroup group)
    {
      CellGroup g = new CellGroup(group);
      if (g.IsNull())
        return null;
      return g;
    }
  }

  public class CaseOptionExtendedT<ExtendedType> : CaseOptionExtended where ExtendedType : class
  {
    ExtendedType m_value;
    ICaseOptionExtentedConverter m_converter;

    public CaseOptionExtendedT(CaseOptionBuildInfo opt,string type_name,
                               ICaseOptionExtentedConverter converter)
    : base(opt,type_name)
    {
      m_converter = converter;
    }

    public ExtendedType Value { get { return m_value; } }

    public ExtendedType value() { return m_value; }

    protected override bool _tryToConvert(string s)
    {
      object v = m_converter.TryToConvert(s);
      m_value = (ExtendedType)v;
      return (v==null);
    }
  }
}
