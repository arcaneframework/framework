//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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
  public class CaseOptionMultiEnumT<EnumType> : CaseOptionMultiEnum
  {
    EnumType[] m_values;
    int[] m_int_values;
    Integer m_nb_element;

    public CaseOptionMultiEnumT(Arcane.CaseOptionBuildInfo opt,string type_name)
    : base(opt,type_name)
    {
    }

    public EnumType[] Values { get { return m_values; } }

    protected override void _allocate(Integer size)
    {
      m_values = new EnumType[size];
      m_int_values = new int[size];
      m_nb_element = size;
    }
    
    protected override Integer _nbElem()
    {
      return m_nb_element;
    }

    protected override void _setOptionValue(Integer index,int v)
    {
      m_int_values[index] = v;
      m_values[index] = (EnumType)Enum.ToObject(typeof(EnumType),v);
    }

    protected override int _optionValue(Integer index)
    {
      return m_int_values[index];
    }
  }
}
