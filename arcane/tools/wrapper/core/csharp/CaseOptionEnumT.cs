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
  public class CaseOptionEnumT<EnumType> : CaseOptionEnum
  {
    EnumType m_value;
    int m_int_value;

    public CaseOptionEnumT(Arcane.CaseOptionBuildInfo opt,string type_name)
    : base(opt,type_name)
    {
    }

    public EnumType Value { get { return m_value; } }

    public EnumType value() { return m_value; }

    protected override void _setOptionValue(int v)
    {
      m_int_value = v;
      m_value = (EnumType)Enum.ToObject(typeof(EnumType),v);
    }

    protected override int _optionValue()
    {
      return m_int_value;
    }
  }
}
