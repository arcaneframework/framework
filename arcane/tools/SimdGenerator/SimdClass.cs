//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
﻿using System;
using System.Collections.Generic;

namespace SimdGenerator
{
  public class SimdClass : ISimdClass
  {
    public static ISimdClass CurrentType;

    public SimdType SimdType { get; private set; }

    public bool IsEmulated { get; private set; }

    public string SimdName { get; private set; }

    public string ClassName { get; private set; }

    public int DoubleVectorLength { get; private set; }

    public int Int32IndexLength { get; private set; }

    public int NbNativeVector { get; private set; }

    Dictionary<BinaryOperation,string> m_binary_operation_names;
    Dictionary<UnaryOperation,string> m_unary_operation_names;

    string m_from_scalar_func_name;

    public SimdClass (SimdType type)
    {
      SimdType = type;
      NbNativeVector = 1;
      m_binary_operation_names = new Dictionary<BinaryOperation, string> ();
      m_unary_operation_names = new Dictionary<UnaryOperation, string> ();
      switch (type) {
      case SimdType.Emulated:
        _SetEmulated ();
        break;
      case SimdType.SSE:
        _SetSSE ();
        break;
      case SimdType.AVX:
        _SetAVX ();
        break;
      case SimdType.AVX512:
        _SetAVX512 ();
        break;
      default:
        throw new NotSupportedException (String.Format ("Simd type {0}", type));
      }
    }

    void _SetEmulated ()
    {
      SimdName = "EMUL";
      IsEmulated = true;
      ClassName = "EMULSimdReal";
      DoubleVectorLength = 2;
      Int32IndexLength = 2;
      m_from_scalar_func_name = "EmulatedSimdReal";
    }

    void _SetSSE ()
    {
      SimdName = "SSE";
      ClassName = "SSESimdReal";
      DoubleVectorLength = 2;
      Int32IndexLength = 2;
      NbNativeVector = 2;
      m_from_scalar_func_name = "_mm_set1_pd";
      _Add (BinaryOperation.Add, "_mm_add_pd");
      _Add (BinaryOperation.Sub, "_mm_sub_pd");
      _Add (BinaryOperation.Mul, "_mm_mul_pd");
      _Add (BinaryOperation.Div, "_mm_div_pd");
      _Add (BinaryOperation.Min, "_mm_min_pd");
      _Add (BinaryOperation.Max, "_mm_max_pd");
    }

    void _SetAVX ()
    {
      SimdName = "AVX";
      ClassName = "AVXSimdReal";
      DoubleVectorLength = 4;
      Int32IndexLength = 4;
      NbNativeVector = 1;
      m_from_scalar_func_name = "_mm256_set1_pd";
      _Add (BinaryOperation.Add, "_mm256_add_pd");
      _Add (BinaryOperation.Sub, "_mm256_sub_pd");
      _Add (BinaryOperation.Mul, "_mm256_mul_pd");
      _Add (BinaryOperation.Div, "_mm256_div_pd");
      _Add (BinaryOperation.Min, "_mm256_min_pd");
      _Add (BinaryOperation.Max, "_mm256_max_pd");
      _Add (UnaryOperation.SquareRoot, "_mm256_sqrt_pd");
    }

    void _SetAVX512 ()
    {
      SimdName = "AVX512";
      ClassName = "AVX512SimdReal";
      DoubleVectorLength = 8;
      Int32IndexLength = 8;
      NbNativeVector = 1;
      m_from_scalar_func_name = "_mm512_set1_pd";
      _Add (BinaryOperation.Add, "_mm512_add_pd");
      _Add (BinaryOperation.Sub, "_mm512_sub_pd");
      _Add (BinaryOperation.Mul, "_mm512_mul_pd");
      _Add (BinaryOperation.Div, "_mm512_div_pd");
      _Add (BinaryOperation.Min, "_mm512_min_pd");
      _Add (BinaryOperation.Max, "_mm512_max_pd");
      _Add (UnaryOperation.SquareRoot, "_mm512_sqrt_pd");
    }

    public string OpName (BinaryOperation op)
    {
      return m_binary_operation_names [op];
    }

    public string OpName(UnaryOperation op) {
      string s;
      if (m_unary_operation_names.TryGetValue (op, out s))
        return s;
      return null;
    }

    public string FromScalar (string arg)
    {
      if (String.IsNullOrEmpty (m_from_scalar_func_name))
        throw new ApplicationException ("Null FromScalar function name");
      return m_from_scalar_func_name + "(" + arg + ")";
    }

    void _Add (BinaryOperation op, string name)
    {
      m_binary_operation_names.Add (op, name);
    }

    void _Add (UnaryOperation op, string name)
    {
      m_unary_operation_names.Add (op, name);
    }
  }
}

