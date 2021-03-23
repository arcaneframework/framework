//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
﻿using System;
using System.Collections.Generic;
using System.Text;

namespace SimdGenerator
{
  public static class CppHelper
  {
    struct BinaryOperatorInfo
    {
      public BinaryOperatorInfo(string _func_namespace,string _func_name,string _symbol)
      {
        func_namespace = _func_namespace;
        func_name = _func_name;
        symbol = _symbol;
      }
      public BinaryOperatorInfo(string _func_name,string _symbol)
      {
        func_namespace = null;
        func_name = _func_name;
        symbol = _symbol;
      }
      internal string func_namespace;
      internal string func_name;
      internal string symbol;
    };

    struct UnaryOperatorInfo
    {
      public UnaryOperatorInfo(string _func_namespace,string _func_name,string _qualified_func_name)
      {
        func_namespace = _func_namespace;
        func_name = _func_name;
        qualified_func_name = _qualified_func_name;
      }
      internal string func_namespace;
      internal string func_name;
      internal string qualified_func_name;
    };

    static Dictionary<BinaryOperation,BinaryOperatorInfo> m_binary_operator_names;
    static Dictionary<UnaryOperation,UnaryOperatorInfo> m_unary_operator_names;

    static CppHelper ()
    {
      m_binary_operator_names = new Dictionary<BinaryOperation, BinaryOperatorInfo > ();
      _Add (BinaryOperation.Sub, "operator-","-");
      _Add (BinaryOperation.Add, "operator+","+");
      _Add (BinaryOperation.Mul, "operator*","*");
      _Add (BinaryOperation.Div, "operator/","/");
      _Add (BinaryOperation.Min, "math", "min","math::min");
      _Add (BinaryOperation.Max, "math", "max","math::max");

      m_unary_operator_names = new Dictionary<UnaryOperation, UnaryOperatorInfo> ();
      _Add (UnaryOperation.SquareRoot, "math", "sqrt", "math::sqrt");
      _Add (UnaryOperation.Exponential, "math", "exp", "math::exp");
      _Add (UnaryOperation.Log10, "math" , "log10", "math::log10");
      _Add (UnaryOperation.UnaryMinus, null, "operator-", "-");
    }

    static void _Add(BinaryOperation op,string func_name,string symbol)
    {
      m_binary_operator_names.Add (op, new BinaryOperatorInfo (func_name, symbol));
    }

    static void _Add(BinaryOperation op,string func_namespace,string func_name,string symbol)
    {
      m_binary_operator_names.Add (op, new BinaryOperatorInfo (func_namespace,func_name, symbol));
    }

    public static string OpFuncName(BinaryOperation op) {
      return m_binary_operator_names [op].func_name;
    }

    public static string OpFuncNamespace(BinaryOperation op) {
      return m_binary_operator_names[op].func_namespace;
    }

    public static string OpSymbolName(BinaryOperation op) {
      return m_binary_operator_names [op].symbol;
    }

    static void _Add(UnaryOperation op,string func_namespace,string func_name,string symbol)
    {
      m_unary_operator_names.Add (op, new UnaryOperatorInfo (func_namespace,func_name, symbol));
    }

    public static string OpFuncName(UnaryOperation op) {
      return m_unary_operator_names [op].func_name;
    }

    public static string OpFuncNamespace(UnaryOperation op) {
      return m_unary_operator_names[op].func_namespace;
    }

    public static string QualifiedFuncName(UnaryOperation op) {
      return m_unary_operator_names [op].qualified_func_name;
    }

    public static string ComputeArgs(string func_name,string arg_name,ISimdClass simd_class)
    {
      StringBuilder sb = new StringBuilder ();
      int simd_length = simd_class.DoubleVectorLength * simd_class.NbNativeVector;
      for (int i = 0; i < simd_length; ++i) {
        if (i != 0)
          sb.Append (',');
        sb.AppendFormat ("{0}({1}[{2}])", func_name, arg_name, i);
      }
      return sb.ToString ();
    }
    public static string ComputeArgs(string op_symbol,string func_name,bool arg1_name,bool arg2_name,ISimdClass simd_class)
    {
      bool is_operator = func_name.StartsWith ("operator");
      StringBuilder sb = new StringBuilder ();
      int simd_length = simd_class.DoubleVectorLength * simd_class.NbNativeVector;
      for (int i = 0; i < simd_length; ++i) {
        if (i != 0)
          sb.Append (',');
        string x1 = (arg1_name) ? String.Format ("a.v{0}", i) : "a";
        string x2 = (arg2_name) ? String.Format ("b.v{0}", i) : "b";
          
        if (is_operator)
          sb.AppendFormat ("{1} {0} {2}", op_symbol, x1, x2);
        else
          sb.AppendFormat ("{0}({1},{2})", func_name, x1, x2);
      }
      return sb.ToString ();
    }
    public static string GenTest(BinaryOperation op)
    {
      string op_symbol = CppHelper.OpSymbolName(op);
      string func_name = CppHelper.OpFuncName (op);
      string func_full_name = String.Format ("{0}::{1}", OpFuncNamespace (op), func_name);
      bool is_operator = func_name.StartsWith ("operator");
      Console.WriteLine ("IS_OPERATOR={0} {1}", is_operator, func_name);
      StringBuilder sb = new StringBuilder ();
      if (is_operator)
        sb.AppendFormat ("a {0} b", op_symbol);
      else
        sb.AppendFormat ("{0}(a,b)", func_full_name);
      return sb.ToString ();
    }
  }
}

