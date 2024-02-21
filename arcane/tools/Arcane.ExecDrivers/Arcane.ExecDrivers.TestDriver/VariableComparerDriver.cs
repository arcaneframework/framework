//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using Arcane.VariableComparer;

namespace Arcane.ExecDrivers.TestDriver
{
  public class VariableComparerDriver
  {
    string m_reference_path;
    string m_target_path;

    bool m_is_compare_hash;
    internal void ParseArgs(string [] main_args)
    {
      Mono.Options.OptionSet opt_set = new Mono.Options.OptionSet();

      opt_set.Add("compare-hash", "compare uniquement les valeurs des hashs", (v) => m_is_compare_hash = v != null);
      List<string> remaining_args = opt_set.Parse(main_args);
      foreach (string s in remaining_args) {
        Console.WriteLine("REMAINING ARG '{0}'", s);
      }

      int nb_arg = remaining_args.Count;
      if (nb_arg < 2) {
        Console.WriteLine("Bad number of args.");
        Console.WriteLine("Usage: [exe] [options] ref_path target_path");
        throw new VariableComparerException("Bad number of arguments");
      }
      m_reference_path = remaining_args [0];
      m_target_path = remaining_args [1];
      Console.WriteLine("IsCompareHashValue?={0}",m_is_compare_hash);
      Console.WriteLine("REFERENCE_PATH='{0}'", m_reference_path);
      Console.WriteLine("TARGET_PATH='{0}'", m_target_path);
    }
    internal int Execute()
    {
      var vc = new Arcane.VariableComparer.VariableComparer(m_reference_path, m_target_path);
      vc.ReadDatabase();
      if (m_is_compare_hash)
        return _ExecuteCompareHash(vc);
      return _ExecuteCompareValues(vc);
    }
    int _ExecuteCompareHash(Arcane.VariableComparer.VariableComparer vc)
    {
      int total_nb_diff = 0;
      foreach (VariableMetaData v in vc.CommonVariables) {
        (string ref_hash,string target_hash) = vc.GetComparisonHashValue(v);
        bool is_different = ref_hash != target_hash;
        System.Console.WriteLine("COMPARE_HASH var={0} ref={1} target={2} is_diff?={3}",v.FullName,ref_hash,target_hash,is_different);
        if (is_different)
          ++total_nb_diff;
      }
      Console.WriteLine($"TotalNbDiffHash={total_nb_diff}");
      return total_nb_diff;
    }
    int _ExecuteCompareValues(Arcane.VariableComparer.VariableComparer vc)
    {
      int total_nb_diff = 0;
      foreach (VariableMetaData v in vc.CommonVariables) {
        double [] v_ref;
        double [] v_target;
        int nb_diff = 0;
        vc.ReadVariableAsRealArray(v, out v_target, out v_ref);
        int length = v_target.Length;
        Console.WriteLine("VAR='{0}' TARGET_SIZE={1} REF_SIZE={2} p={3}", v.FullName, length, v_ref.Length, v.Property);
        if (length != v_ref.Length) {
          Console.WriteLine("Different length for variable name={0} ref_length={1} target_length={2}", v.FullName, v_ref.Length, length);
          total_nb_diff += length;
          continue;
        }
        for (int i = 0; i < length; ++i) {
          if (v_ref [i] != v_target [i]) {
            ++nb_diff;
            if (nb_diff <= 10) {
              double diff_val = v_target [i] - v_ref [i];
              double sum = v_target [i] + v_ref [i];
              if (sum != 0.0)
                diff_val /= sum;
              Console.WriteLine("Difference i={0} ref={1} target={2} rel_diff={3}", i, v_ref [i], v_target [i], diff_val);
            }
          }
        }
        if (nb_diff != 0)
          Console.WriteLine("Variable name={0} has nb_diff={1}/{2}", v.FullName, nb_diff, length);

        total_nb_diff += nb_diff;
        //int n = Math.Min(10,length);
        //for( int i=0; i<n; ++i )
        //Console.WriteLine("VALUES TARGET={0} REF={1}",v_target[i],v_ref[i]);
      }
      Console.WriteLine("Total nb diff={0}", total_nb_diff);
      return 0;
    }
  }
}
