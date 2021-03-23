//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
﻿using System;
using System.Text;
using System.IO;

namespace SimdGenerator
{
  class MainClass
  {
    const string GENERATED_HEADER_TEXT = "// WARNING: This file is generated. Do not edit.\n";
    static Encoding m_output_encoding;

    public static void Main (string[] args)
    {
      m_output_encoding = Encoding.GetEncoding ("iso-8859-1");

      Console.WriteLine ("Hello World!");
      SimdClass[] all_simd = new SimdClass[] {
        new SimdClass (SimdType.Emulated),
        new SimdClass (SimdType.SSE),
        new SimdClass (SimdType.AVX),
        new SimdClass (SimdType.AVX512),
      };
      foreach (SimdClass st in all_simd) {
        SimdClass.CurrentType = st;
        var x = new Simd ();
        string s = x.TransformText ();
        Console.WriteLine ("RESULT={0}", s);
        string out_file_name = "Simd" + st.SimdName + "Generated.h";
        //ATTENTION: chemin relatif sur le répertoire utils de Arcane. A modifier en cas de reorganisation
        // des sources
        string out_path = Path.Combine ("../../../../src/arcane/utils");
        _WriteGeneratedText (out_path, out_file_name, s);
      }

      var unit_test = new UnitTestSimd ();
      _WriteGeneratedText ("../../../../src/arcane/tests", "SimdGeneratedUnitTest.h", unit_test.TransformText ());
    }

    static void _WriteGeneratedText (string base_path, string file_name, string content)
    {
      //TODO: ne générer que si le fichier a changé
      string full_out_file_name = Path.Combine (base_path, file_name);
      Console.WriteLine ("Writing result to file {0}", full_out_file_name);
      string s = GENERATED_HEADER_TEXT + content;
      File.WriteAllText (full_out_file_name, s, m_output_encoding);
    }

  }
}
