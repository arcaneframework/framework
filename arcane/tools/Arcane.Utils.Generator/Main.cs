//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.IO;
using System.Collections.Generic;
using System.Reflection;

namespace Arcane.Utils.Generator
{
  class Generator
  {
    string m_basepath;
    public Generator()
    {
      string s = Directory.GetCurrentDirectory();
      DirectoryInfo d1 = Directory.GetParent(s);
      string d2 = Path.Combine(d1.FullName,"Arcane.Utils","Arcane.Utils");
      m_basepath = d2;
      Console.WriteLine("PARENT ={0}",m_basepath);
    }
    
    public int Generate()
    {
      Stream file_stream = Assembly.GetAssembly(typeof(Generator)).GetManifestResourceStream("Arcane.Utils.Generator.Array.txt");
      //Console.WriteLine("V={0}",file_stream);
      TextReader tr = new StreamReader(file_stream);
      List<string> lines = new List<string>();
      while(true){
        string line = tr.ReadLine();
        if (line==null)
          break;
        lines.Add(line);
        //Console.WriteLine("LINE={0}",line);
      }
      string[] type_list = new string[]{ "Int16", "Int32", "Int64", "UInt16", "UInt32", "UInt64", "Real", "Byte", "Real2", "Real2x2", "Real3", "Real3x3" };
      //string[] type_list = new string[]{ "Int32", "Int64" };
      foreach(string s in type_list){
        _Generate(s,lines);
      }
      return 0;
    }

    private void _Generate(string type_name,List<string> lines)
    {
      string type_str = "@CTYPE@";
      string path = Path.Combine(m_basepath,type_name+"Array.cs");
      using(StreamWriter tw = File.CreateText(path)){
        tw.WriteLine("//WARNING: this file is generated. Do not Edit");
        tw.WriteLine("//Date "+DateTime.Now);
      foreach(string line in lines){
        string ns = line.Replace(type_str,type_name);
          tw.WriteLine(ns);
      }
      }
      Console.WriteLine("End generate for '{0}' in file '{1}'",type_name,path);
    }
    public static int Main(string[] args)
    {
      Generator gen = new Generator();
      return gen.Generate();
    }
  }
}
