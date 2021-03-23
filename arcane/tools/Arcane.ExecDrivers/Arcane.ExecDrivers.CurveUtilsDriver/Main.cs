//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Arcane.Curves;

namespace Arcane.ExecDrivers.CurveUtilsDriver
{
  class CurveUtilsDriver
  {
    public static int Main (string[] args)
    { 
      int nb_arg = args.Length;
      if (nb_arg==0){
        Console.WriteLine("Usage: program add|read|compare|gnuplot-to-acv|smooth|to-gnuplot|rename [options]");
        return (-1);
      }
      List<string> l_remaining_args = new List<string>();
      for( int i=1; i<nb_arg; ++i )
        l_remaining_args.Add(args[i]);
      int r = 0;
      switch(args[0]){
      case "add":
        _DoAddCurve(l_remaining_args);
        break;
      case "read":
        _Read2(l_remaining_args);
         break;
      case "compare":
        _DoCompare(l_remaining_args);
        break;
      case "gnuplot":
        // Obsolète
        _DoReadGnuplot(l_remaining_args);
        break;
      case "gnuplot-to-acv":
        _DoGnuplotToACV(l_remaining_args);
        break;
      case "smooth":
        _DoSmooth(l_remaining_args);
        break;
      case "to-gnuplot":
        _DoConvertToGnuplot(l_remaining_args);
        break;
      case "rename":
        _DoRenameCurves(l_remaining_args);
        break;
      default:
        string valid_values = "'read', 'compare', 'gnuplot', 'smooth', 'to-gnuplot', 'rename'";
        throw new ApplicationException(String.Format("Bad option '{0}' for driver. Valid values are {1}",args[0],valid_values));
      }
      return r;
    }
    
    public static void _ReadCurves(string path,bool is_parallel)
    {
      int nb_curve = 0;
      var a = CaseCurves.ReadCase(path);
      nb_curve += a.Curves.Count;
      int total_nb_point = 0;
      if (is_parallel){
        Parallel.ForEach(a.Curves,(ICaseCurve c) => { ICurve dc = c.Read(); total_nb_point += dc.NbPoint; });
      }
      else{
        foreach(ICaseCurve c in a.Curves){
          ICurve dc = c.Read();
          //Console.WriteLine("Curve name={0} nb_point={1}",dc.Name,dc.NbPoint);
          total_nb_point += dc.NbPoint;
        }
      }
      Console.WriteLine("NB_CURVE={0} nb_point={1}",nb_curve,total_nb_point);
    }

    public static void _Read2(List<string> args)
    {
      string path = args[0];
      if (path.StartsWith("@")){
        _ReadOne(args[0],true);
        return;
      }
      CaseCurves cc = CaseCurves.ReadCase(path);
      int nb_curve = 0;
      nb_curve += cc.Curves.Count;
      Console.WriteLine("NB_CURVE={0}",nb_curve);
      foreach(ICaseCurve c in cc.Curves){
        ICurve dc = c.Read();
        Console.WriteLine("Curve name={0} nb_point={1}",dc.Name,dc.NbPoint);
      }
      
    }

    public static void _ReadOne(string file,bool is_parallel)
    {
      string true_name = file.Substring(1);
      Console.WriteLine("FILE={0}",true_name);
      string[] all_names = System.IO.File.ReadAllLines(true_name);
      if (is_parallel){
        Parallel.ForEach(all_names,(string name) => { _ReadCurves(name,false);} );
      }
      else{
        foreach(string name in all_names){
          Console.WriteLine("READ file={0}",name);
          _ReadCurves(name,false);
        }
      }
      Console.WriteLine("Total nb_file={0}",all_names.Length);
    }

    /*!
     * \brief Ajoute des courbes à un fichier 'curves.acv'.
     *
     * \a args contient la liste des courbes au format GNUPLOT xy à ajouter.
     */
    static void _DoAddCurve(List<string> args)
    {
      int nb_arg = args.Count;
      if (nb_arg < 2) {
        Console.WriteLine("Usage: a.out add [acvfile] xy1 [xy2 ...]");
        Console.WriteLine("Add curves from files 'xy1' with gnuplot XY format to 'acvfile' file");
        return;
      }
      string acvfile_name = args [0];
      Console.WriteLine("Reading ACV file '{0}'", acvfile_name);
      // Comme le fichier de sortie peut être le même que le fichier d'entrée, il
      // faut lire d'un seul coup (NOTE le problème ne se pose que si le
      // fichier d'entrée s'appelle 'curves.acv'; il serait possible de ne faire
      // cette lecture complète que dans ce cas)
      byte [] acv_bytes = File.ReadAllBytes(acvfile_name);
      ArcaneCaseReader reader = ArcaneCaseReader.CreateFromMemory(acv_bytes, new CaseReaderSettings());
      CaseCurves curves = reader.CaseCurves;

      Console.WriteLine("NB_CURVE={0}", curves.Curves.Count);
      for (int i = 1; i < nb_arg; ++i) {
        string file_name = args [i];
        string curve_name = Path.GetFileName(file_name);
        Console.WriteLine("Adding curve name={0} filename={1}", file_name, curve_name);
        ICurve curve = GnuplotCurveReader.ReadCurve(curve_name, file_name);
        curves.AddCurve(new BasicCaseCurve(curve));
      }
      string outpath = Path.GetDirectoryName(acvfile_name);
      Console.WriteLine("Writing ACV file in directory '{0}'", outpath);
      ArcaneCaseWriter.WriteCurves(curves, outpath);
    }

    static void _DoReadGnuplot(List<string> args)
    {
      Console.WriteLine("WARNING: command 'gnuplot' is deprecated. Use 'gnuplot-to-acv' instead");
      _DoGnuplotToACV(args);
    }

    static void _DoGnuplotToACV(List<string> args)
    {
      GnuplotCaseReader reader = new GnuplotCaseReader();
      int nb_arg = args.Count;
      if (nb_arg < 2) {
        Console.WriteLine("Usage: a.out gnuplot-to-acv input_path output_path");
        return;
      }
      reader.ReadPath(args [0]);
      string out_path = args [1];
      ArcaneCaseWriter.WriteCurves(reader.CaseCurves, out_path);
    }

    static void _DoConvertToGnuplot(List<string> args)
    {
      int nb_arg = args.Count;
      if (nb_arg<2){
        Console.WriteLine("Usage: a.out to-gnuplot input_path output_path [name1] [name2] ... [namen]");
        return;
      }
      CaseCurves curves = CaseCurves.ReadCase(args[0]);
      string out_path = args[1];
      // Les arguments supplémentaires s'ils existent indiquent le nom des courbes à extraire
      HashSet<string> names_to_output = null;
      if (nb_arg > 2) {
        names_to_output = new HashSet<string>();
        for (int i = 2; i < nb_arg; ++i)
          names_to_output.Add(args[i]);
      }
      GnuplotCurveWriter writer = new GnuplotCurveWriter(out_path);
      foreach(ICaseCurve cc in curves.Curves){
        ICurve c = cc.Read();
        bool do_output = true;
        if (names_to_output != null)
          do_output = names_to_output.Contains(c.Name);
        if (do_output)
          writer.Write(c);
      }
    }

    static void _DoSmooth(List<string> args)
    {
      if (args.Count<2){
        Console.WriteLine("USAGE: a.out smooth nb_point curve_path");
        return;
      }
      string nb_point_str = args[0];
      Console.WriteLine("NB_WANTED_POINT={0}",nb_point_str);
      int nb_point = Int32.Parse(nb_point_str);
      CaseSmoother smoother = new CaseSmoother(args[1],nb_point);
      smoother.Smooth();
    }

    static void _DoCompare(List<string> args)
    {
      if (args.Count<2){
        Console.WriteLine ("Usage: a.out compare curve_dir1 curve_dir2");
        return;
      }
      CaseComparer cc = new CaseComparer(args[0],args[1]);
      cc.Compare();
    }

    static void _DoRenameCurves(List<string> args)
    {
      if (args.Count < 2) {
        Console.WriteLine("Usage: a.out rename rename.xml curve_axl_file");
        return;
      }
      var cc = new CaseRenamer(args[0], args[1]);
      cc.Execute();
    }
  }
}