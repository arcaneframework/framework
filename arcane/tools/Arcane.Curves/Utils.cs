//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections.Generic;

namespace Arcane.Curves
{
  public static class Utils
  {
    static readonly bool g_verbose = false;
    /// <summary>
    /// Intersection entre deux grilles
    /// </summary>
    /// <param name="grid1">
    /// A <see cref="RealConstArrayView"/>
    /// </param>
    /// <param name="grid2">
    /// A <see cref="RealConstArrayView"/>
    /// </param>
    /// <returns>
    /// A <see cref="RealArray"/>
    /// </returns>
    public static RealArray Intersection(RealConstArrayView grille1,RealConstArrayView grille2)
    {
      RealArray grilleRes = new RealArray();
      //List<double> grilleRes = new List<double>();
      //CurveGrid res = new CurveGrid();
      //si la premiere grille est vide: on retourne la deuxieme grille
      if (grille1.Size==0){
        grilleRes.Copy(grille2);
        return grilleRes;
      }
      // si la deuxieme grille est vide: on retourne la premiere grille
      if (grille2.Size==0){
        grilleRes.Copy(grille1);
        return grilleRes;
      }

      int long1 = grille1.Length - 1;
      int long2 = grille2.Length - 1;

      double a = Math.Max(grille1[0],grille2[0]);
      double b = Math.Min(grille1[long1],grille2[long2]);

      if (b<a){
        Console.WriteLine("erreur dans l'intersection des 2 grilles: intersection vide");
        return grilleRes;
      }
      if (g_verbose){
        Console.WriteLine("les bornes de l'intersection des 2 grilles sont : {0},{1}",a,b);
      }
      // 2: remplissage
      grilleRes.Add(a);
      for( int i=0; i<grille1.Length; ++i ){
        double v = grille1[i];
        if ( (v > a) && (v < b))
          grilleRes.Add(v);
      }
      grilleRes.Add(b);
      //res.m_values = grilleRes.ToArray();
      return grilleRes;
    }

    public static ICurve Projection(ICurve curve,RealConstArrayView grid,string methode)
    {
      int n = 0;
      if (methode == "p0")
        n = 0;
      else if (methode == "p1")
        n = 1;
      else if (methode == "p2")
        n = 2;
      else{
        Console.WriteLine("erreur dans le choix de la methode d'interpolation");
        return null;
      }

      ICurve res = Interpolation.Interpolate( n, grid,curve);

      return res;
    }

    public static ICurve CropCurve (ICurve ref_curve, double min_x, double max_x)
    {
      if (min_x == Double.MinValue && max_x == Double.MaxValue)
        return ref_curve;
      RealArray new_x = new RealArray();
      RealArray new_y = new RealArray();
      RealConstArrayView x = ref_curve.X;
      RealConstArrayView y = ref_curve.Y;
      int nb_point = ref_curve.NbPoint;
      for (int i = 0; i < nb_point; ++i) {
        double xx = x[i];
        if (xx >= min_x && xx < max_x) {
          new_x.Add(xx);
          new_y.Add(y[i]);
        }
      }
      return new BasicCurve(ref_curve.Name, new_x, new_y);
    }

    public static void ComputeProjection (ICurve ref_curve, ICurve target_curve,
                                         out ICurve ref_projection, out ICurve target_projection, out RealArray grid)
    {
      ComputeProjection(ref_curve, target_curve, Double.MinValue, Double.MaxValue, out ref_projection, out target_projection, out grid);
    }

    public static void ComputeProjection(ICurve ref_curve,ICurve target_curve,double min_x,double max_x,
                                         out ICurve ref_projection,out ICurve target_projection,out RealArray grid)
    {
      ref_curve = CropCurve(ref_curve, min_x, max_x);
      target_curve = CropCurve(target_curve, min_x, max_x);
      RealConstArrayView gr1 = ref_curve.X;
      RealConstArrayView gr2 = target_curve.X;

      using(RealArray igr1 = Utils.Intersection(gr1,gr2))
      using(RealArray igr2 = Utils.Intersection(gr2,gr1)){

        RealArray pgr = Utils.Intersection(igr1.ConstView, igr2.ConstView);

        // Pour utiliser la methode p2, il faut au moins 3 points sur la courbe
        string pmethod = "p2";
        if (pgr.Size==2)
          pmethod = "p1";

        ICurve pcrb1 = Utils.Projection(ref_curve,pgr.ConstView, pmethod);
        ICurve pcrb2 = Utils.Projection(target_curve,pgr.ConstView, pmethod);

        ref_projection = pcrb1;
        target_projection = pcrb2;
        grid = pgr;
      }
    }

    public static double NormeInf(ICurve curve)
    {
      double v = 0.0;
      RealConstArrayView values = curve.Y;
      for( int i=0; i<values.Length; ++i ){
        v = Math.Max(v,Math.Abs(values[i]));
      }
      return v;
    }

    public static void WriteCurves(IList<ICurve> curves,ICurveWriter writer)
    {
      DateTime t1 = DateTime.Now;
      foreach(BasicCurve c in curves)
        writer.Write(c);
      DateTime t2 = DateTime.Now;
      TimeSpan diff = t2 - t1;
      Console.WriteLine("TIME_TO_WRITE={0} (in s)",diff.TotalSeconds);
    }

    public static ICurve CreateCurve(string name,Int32ConstArrayView iterations,RealArray values,RealConstArrayView times)
    {
      int nb_value = values.Length;
      int nb_iter = iterations.Length;
      RealArray c_x = new RealArray(nb_value);

      if (nb_iter==nb_value){
        for( int i=0; i<values.Length; ++i ){
          c_x[i] = times[iterations[i]];
        }
      }
      else if (nb_iter==2){
        // Contient un intervalle d'iteration
        int begin_iter = iterations[0];
        for( int i=0; i<values.Length; ++i ){
          c_x[i] = times[begin_iter+i];
        }
      }
      return new BasicCurve(name,c_x,values);
    }

    public static int VerboseLevel =0;
    static Utils()
    {
      string s = Environment.GetEnvironmentVariable ("ARCANE_CURVE_VERBOSE");
      if (String.IsNullOrEmpty (s))
        return;
      int r = 0;
      int.TryParse(s,out r);
      VerboseLevel = r;
    }
  }
}
