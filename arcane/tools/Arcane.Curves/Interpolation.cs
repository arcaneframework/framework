//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Arcane.Curves
{
  internal struct SubGrid
  {
    internal double x;
    internal double y;
    internal int i;
    internal SubGrid(double _x, double _y)
    {
      x = _x;
      y = _y;
      i = -1;
    }
    internal SubGrid(double _x, double _y, int _i)
    {
      x = _x;
      y = _y;
      i = _i;
    }
  }

  public class Interpolation
  {
    /// Fonction de lagrange generique
    static double _Lagrange(double u, double[] x, double[] y)
    {
      int n = x.Length;
      double interp = 0.0;
      for (int i = 0; i < n; ++i){
        double p = 1.0;
        for (int j = 0; j < n; ++j){
          if (i != j)
            p *= (u - x[j]) / (x[i] - x[j]);
        }
        interp += p * y[i];
      }
      return interp;
    }

    static public double Lagrange(double u, double x0, double y0, double x1, double y1, double x2, double y2)
    {
#if false
      double[] x = new double[3]{x0,x1,x2};
      double[] y = new double[3]{y0,y1,y2};
      return _Lagrange(u,x,y);
#else
      double interp = 0.0;
      // i = 0;
      double p = 1.0;
      p *= (u - x1) / (x0 - x1); // j = 1
      p *= (u - x2) / (x0 - x2); // j = 2
      interp += p * y0;
      // i = 1;
      p = 1.0;
      p *= (u - x0) / (x1 - x0); // j = 0
      p *= (u - x2) / (x1 - x2); // j = 2
      interp += p * y1;
      // i = 2;
      p = 1.0;
      p *= (u - x0) / (x2 - x0); // j = 0
      p *= (u - x1) / (x2 - x1); // j = 1
      interp += p * y2;
      return interp;
#endif
    }

    static double Lagrange(double a, double x0, double y0, double x1, double y1)
    {
      double[] x = new double[2] { x0, x1 };
      double[] y = new double[2] { y0, y1 };
      return _Lagrange(a, x, y);
    }

    /// retourne le point, le couple ou le triplet de points necessaire pour le calcul du polynome d'interpolation
    static SubGrid sousGrille(double a, int n, ICurve cv, int i)
    {
      // si a n'appartient pas à l'intervalle grille =>erreur
      int length = cv.NbPoint;
      RealConstArrayView curve_grid = cv.X;
      RealConstArrayView f = cv.Y;
      if (a < curve_grid[0] || a > curve_grid[length - 1]){
        Trace.WriteLine(a + " n'appartient pas à la grille");
        return new SubGrid(0, 0);
      }
      if (a == curve_grid[0]){
        //pas d'interpolation
        return new SubGrid(curve_grid[0], f[0], i);
      }

      if (n == 0){
        // interpolation P0: on retourne le point le plus proche du point d'interpolation a
        while (curve_grid[i] <= a){
          if (curve_grid[i] == a){
            //pas d'interpolation
            return new SubGrid(a, f[i], i);
          }
          i = i + 1;
        }
        if (Math.Abs(a - curve_grid[i - 1]) >= Math.Abs(a - curve_grid[i]))
          return new SubGrid(a, f[i], i);
        return new SubGrid(a, f[i - 1], i);
      }

      if (n == 1){
        if (length < 2){
          Trace.WriteLine("pas assez de points pour permettre une interpolation de Lagrange");
          return new SubGrid(0, 0);
        }
        while (curve_grid[i] <= a){
          if (curve_grid[i] == a){
            // pas d'interpolation
            return new SubGrid(a, f[i], i);
          }
          i = i + 1;
        }
        return new SubGrid(a, Lagrange(a, curve_grid[i - 1], f[i - 1], curve_grid[i], f[i]), i);
      }

      if (n == 2){
        if (length < 3){
          Console.WriteLine("pas assez de points pour permettre une interpolation de Lagrange");
          return new SubGrid(0, 0);
        }
        //Console.WriteLine("I={0} grid.x={1} a={2}",i,curve_grid[i],a);
        while (curve_grid[i] < a)
          i = i + 1;
        if (curve_grid[i] == a){
          // || i>=(length-1)){
          //pas d'interpolation
          return new SubGrid(a, f[i], i);
        }
        //if(i<(length-1)){
        // test sur la distance
        try
        {
          bool do_bi = false;
          if (i < length - 1 && i >= 2){
            double val = Math.Abs(a - curve_grid[i + 1]);
            do_bi = Math.Abs(a - curve_grid[i - 2]) >= val;
          }
          do_bi |= (i < 2);
          if (do_bi)
            return new SubGrid(a, Lagrange(a, curve_grid[i - 1], f[i - 1], curve_grid[i], f[i], curve_grid[i + 1], f[i + 1]), i);
          else
            return new SubGrid(a, Lagrange(a, curve_grid[i - 2], f[i - 2], curve_grid[i - 1], f[i - 1], curve_grid[i], f[i]), i);
        }
        catch (Exception e){
          Console.WriteLine("Interpolation_sub_grid: Exception catched {0} i={1} a={2} grid[i]={3} length={4}", e.Message, i, a, curve_grid[i], length);
          return new SubGrid(a, Lagrange(a, curve_grid[i - 2], f[i - 2], curve_grid[i - 1], f[i - 1], curve_grid[i], f[i]), i);
        }
      }

      throw new ArgumentException("Bad value for interpolation order");
    }

    //----------------------------------------------------------------------------
    //# Calculer la fonction resultat de la projection d'une courbe connue
    //# sur une grille choisie avec la methode d'interpolation Pn
    //#
    //# Parametres:
    //#   curve  : courbe a projeter
    //#   grid   : liste des x de la nouvelle grille
    //#   type   : degres de la methode d'interpolation
    //#
    //# Retour:
    //#   fonctionResultat : courbe de la fonction projetee
    static public ICurve Interpolate(int type, RealConstArrayView grid, ICurve curve)
    {
      RealArray result_x = new RealArray();
      RealArray result_y = new RealArray();

      // boucle sur les elements de la grille choisie
      try{
        int i = 1;
        Trace.WriteLine(String.Format("Curve nb_point={0} grid_size={1}", curve.NbPoint, grid.Length));
        foreach (double x in grid){
          SubGrid result = sousGrille(x, type, curve, i);  // intervalle ou l'on va interpoler
          double cx = result.x;
          double cy = result.y;
          i = result.i;

          if (i < 0)
            Trace.WriteLine(String.Format(" x= {0} n'appartient pas a la grille de la fonction", x));
          else{
            result_x.Add(cx);
            result_y.Add(cy);
          } 
        }

        return new BasicCurve("ComparedCurve",result_x, result_y);
      }
      catch(Exception){
        result_x.Dispose();
        result_y.Dispose();
        throw;
      }
    }
  }
}

