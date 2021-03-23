//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

using System;

namespace Arcane.Curves
{
  public class Smoother
  {
    int m_nb_interpolation_point;
    public Smoother (int nb_point)
    {
      m_nb_interpolation_point = nb_point;
    }

    public ICurve Apply(ICurve curve)
    {
      RealConstArrayView x = curve.X;
      RealConstArrayView y = curve.Y;
      int nbp = curve.NbPoint;
      RealArray new_x = new RealArray(nbp);
      RealArray new_y = new RealArray(nbp);

      for (int i = 0; i < nbp; ++i){
        double sum = 0.0;
        /*for( int z=i_begin; z<i_end; ++z ){
          sum = sum + y[z]*(x[i]-x[z]);
        }
        sum /= (x[i_end] - x[i_begin]);*/
        //if (i>0 && i<(nbp-1)){
          //sum = Interpolation.Lagrange(x[i],x[i-1],y[i-1],x[i],y[i],x[i+1],y[i+1]);
          //sum = (y[i+1] + y[i]) / 2.0;
        //}
        //else
          //sum = y[i];

        // Moyenne mobile
#if false
        int n = 10;
        int nleft = n;
        if (i<n)
          nleft = i;
        int nright = n;
        if ((i+n)>=nbp)
          nright = nbp - i;
        sum = 0.0;
        int nb_used = 0;
        for( int z = 1; z<nleft; ++z ){
          sum += y[i-z];
          ++nb_used;
        }
        for( int z = 1; z<nright; ++z ){
          sum += y[i+z];
          ++nb_used;
        }
        sum += y[i];

        sum /= (double)(nb_used + 1);
#endif
        int n = m_nb_interpolation_point;
        int i_begin = i - n;
        int i_end = i + n;
        if (i_begin<1)
          i_begin = 1;
        if (i_end>=(nbp-1))
          i_end = nbp - 2;
        if (i==0 || i==(nbp-1)){
          sum = y[i];
        }
        else{
          double sum_dx = 0.0;
          for( int z=i_begin; z<i_end; ++z ){
            double dx = (x[z+1] + x[z-1]) / 2.0;
            sum_dx += dx;
            sum += dx * y[z];
          }
          sum /= sum_dx;
        }

        //if (curve.Name=="OutFlux-MonteCarlo-InstantaneousPower-ch_Sg5"){
          //Console.WriteLine("I={0} Y={1} NEW_Y={2}",i,y[i],sum);
        //}
        new_x[i] = x[i];
        new_y[i] = sum;
      }

      ICurve new_curve = new BasicCurve(curve.Name,new_x,new_y);
      return new_curve;
    }
  }
}
