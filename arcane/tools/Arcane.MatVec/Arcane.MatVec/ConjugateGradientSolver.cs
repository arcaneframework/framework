//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Text;
using Real = System.Double;
using Integer = System.Int32;

namespace Arcane.MatVec
{
  public class ConjugateGradientSolver
  {
    public ConjugateGradientSolver()
    {
      m_nb_iteration = 0;
      m_residual_norm = 0.0;
      m_max_iteration = 5000;
    }
    /// <summary>
    /// Résoud le système d'équation Ax=b avec ou sans préconditionneur
    /// </summary>
    /// <param name="a">Matrice</param>
    /// <param name="b">Vecteur second membre</param>
    /// <param name="x">Vecteur solution</param>
    /// <param name="epsilon"></param>
    /// <param name="p">Précondtionneur (null si aucun)</param>
    /// <returns></returns>
    public bool Solve(Matrix a, Vector b, Vector x, Real epsilon, IPreconditioner p)
    {
      m_nb_iteration = 0;
      m_residual_norm = 0.0;

      DiagonalPreconditioner p2 = null;
      if (p == null){
        p2 = new DiagonalPreconditioner(a);
        p = p2;
      }
      _applySolver(a, b, x, epsilon, p);
      if (p2!=null)
        p2.Dispose();
      return false;
    }

    /// <summary>
    /// Nombre d'itérations effectuées par le solveur
    /// </summary>
    public Integer NbIteration { get { return m_nb_iteration; } }

    public Real ResidualNorm { get { return m_residual_norm; } }

    public void SetMaxIteration(Integer max_iteration)
    {
      m_max_iteration = max_iteration;
    }
    private Integer m_nb_iteration;
    private Real m_residual_norm;
    private Integer m_max_iteration;
    private void _applySolver(Matrix a, Vector b, Vector x, Real epsilon, IPreconditioner p)
    {
      MatrixOperation mat_op = new MatrixOperation();
      Integer vec_size = a.NbRow;
      using(Vector r = new Vector(vec_size)){
        mat_op.MatrixVectorProduct(a,x,r);
        mat_op.NegateVector(r);
        mat_op.AddVector(r,b);

        m_nb_iteration = 0;
        m_residual_norm = 0.0;

        using(Vector d = new Vector(r.Size)){
          d.Copy(r);
          if (p!=null)
            p.Apply(d,r);
          using(Vector q = new Vector(r.Size))
          using(Vector t = new Vector(r.Size))
          using(Vector s = new Vector(r.Size)){
            Real delta_new = 0.0;
            //Real r0=mat_op.Dot(r);
            if (p!=null){
              delta_new = mat_op.Dot(r,d);
            }
            else
              delta_new = mat_op.Dot(r);
            Real delta0 = delta_new;
            //Console.WriteLine("delta0={0}",delta0);
            //Console.WriteLine("deltanew={0}",delta_new);
            Integer nb_iter = 0;
            for( nb_iter=0; nb_iter<m_max_iteration; ++nb_iter ){
              Console.WriteLine("ITER {0} delta_new={1} want={2}",nb_iter,delta_new,epsilon*epsilon*delta0);
              if (delta_new < epsilon*epsilon*delta0)
                break;
              //cout << "delta_new=" << delta_new << '\n';
              mat_op.MatrixVectorProduct(a,d,q);
              Real alpha = delta_new / mat_op.Dot(d,q);
              t.Copy(d);
              mat_op.ScaleVector(t,alpha);
              mat_op.AddVector(x,t);
              // r <= b - Ax
              mat_op.MatrixVectorProduct(a,x,r);
              mat_op.NegateVector(r);
              mat_op.AddVector(r,b);
              if (p!=null)
                p.Apply(s,r);
              Real delta_old = delta_new;
              if (p!=null)
                delta_new = mat_op.Dot(r,s);
              else
                delta_new = mat_op.Dot(r);
              Real beta = delta_new / delta_old;
              //cout << " alpha=" << alpha << " beta=" << beta << " delta_new=" << delta_new << '\n';
              mat_op.ScaleVector(d,beta);
              if (p!=null){
                //mat_op.addVector(s,r);
                mat_op.AddVector(d,s);
              }
              else
                mat_op.AddVector(d,r);
              //cout << '\n';
            }
            //cout << " X=";
            //x.dump(cout);
            //cout << '\n';
            //cout << "NB ITER=" << nb_iter << " epsilon=" << epsilon
            //     << " delta0=" << delta0
            //     << " delta_new=" << delta_new << " r=" << mat_op.dot(r) << " r0=" << r0 << '\n';
            m_nb_iteration = nb_iter;
            m_residual_norm = delta_new;
          }
        }
      }
    }
    //private void _applySolver2(Matrix a, Vector b, Vector x, Real epsilon, IPreconditioner precond);
  }
}
