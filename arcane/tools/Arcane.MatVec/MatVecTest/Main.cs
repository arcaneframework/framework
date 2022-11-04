//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using Arcane;
using Arcane.MatVec;

namespace MatVecTest
{
  class MatVecTest
  {
    internal void Test1()
    {
      using Matrix m = new Matrix(5,4);
      double ref_v1 = 1.0;
      double ref_v2 = 2.0;
      m.SetValue(0,2,ref_v1);
      m.SetValue(1,3,ref_v2);
      double v1 = m.Value(0,2);
      double v2 = m.Value(1,3);
      if (v1!=ref_v1)
        throw new ApplicationException("Bad value 1");
      if (v2!=ref_v2)
        throw new ApplicationException("Bad value 2");
      m.Dump(Console.Out);
      _PrintMatrix(m);
      m.SetValue(0,1,3.0);
      using(Vector vec = new Vector(4)){
        RealArrayView vecv = vec.Values;
        vecv[0] = 1.0;
        vecv[1] = 2.0;
        vecv[2] = 3.0;
        vecv[3] = 4.0;
        using(Vector v_out = new Vector(5)){
          MatrixOperation mo = new MatrixOperation();
          mo.MatrixVectorProduct(m,vec,v_out);
          v_out.Dump(Console.Out);
        }
      }
    }

  internal void Test2()
  {
    using(Matrix matrix = new Matrix(6,6)){

      matrix[0,0] = 1.31814608610996E-12;
      matrix[0,1] =-1.31814608610859E-12;

      matrix.SetValue(1,0,-1.31814608610859E-12);
      matrix.SetValue(1,1, 4.66093078486141E-11);
      matrix.SetValue(1,2,-4.5291161762499E-11);

      matrix.SetValue(2,1,-4.5291161762499E-11);
      matrix.SetValue(2,2,1.95243397695228E-10);
      matrix.SetValue(2,3,-1.49952235932723E-10);

      matrix.SetValue(3,2,-1.49952235932723E-10);
      matrix.SetValue(3,3,4.31954546380107E-10);
      matrix.SetValue(3,4,-2.82002310447369E-10);

      matrix.SetValue(4,3,-2.82002310447369E-10);
      matrix.SetValue(4,4,3.27500080633377E-08);
      matrix.SetValue(4,5,-3.24680057528885E-08);

      matrix.SetValue(5,4,-3.24680057528885E-08);
      matrix.SetValue(5,5,3.24680057528886E-08);

      matrix.Dump(Console.Out);
      Console.WriteLine();
      _PrintMatrix(matrix);

      foreach(MatrixIndex mi in matrix.Indexes){
        Console.WriteLine("ROW={0} COL={1} VAL={2}",mi.Row,mi.Col,matrix[mi]);
      }
      using(Vector b = new Vector(6)){
        RealArrayView vb = b.Values;
        vb[0] =  2.883E+014;
        vb[1] = -1.745E+016;
        vb[2] =  5.996E+015;
        vb[3] = -1.530E+017;
        vb[4] = -1.021E+019;
        vb[5] =  9.716E+018;

        //Vector xref = new Vector(6);
        //xref.Values[0] =
        using(Vector x = new Vector(6)){
          //for( int i=0; i<6; ++i )
          //x.Values[i] = 0.0;
          ConjugateGradientSolver cg = new ConjugateGradientSolver();
          using(DiagonalPreconditioner diag = new DiagonalPreconditioner(matrix)){
            cg.Solve(matrix,b,x,1e-3,diag);
          }
          Console.WriteLine("NB ITER={0} RESIDU={1}",cg.NbIteration,cg.ResidualNorm);
          Console.WriteLine("X=");
          x.Dump(Console.Out);
          Console.WriteLine();

          Console.WriteLine("B=");
          b.Dump(Console.Out);
          Console.WriteLine();

          using(Vector b2 = new Vector(6)){
            MatrixOperation mo = new MatrixOperation();
            mo.MatrixVectorProduct(matrix,x,b2);
            Console.WriteLine("AX=");
            b2.Dump(Console.Out);
            Console.WriteLine();
            using(Vector s = new Vector(6)){
              s.Copy(b2);
              mo.NegateVector(s);
              mo.AddVector(s,b2);
              Console.WriteLine("B-AX=");
              s.Dump(Console.Out);
              Console.WriteLine();
            }
          }

          RealArrayView xv = x.Values;
          for( int i=0; i<6; ++i )
            xv[i] = 0.0;
          DirectGaussSolver dgs = new DirectGaussSolver();
          dgs.Solve(matrix,b,x);
          Console.WriteLine("XDIRECT=");
          x.Dump(Console.Out);
          Console.WriteLine();
        }
      }
    }
  }
  void _PrintMatrix(Matrix m)
  {
      for( int i=0; i<m.NbRow; ++i ){
        for( int j=0; j<m.NbColumn; ++j ){
          Console.Write(" ");
          Console.Write(" "+m.Value(i,j));
        }
        Console.WriteLine();
      }
  }
  }

  class MainClass
  {
    public static void Main(string[] args)
    {
      MatVecTest mvt = new MatVecTest();
      mvt.Test1();
      mvt.Test2();
      Console.WriteLine("END OF TEST OK");
    }
  }
}
