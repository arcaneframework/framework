//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using Integer = System.Int32;
using Real = System.Double;

namespace Arcane.MatVec
{
  public class DirectGaussSolver
  {
    public void Solve(Matrix matrix,Vector vector_b,Vector vector_x)
  {
    Int32ConstArrayView rows = matrix.RowsIndex;
    Int32ConstArrayView columns = matrix.Columns;
    RealArrayView mat_values = matrix.Values;

    Integer nb_row = matrix.NbRow;
    Real[] solution_values = new Real[nb_row];
    Real[] full_matrix_values = new Real[nb_row*nb_row];
      for( Integer i=0; i<(nb_row*nb_row); ++i )
    full_matrix_values[i] = 0.0;
    for( Integer i=0; i<nb_row; ++i )
      solution_values[i] = vector_b.Values[i];
    for( Integer row=0; row<nb_row; ++row ){
      for( Integer j=rows[row]; j<rows[row+1]; ++j ){
        if (columns[j]!=(-1))
          full_matrix_values[row*nb_row + columns[j]] = mat_values[j];
      }
    }
    _Solve(full_matrix_values,solution_values,nb_row);
      RealArrayView vx = vector_x.Values;
    for( Integer i=0; i<nb_row; ++i )
      vx[i] = solution_values[i];
    //vector_x.values().copy(solution_values);
    Console.WriteLine("END OF DIRECT SOLVER");
  }

    void _Solve(Real[] mat_values,Real[] vec_values,Integer size)
    {
      if (size==1){
        if (mat_values[0]==0.0)
          throw new ApplicationException("DirectSolver::Null matrix");
        vec_values[0] /= mat_values[0];
        return;
      }

      for( Integer k=0; k<size-1; ++k ){
        if (mat_values[k*size+k]!=0){
          for( Integer j=k+1; j<size; ++j ){
            if (mat_values[j*size+k]!=0){
              Real factor = mat_values[j*size+k] / mat_values[k*size+k];
              for( Integer m=k+1; m<size; ++m )
                mat_values[j*size+m] -= factor * mat_values[k*size+m];
              vec_values[j] -= factor * vec_values[k];
            }
          }
        }
      }
      
      for( Integer k=(size-1); k>0; --k ){
        vec_values[k] /= mat_values[k*size+k];
        for( Integer j=0; j<k; ++j ){
          if (mat_values[j*size+k]!=0)
            vec_values[j] -= vec_values[k] * mat_values[j*size+k];
        }
      }
      
      vec_values[0] /= mat_values[0];
    }
  }
}
