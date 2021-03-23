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
  public class DiagonalPreconditioner : IPreconditioner, IDisposable
  {
    private Vector m_inverse_diagonal;
    public DiagonalPreconditioner(Matrix matrix)
    {
      m_inverse_diagonal = new Vector(matrix.NbRow);
      Integer size = m_inverse_diagonal.Size;
      Int32ConstArrayView rows_index = matrix.RowsIndex;
      //Integer[] columns = matrix.Columns;
      RealArrayView mat_values = matrix.Values;
      RealArrayView vec_values = m_inverse_diagonal.Values;
      for (Integer i = 0, iss = size; i < iss; ++i) {
        vec_values[i] = mat_values[rows_index[i]];
        //for (Integer z = rows_index[i], zs = rows_index[i + 1]; z < zs; ++z) {
          //Integer mj = columns[z];
          //if (mj == i)
            //vec_values[i] = mat_values[z];
        //}
      }

      for (Integer i = 0; i < size; ++i)
        vec_values[i] = 1.0 / vec_values[i];
    }
    ~DiagonalPreconditioner()
    {
      _Dispose(false);
    }
    public void Dispose()
    {
      _Dispose(true);
      GC.SuppressFinalize(this);
    }
    void _Dispose(bool disposing)
    {
      if (disposing){
        Console.WriteLine("DiagonalPreconditioner.Dispose()");
        m_inverse_diagonal.Dispose();
      }
    }
    
    public void Apply(Vector out_vec, Vector vec)
    {
      Integer size = m_inverse_diagonal.Size;
      RealConstArrayView inverse_diagonal_values = m_inverse_diagonal.ConstValues;
      RealConstArrayView vec_values = vec.ConstValues;
      RealArrayView out_vec_values = out_vec.Values;
      for (Integer i = 0; i < size; ++i)
        out_vec_values[i] = vec_values[i] * inverse_diagonal_values[i];
    }
  }
}
