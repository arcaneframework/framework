//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
﻿using System;
using System.Collections.Generic;
using System.Text;
using Real = System.Double;
using Integer = System.Int32;

namespace Arcane.MatVec
{
  public class MatrixOperation
  {
    
    public void MatrixVectorProduct(Matrix mat, Vector vec, Vector out_vec)
    {
      MatrixVectorProduct(mat,vec.ConstValues,out_vec.Values);
    }
    
    public void MatrixVectorProduct(Matrix mat, RealConstArrayView vec_values, RealArrayView out_vec_values)
    {
      Integer nb_row = mat.NbRow;
      Integer nb_column = mat.NbColumn;
      if (nb_column != vec_values.Length)
        throw new ArgumentException("MatrixVectorProduct", "Bad size for input vector");
      if (nb_row != out_vec_values.Length)
        throw new ArgumentException("MatrixVectorProduct", "Bad size for output_vector");
      Int32ConstArrayView rows_index = mat.RowsIndex;
      Int32ConstArrayView columns = mat.Columns;
      Int32ConstArrayView nb_columns = mat.ColumnsSize;
      RealArrayView mat_values = mat.Values;
      //RealConstArrayView vec_values = vec.ConstValues;
      //RealArrayView out_vec_values = out_vec.Values;

      for (Integer i = 0, iss = nb_row; i < iss; ++i) {
        Real sum = 0.0;
        for (Integer z = rows_index[i], zs = z + nb_columns[i]; z < zs; ++z) {
          Integer mj = columns[z];
          sum += vec_values[mj] * mat_values[z];
        }
        out_vec_values[i] = sum;
      }
    }
    //void MatrixVectorProduct2(Matrix mat, Vector vec, Vector out_vec);
    public Real Dot(Vector vec)
    {
      Integer size = vec.Size;
      RealConstArrayView vec_values = vec.ConstValues;
      Real v = 0.0;
      for (Integer i = 0; i < size; ++i)
        v += vec_values[i] * vec_values[i];
      return v;

    }
    public Real Dot(Vector vec1, Vector vec2)
    {
      Real v = 0.0;
      Integer size = vec1.Size;
      RealConstArrayView vec1_values = vec1.ConstValues;
      RealConstArrayView vec2_values = vec2.ConstValues;
      for (Integer i = 0; i < size; ++i) {
        v += vec1_values[i] * vec2_values[i];
        //cout << " i=" << i << " v=" << v << '\n';
      }
      return v;

    }

    public void NegateVector(Vector vec)
    {
      Integer size = vec.Size;
      RealArrayView vec_values = vec.Values;
      for (Integer i = 0; i < size; ++i)
        vec_values[i] = -vec_values[i];

    }
    public void ScaleVector(Vector vec, Real mul)
    {
      Integer size = vec.Size;
      RealArrayView vec_values = vec.Values;
      for (Integer i = 0; i < size; ++i)
        vec_values[i] *= mul;
    }

    public void AddVector(Vector out_vec, Vector vec)
    {
      Integer size = vec.Size;
      RealConstArrayView vec_values = vec.ConstValues;
      RealArrayView out_vec_values = out_vec.Values;
      for (Integer i = 0; i < size; ++i)
        out_vec_values[i] += vec_values[i];
    }
  }
}
