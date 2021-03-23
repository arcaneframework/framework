//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.IO;
using Real = System.Double;
using Arcane;
//using Arcane.Compiler.Directives;
using Integer = System.Int32;

namespace Arcane.MatVec
{
  public class MatrixEnumerable : IEnumerable<MatrixIndex>, IEnumerable
  {
    Matrix m_data;
    public MatrixEnumerable(Matrix data)
    {
      m_data = data;
    }
    IEnumerator IEnumerable.GetEnumerator()
    {
      return new MatrixEnumerator(m_data);
    }
    IEnumerator<MatrixIndex> IEnumerable<MatrixIndex>.GetEnumerator()
    {
      return new MatrixEnumerator(m_data);
    }
    public MatrixEnumerator GetEnumerator()
    {
      return new MatrixEnumerator(m_data);
    }
  }
  
  /// <summary>
  /// Enumerateur sur les elements de la matrice
  /// </summary>
  //[CppClassGenerationInfo(TypeMapping=TypeMapping.Value)]
  public class MatrixEnumerator : IEnumerator<MatrixIndex>, IEnumerator
  {
    Matrix m_data;
    //int m_current_row;
    //int m_current_column;
    Int32ConstArrayView m_columns;
    Int32ConstArrayView m_valid_rows;
    Int32ConstArrayView m_valid_indexes;
    Int32 m_current_index;
    Int32 m_max_index;
    public MatrixEnumerator(Matrix data)
    {
      m_data = data;
      //m_current_row = 0;
      //m_current_column = -1;
      m_columns = data.Columns;
      m_valid_rows = data.ValidRows;
      m_valid_indexes = data.ValidIndexes;
      m_max_index = m_valid_indexes.Length;
      m_current_index = -1;
    }
    void IDisposable.Dispose()
    {
    }
    void IEnumerator.Reset()
    {
      //m_current_row = 0;
      //m_current_column = -1;
      m_current_index = -1;
    }
    bool IEnumerator.MoveNext()
    {
      return MoveNext();
    }
    public bool MoveNext()
    {
      ++m_current_index;
      return (m_current_index<m_max_index);
    }
    public MatrixIndex Current
    {
      get{
        //return new MatrixIndex(m_data,m_current_column,m_current_row,m_data.Columns[m_current_column]);
        return new MatrixIndex(m_data,m_valid_indexes[m_current_index],m_valid_rows[m_current_index],m_columns[m_valid_indexes[m_current_index]]);
      }
    }
    MatrixIndex IEnumerator<MatrixIndex>.Current
    {
      get{
        return Current;
      }
    }
    object IEnumerator.Current
    {
      get {
        return Current;
      }
    }
  }
}
