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
  /// <summary>
  /// Indice d'un element de la matrice
  /// Cet indice doit servir lors d'une iteration
  /// </summary>
  public struct MatrixIndex
  {
    private int m_row;
    private int m_col;
    internal int m_index;
    internal Matrix m_matrix;
    
    public MatrixIndex(int row, int col)
    {
      m_row = row;
      m_col = col;
      m_matrix = null;
      m_index = -1;
    }

    public MatrixIndex(Matrix matrix,int index,int row,int col)
    {
      m_row = row;
      m_col = col;
      m_matrix = matrix;
      m_index = index;
    }
    public int Row { get { return m_row; } }
    public int Col { get { return m_col; } }
  }
  
  /// <summary>
  /// Matrice
  /// </summary>
  public class Matrix : IDisposable
  {
    Integer m_nb_row;
    Integer m_nb_column;
    //Integer m_nb_element;
    RealArray m_values = new RealArray();
    //! Indice dans m_columns pour chaque ligne de la premiere colonne
    Int32Array m_rows_index = new Int32Array();
    Int32Array m_columns = new Int32Array();
    Int32Array m_rows = new Int32Array();
    
    DiagonalMatrixIndexer m_diagonal_indexer;
    Int64 m_epoch = 1;
    Int64 m_index_epoch;
    Int32Array m_valid_indexes = new Int32Array();
    Int32Array m_valid_rows = new Int32Array();
    Int32Array m_nb_columns = new Arcane.Int32Array(); //! Nombre de colonne allouees de chaque ligne
    Integer m_max_nb_column;
    
    public Matrix(Integer nb_row, Integer nb_column)
    {
      m_nb_row = nb_row;
      m_nb_column = nb_column;
      //m_nb_element = 0;
      m_rows_index.Resize(nb_row + 1);
      Int32[] rows_size = new Int32[nb_row];
      m_nb_columns.Resize(nb_row);
      for( Integer i=0; i<nb_row; ++i ){
        rows_size[i] = nb_column;
        m_nb_columns[i] = 0;
      }
      m_max_nb_column = nb_column;
      SetRowsSize(rows_size);
      m_diagonal_indexer = new DiagonalMatrixIndexer(this,0,nb_row);
    }

    /// <summary>
    /// Cree une matrice (nb_row,nb_column) avec au maximum nb_allocated_column colonnes par ligne
    /// </summary>
    /// <param name="nb_row">
    /// A <see cref="Integer"/>
    /// </param>
    /// <param name="nb_column">
    /// A <see cref="Integer"/>
    /// </param>
    /// <param name="nb_allocated_column">
    /// A <see cref="Integer"/>
    /// </param>
    public Matrix(Integer nb_row, Integer nb_column,Integer nb_allocated_column)
    {
      m_nb_row = nb_row;
      m_nb_column = nb_column;
      //m_nb_element = 0;
      m_rows_index.Resize(nb_row + 1);
      Integer[] rows_size = new Integer[nb_row];
      m_nb_columns.Resize(nb_row);
      for( Integer i=0; i<nb_row; ++i ){
        rows_size[i] = nb_allocated_column;
        m_nb_columns[i] = 0;
      }
      m_max_nb_column = nb_allocated_column;
      SetRowsSize(rows_size);
      m_diagonal_indexer = new DiagonalMatrixIndexer(this,0,nb_row);
    }

    public void Dispose()
    {
      _Dispose(true);
      GC.SuppressFinalize(this);
    }
    
    void _Dispose(bool disposing)
    {
      if (disposing){
        m_values.Dispose();
        m_rows_index.Dispose();
        m_columns.Dispose();
        m_rows.Dispose();
        m_nb_columns.Dispose();
        m_valid_indexes.Dispose();
        m_valid_rows.Dispose();
        m_diagonal_indexer.Dispose();
      }
    }

    ~Matrix()
    {
      Console.WriteLine("Missing call to Arcane.Matrix.Dispose()");
      _Dispose(false);
    }

    //! Clone la matrice
    public Matrix Clone()
    {
      //Matrix new_matrix = new Matrix(m_nb_row, m_nb_column);
      throw new NotImplementedException();
    }

    // Nombre de lignes de la matrice
    public Integer NbRow { get { return m_nb_row; } }

    // Nombre de colonnes de la matrice
    public Integer NbColumn { get { return m_nb_column; } }

    // Nombre de colonnes de la matrice
    public Int32ConstArrayView ColumnsSize { get { return m_nb_columns.ConstView; } }
    
    //! Positionne le nombre d'éléments non nuls de chaque ligne
    public void SetRowsSize(Integer[] rows_size)
    {
      ++m_epoch;
      Integer index = 0;
      for (Integer i = 0, s = m_nb_row; i < s; ++i) {
        m_rows_index[i] = index;
        index += rows_size[i];
      }
      m_rows_index[m_nb_row] = index;
      //m_nb_element = index;
      m_columns.Resize(index);
      m_columns.Fill(-1);
      m_rows.Resize(index);
      //for( int i=0; i<m_columns.Length; ++i )
      //  m_columns[i] = -1;
      m_values.Resize(index);
      
      // Positionne des zeros sur la diagonale
      int max_index = Math.Min(m_nb_column,m_nb_row);
      for( int i=0; i<max_index; ++i )
        SetValue(i,i,0.0);
    }

    //! Positionne les valeurs des éléments de la matrice
    public void SetValues(Int32ConstArrayView columns, RealConstArrayView values)
    {
      m_columns.Copy(columns);
      m_values.Copy(values);
      CheckValid();
    }

    void CheckValid()
    {
      Int32ConstArrayView columns = m_columns.ConstView;
      Int32ConstArrayView rows_index = m_rows_index.ConstView;

      Integer nb_column = m_nb_column;
      for (Integer row = 0, nb_row = m_nb_row; row < nb_row; ++row) {
        for (Integer j = rows_index[row], js = j+m_nb_columns[row]; j < js; ++j) {
          if (columns[j] >= nb_column || columns[j] < 0) {
            Console.WriteLine("BAD COLUMN VALUE for row={0} column={1} column_index={2} nb_column={3} nb_row={4}",
                         row, columns[j], j, nb_column, nb_row);
            throw new ApplicationException("Bad Matrix");
          }
        }
      }
    }

    //! Imprime la matrice
    public void Dump(TextWriter o)
    {
      Int32ConstArrayView columns = Columns;
      Int32ConstArrayView rows_index = RowsIndex;
      for (Integer row = 0, nb_row = m_nb_row; row < nb_row; ++row) {
        for (Integer j = rows_index[row], js = j+m_nb_columns[row]; j < js; ++j)
          o.Write("({0},{1})={2} ",row,columns[j],m_values[j]);
      }
    }

    //! Valeurs de la matrice
    public RealArrayView Values { get { return m_values.View; } }

    //! Indices des premiers éléments de chaque ligne
    public Int32ConstArrayView RowsIndex { get { return m_rows_index.ConstView; } }

    //! Indices des colonnes des valeurs
    public Int32ConstArrayView Columns { get { return m_columns.ConstView; } }

    //! Indices des lignes des valeurs
    public Int32ConstArrayView Rows { get { return m_rows.ConstView; } }

    public Int32ConstArrayView ValidIndexes
    {
      get
      {
        if (m_epoch!=m_index_epoch){
          _ComputeValidIndexes();
        }
        return m_valid_indexes.ConstView;
      }
    }

    public Int32ConstArrayView ValidRows
    {
      get
      {
        if (m_epoch!=m_index_epoch){
          _ComputeValidIndexes();
        }
        return m_valid_rows.ConstView;
      }
    }

    private void _ComputeValidIndexes()
    {

      //Console.WriteLine("COMPUTE INDEXES");
      Int32 nb = 0;
      //Int32[] columns = m_columns;
      Int32ArrayView rows_index = m_rows_index.View;
      for (Integer row = 0, nb_row = m_nb_row; row < nb_row; ++row) {
        nb += m_nb_columns[row];
      }
      
      m_valid_indexes.Resize(nb);
      m_valid_rows.Resize(nb);
      
      Int32ArrayView v_indexes = m_valid_indexes.View;
      Int32ArrayView v_rows = m_valid_rows.View;
      nb = 0;
      
      for (Integer row = 0, nb_row = m_nb_row; row < nb_row; ++row) {
        for (Integer j = rows_index[row], js = j + m_nb_columns[row]; j < js; ++j){
          v_indexes[nb] = j;
          v_rows[nb] = row;
          ++nb;
        }
      }
      
      m_index_epoch = m_epoch;
      //m_valid_indexes = v_indexes;
      //m_valid_rows = v_rows;
    }
    
    /// <summary>
    /// Remplit la matrice avec la valeur \a v
    /// </summary>
    /// <param name="v">
    /// A <see cref="Real"/>
    /// </param>
    public void Fill(Real v)
    {
      Integer nb_row = NbRow;
      Int32ConstArrayView rows_index = RowsIndex;
      Int32ConstArrayView nb_columns = ColumnsSize;
      RealArrayView mat_values = Values;

      for (Integer i = 0, iss = nb_row; i < iss; ++i) {
        for (Integer z = rows_index[i], zs = z + nb_columns[i]; z < zs; ++z) {
          mat_values[z] = v;
        }
      }
    }
    
    /// <summary>
    /// Accesseur de la matrice
    /// </summary>
    /// <param name="index">
    /// Indice de l'element de la matrice <see cref="Integer"/>
    /// </param>
    public Real this[MatrixIndex index]
    {
      get
      {
        if (index.m_matrix != this)
          return Value(index.Row,index.Col);
        return m_values[index.m_index];
      }
      set
      {
        if (index.m_matrix!=this){
          SetValue(index.Row,index.Col,value);
          return;
        }
        m_values[index.m_index] = value;
      }
    }
    
    public Real this[Integer row,Integer col]
    {
      get { return Value(row,col); }
      set { SetValue(row,col,value); }
    }
    public MatrixIndex Index(Integer row,Integer col)
    {
      return new MatrixIndex(row,col);
    }
    
    //! Positionne la valeur d'un élément de la matrice
    public void SetValue(Integer row, Integer column, Real v)
    {
      m_values[GetIndex(row,column)] = v;
    }

    /// <summary>
    /// Recupere l'index dans le tableau des valeurs de la position (row,column).
    /// </summary>
    public Int32 GetIndex(Integer row, Integer column)
    {
      Int32ConstArrayView rows_index = RowsIndex;
      Int32ArrayView columns = m_columns.View;
#if ARCANE_CHECK
      if (row >= m_nb_row)
        throw new ArgumentException("Invalid row");
      if (column >= m_nb_column)
        throw new ArgumentException("Invalid column");
      if (row < 0)
        throw new ArgumentException("Invalid row");
      if (column < 0)
        throw new ArgumentException("Invalid column");
#endif
      Int32 n = m_nb_columns[row];
      Int32 js = rows_index[row] + m_nb_columns[row];
      for (Integer j = rows_index[row]; j < js; ++j) {
        if (columns[j] == column) {
          return j;
        }
      }
      if (n>=m_max_nb_column)
        throw new ArgumentException("Too many column");

      columns[js] = column;
      m_rows[js] = row;
      m_nb_columns[row] = m_nb_columns[row] + 1;
      ++m_epoch;
      return js;
    }

    public void GetIndexes(Int32ArrayView indexes,Int32ConstArrayView rows, Int32ConstArrayView columns)
    {
      Integer n = rows.Length;
      for(Integer i=0; i<n; ++i )
      	indexes[i] = GetIndex(rows[i],columns[i]);  
    }

    public void GetIndexesWithOffset(Int32ArrayView indexes,Int32ConstArrayView rows,
                                     Int32ConstArrayView columns,Int32 row_offset,Int32 column_offset)
    {
      Integer n = rows.Length;
      for(Integer i=0; i<n; ++i )
      	indexes[i] = GetIndex(rows[i]+row_offset,columns[i]+column_offset);  
    }

    //! Retourne la valeur d'un élément de la matrice
    public Real Value(Integer row, Integer column)
    {
      Int32ConstArrayView rows_index = RowsIndex;
      Int32ConstArrayView columns = Columns;
      RealConstArrayView values = m_values.ConstView;

      for (Integer z = rows_index[row], zs = rows_index[row + 1]; z < zs; ++z) {
        if (columns[z] == column)
          return values[z];
      }
      return 0.0;
    }
    
    public Int64 Epoch { get { return m_epoch; } }
    
    public MatrixEnumerable Indexes
    {
      get{ return new MatrixEnumerable(this); }
    }
    
    public DiagonalMatrixIndexer DiagonalIndexer
    {
      get
      {
        m_diagonal_indexer.CheckRebuild();
        return m_diagonal_indexer;
      }
    }
  }
}
