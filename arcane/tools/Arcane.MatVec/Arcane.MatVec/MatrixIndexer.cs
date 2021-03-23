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
using Integer = System.Int32;

namespace Arcane.MatVec
{
/// <summary>
/// Classe de base des Indexer sur les elements d'une matrice 
/// </summary>
  public class MatrixIndexer : IDisposable
  {
    protected Matrix m_matrix;
    protected Int32Array m_indexes;
    protected Int32ConstArrayView m_indexes_view;
    protected Int64 m_epoch;
    protected RealArrayView m_values;
    bool m_is_disposed;
    
    public MatrixIndexer(Matrix matrix)
    {
      m_matrix = matrix;
      m_indexes = new Int32Array();
      m_epoch = -1;
    }

    ~MatrixIndexer()
    {
      Console.WriteLine("Missing call to Arcane.Matvec.MatrixIndexer.Dispose()");
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
        m_indexes.Dispose();
      }
      m_is_disposed = true;
    }
    protected void _EndBuild()
    {
      m_values = m_matrix.Values;
      m_epoch = m_matrix.Epoch;
      m_indexes_view = m_indexes.ConstView;
    }
    
    //! Vrai si l'instance a ete disposee et ne doit donc plus etre utilisee
    public bool IsDisposed { get { return m_is_disposed; } }
    //! Liste des indexes
    public Int32ConstArrayView Indexes { get { return m_indexes_view; } }
    //! Epoque a laquelle a ete calculee cet indexer
    public Int64 Epoch { get { return m_epoch; } }

    public double this[int i]
    {
      get
      {
#if ARCANE_CHECK
        if (m_epoch!=m_matrix.Epoch)
          throw new ApplicationException("Bad epoch for DiagonalMatrixIndexer.Item");
#endif
        return m_values[m_indexes_view[i]];
      }
      set
      {
#if ARCANE_CHECK
        if (m_epoch!=m_matrix.Epoch)
          throw new ApplicationException("Bad epoch for DiagonalMatrixIndexer.SetItem");
        if (Double.IsNaN(value) || Double.IsInfinity(value))
          throw new ApplicationException("Can not set NAN or Infinite value");
#endif
	     m_values[m_indexes_view[i]] = value;
      }
    }
  }


  public class CustomMatrixIndexer : MatrixIndexer
  {
    public CustomMatrixIndexer(Matrix matrix) : base(matrix)
    {
    }
    
    public void Build(Int32ConstArrayView rows,Int32ConstArrayView columns)
    {
      Integer nb_row = rows.Length;
      Integer nb_col = columns.Length;
      if (nb_row!=nb_col)
        throw new ArgumentException(String.Format("number of rows and columns differs nb_row={0} nb_col={1}",nb_row,nb_col));
      m_indexes.Resize(nb_row);
      m_matrix.GetIndexes(m_indexes.View,rows,columns);
      _EndBuild();
    }
    
    public void Build(Int32ConstArrayView rows,Int32ConstArrayView columns,Int32 row_offset,Int32 column_offset)
    {
      Integer nb_row = rows.Length;
      Integer nb_col = columns.Length;
      if (nb_row!=nb_col)
        throw new ArgumentException(String.Format("number of rows and columns differs nb_row={0} nb_col={1}",nb_row,nb_col));
      m_indexes.Resize(nb_row);
      m_matrix.GetIndexesWithOffset(m_indexes.View,rows,columns,row_offset,column_offset);
      _EndBuild();
    }
    
  }
}
