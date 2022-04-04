//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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
  /// Indexer sur la diagonale d'une matrice
  /// Cet objet est temporaire et ne doit pas etre conserve.
  /// Cet objet est invalide si la matrice change de structure
  /// </summary>
  public class DiagonalMatrixIndexer : MatrixIndexer
  {
    Integer m_first_row;
    Integer m_nb_row;
    
    public DiagonalMatrixIndexer(Matrix matrix,Integer first_row,Integer nb_row) : base(matrix)
    {
      m_first_row = first_row;
      m_nb_row = nb_row;
    }
    
    public void CheckRebuild()
    {
      if (m_epoch!=m_matrix.Epoch)
        _Build();
    }
    
    private void _Build()
    {
      int nb_row = m_nb_row;
      Integer first_row = m_first_row;
      m_indexes.Resize(nb_row);
      //m_indexes = m_diagonal_indexes.View;
      for( int i=0; i<nb_row; ++i ){
        m_indexes[i] = m_matrix.GetIndex(first_row+i,first_row+i);
        //Console.WriteLine("INDEX I={0} DIAG={1} INDEX={2}",i,first_row+i,m_indexes[i]);
      }
      _EndBuild();
    }
  }
}
