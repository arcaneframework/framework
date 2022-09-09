// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMG.cc                                                      (C) 2000-2018 */
/*                                                                           */
/* Multi-grille algébrique.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/StringBuilder.h"

#include "arcane/matvec/Matrix.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
namespace math
{
Real divide(Real a,Real b)
{
  if (b==0.0)
    throw FatalErrorException("Division by zero");
  return a / b;
}
}
ARCANE_END_NAMESPACE

ARCANE_BEGIN_NAMESPACE
namespace MatVec
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DirectSolver
{
 public:
  void solve(const Matrix& matrix,const Vector& vector_b,Vector& vector_x)
  {
    IntegerConstArrayView rows = matrix.rowsIndex();
    IntegerConstArrayView columns = matrix.columns();
    RealConstArrayView mat_values = matrix.values();

    Integer nb_row = matrix.nbRow();
    RealUniqueArray solution_values(nb_row);
    RealUniqueArray full_matrix_values(nb_row*nb_row);
    full_matrix_values.fill(0.0);
    solution_values.copy(vector_b.values());
    for( Integer row=0; row<nb_row; ++row ){
      for( Integer j=rows[row]; j<rows[row+1]; ++j ){
        full_matrix_values[row*nb_row + columns[j]] = mat_values[j];
      }
    }
    _solve(full_matrix_values,solution_values,nb_row);
    vector_x.values().copy(solution_values);
  }
 private:
  void _solve(RealArrayView mat_values,RealArrayView vec_values,Integer size)
  {
    if (size==1){
      if (math::isZero(mat_values[0]))
        throw FatalErrorException("DirectSolver","Null matrix");
      vec_values[0] /= mat_values[0];
      return;
    }

    for( Integer k=0; k<size-1; ++k ){
      if (!math::isZero(mat_values[k*size+k])){
        for( Integer j=k+1; j<size; ++j ){
          if (!math::isZero(mat_values[j*size+k])){
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
        if (!math::isZero(mat_values[j*size+k]))
          vec_values[j] -= vec_values[k] * mat_values[j*size+k];
      }
    }

    vec_values[0] /= mat_values[0];
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Matrix MatrixOperation2::
matrixMatrixProduct(const Matrix& left_matrix,const Matrix& right_matrix)
{
  Integer nb_left_col = left_matrix.nbColumn();
  Integer nb_right_col = right_matrix.nbColumn();
  Integer nb_right_row = right_matrix.nbRow();
  Integer nb_left_row = left_matrix.nbRow();
  if (nb_left_col!=nb_right_row)
    throw new ArgumentException("MatrixMatrixProduction","Bad size");
  Integer nb_row_col = nb_left_col;

  //IntegerConstArrayView left_rows_index = left_matrix.rowsIndex();
  //IntegerConstArrayView left_columns = left_matrix.columns();
  //RealConstArrayView left_values = left_matrix.values();

  //IntegerConstArrayView right_rows_index = right_matrix.rowsIndex();
  //IntegerConstArrayView right_columns = right_matrix.columns();
  //RealConstArrayView right_values = right_matrix.values();
    
  Matrix new_matrix(nb_left_row,nb_right_col);
  IntegerUniqueArray new_matrix_rows_size(nb_left_row);
  RealUniqueArray new_matrix_values;
  IntegerUniqueArray new_matrix_columns;
  
  for( Integer i=0; i<nb_left_row; ++i ){
    Integer local_nb_col = 0;
    for( Integer j=0; j<nb_right_col; ++j){
      Real v = 0.0;
      for( Integer k=0; k<nb_row_col; ++k ){
        //if (i==1 && j==0){
        // Real v0 = left_matrix.value(i,k) * right_matrix.value(k,j);
        //  cout << "** CHECK CONTRIBUTION k=" << k
        //       << " l=" << left_matrix.value(i,k) << " r=" << right_matrix.value(k,j) << '\n';
        //  if (!math::isZero(v0))
        //    cout << "** ADD CONTRIBUTION k=" << k << " v0=" << v0
        //         << " l=" << left_matrix.value(i,k) << " r=" << right_matrix.value(k,j) << '\n';
        // }
        v += left_matrix.value(i,k) * right_matrix.value(k,j);
      }
      if (!math::isZero(v)){
        ++local_nb_col;
        new_matrix_columns.add(j);
        new_matrix_values.add(v);
      }
    }
    new_matrix_rows_size[i] = local_nb_col;
  }
  new_matrix.setRowsSize(new_matrix_rows_size);
  new_matrix.setValues(new_matrix_columns,new_matrix_values);
  return new_matrix;
}

Matrix MatrixOperation2::
matrixMatrixProductFast(const Matrix& left_matrix,const Matrix& right_matrix)
{
  Integer nb_left_col = left_matrix.nbColumn();
  Integer nb_right_col = right_matrix.nbColumn();
  Integer nb_right_row = right_matrix.nbRow();
  Integer nb_left_row = left_matrix.nbRow();
  if (nb_left_col!=nb_right_row)
    throw new ArgumentException("MatrixMatrixProduction","Bad size");
  //Integer nb_row_col = nb_left_col;

  IntegerConstArrayView left_rows_index = left_matrix.rowsIndex();
  IntegerConstArrayView left_columns = left_matrix.columns();
  RealConstArrayView left_values = left_matrix.values();

  IntegerConstArrayView right_rows_index = right_matrix.rowsIndex();
  IntegerConstArrayView right_columns = right_matrix.columns();
  RealConstArrayView right_values = right_matrix.values();
    
  Matrix new_matrix(nb_left_row,nb_right_col);
  IntegerUniqueArray new_matrix_rows_size(nb_left_row);
  RealUniqueArray new_matrix_values;
  IntegerUniqueArray new_matrix_columns;

  IntegerUniqueArray col_right_columns_index(nb_right_col+1);
  IntegerUniqueArray col_right_rows;
  RealUniqueArray col_right_values;
  IntegerUniqueArray col_right_columns_size(nb_right_col);
  {
    // Calcule le nombre d'éléments de chaque colonne
    col_right_columns_size.fill(0);
    for( Integer i=0; i<nb_right_row; ++i ){
      for( Integer j=right_rows_index[i]; j<right_rows_index[i+1]; ++j ){
        ++col_right_columns_size[right_columns[j]];
      }
    }
    // Calcule l'index du premier élément de chaque colonne.
    Integer index = 0;
    for( Integer j=0; j<nb_right_col; ++j ){
      col_right_columns_index[j] = index;
      index += col_right_columns_size[j];
    }
    col_right_columns_index[nb_right_col] = index;

    col_right_rows.resize(index);
    col_right_values.resize(index);
    index = 0;
    // Remplit les valeurs par colonne
    col_right_columns_size.fill(0);
    for( Integer i=0; i<nb_right_row; ++i ){
      for( Integer j=right_rows_index[i]; j<right_rows_index[i+1]; ++j ){
        Integer col = right_columns[j];
        Real value = right_values[j];
        Integer col_index = col_right_columns_size[col] + col_right_columns_index[col];
        ++col_right_columns_size[col];
        col_right_rows[col_index] = i;
        col_right_values[col_index] = value;          
      }
    }
  }
  //_dumpColumnMatrix(cout,col_right_columns_index,col_right_rows,col_right_values);
  //cout << '\n';
  RealUniqueArray current_row_values(nb_left_col);
  current_row_values.fill(0.0);
  for( Integer i=0; i<nb_left_row; ++i ){
    Integer local_nb_col = 0;
    // Remplit la ligne avec les valeurs courantes
    for( Integer z=left_rows_index[i] ,zs=left_rows_index[i+1]; z<zs; ++z ){
      current_row_values[ left_columns[z] ] = left_values[z];
      //if (i==1)
      //cout << " ** FILL VALUE col=" << left_columns[z] << " v=" << left_values[z] << '\n';
    }
    //if (i==1){
    //for( Integer z=0; z<nb_left_col; ++z ){
    //current_row_values[ left_columns[z] ] = left_values[z];
    //  cout << " ** VALUE col=" << z << " v=" << current_row_values[z] << '\n';
    //}
    //}

    for( Integer j=0; j<nb_right_col; ++j ){
      Real v = 0.0;
      for( Integer zj=col_right_columns_index[j]; zj<col_right_columns_index[j+1]; ++zj ){
        //if (i==1 && j==0){
        //  Real v0 = col_right_values[zj] * current_row_values[ col_right_rows[zj] ];
        //  cout << "** CHECK CONTRIBUTION2 k=" << col_right_rows[zj]
        //       << " l=" << current_row_values[ col_right_rows[zj] ] << " r=" << col_right_values[zj] << '\n';
        //  if (!math::isZero(v0))
        //    cout << "** ADD CONTRIBUTION2 k=" << col_right_rows[zj] << " v0=" << v0
        //         << " l=" << current_row_values[ col_right_rows[zj] ] << " r=" << col_right_values[zj] << '\n';
        // }
        v += col_right_values[zj] * current_row_values[ col_right_rows[zj] ];
      }
      if (!math::isZero(v)){
        ++local_nb_col;
        new_matrix_columns.add(j);
        new_matrix_values.add(v);
      }
    }

    new_matrix_rows_size[i] = local_nb_col;

    // Remet des zeros dans la ligne courante.
    for( Integer z=left_rows_index[i] ,zs=left_rows_index[i+1]; z<zs; ++z )
      current_row_values[ left_columns[z] ] = 0.0;
  }
  new_matrix.setRowsSize(new_matrix_rows_size);
  new_matrix.setValues(new_matrix_columns,new_matrix_values);
  return new_matrix;
}

void MatrixOperation2::
_dumpColumnMatrix(std::ostream& o,IntegerConstArrayView columns_index,IntegerConstArrayView rows,
                  RealConstArrayView values)
{
  Integer nb_col = columns_index.size() - 1;
  o << "(ColumnMatrix nb_col=" << nb_col;
  for( Integer j=0; j<nb_col; ++j ){
    for( Integer z=columns_index[j], zs=columns_index[j+1]; z<zs; ++z ){
      Integer i = rows[z];
      Real v = values[z];
      o << " ["<<i<<","<<j<<"]="<<v;
    }
  }
  o << ")";
}

Matrix MatrixOperation2::
transpose(const Matrix& matrix)
{
  Integer nb_column = matrix.nbColumn();
  Integer nb_row = matrix.nbRow();

  //IntegerConstArrayView rows_index = matrix.rowsIndex();
  //IntegerConstArrayView columns = matrix.columns();
  //RealConstArrayView values = matrix.values();

  Integer new_matrix_nb_row = nb_column;
  Integer new_matrix_nb_column = nb_row;
  Matrix new_matrix(new_matrix_nb_row,new_matrix_nb_column);
  IntegerUniqueArray new_matrix_rows_size(new_matrix_nb_row);
  RealUniqueArray new_matrix_values;
  IntegerUniqueArray new_matrix_columns;

  for( Integer i=0; i<new_matrix_nb_row; ++i ){
    Integer local_nb_col = 0;
    for( Integer j=0; j<new_matrix_nb_column; ++j ){
      Real v = matrix.value(j,i);
      if (!math::isZero(v)){
        ++local_nb_col;
        new_matrix_columns.add(j);
        new_matrix_values.add(v);
      }
    }
    new_matrix_rows_size[i] = local_nb_col;
  }
  new_matrix.setRowsSize(new_matrix_rows_size);
  new_matrix.setValues(new_matrix_columns,new_matrix_values);
  return new_matrix;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Matrix MatrixOperation2::
transposeFast(const Matrix& matrix)
{
  Integer nb_column = matrix.nbColumn();
  Integer nb_row = matrix.nbRow();

  IntegerConstArrayView rows_index = matrix.rowsIndex();
  IntegerConstArrayView columns = matrix.columns();
  RealConstArrayView values = matrix.values();

  Integer new_matrix_nb_row = nb_column;
  Integer new_matrix_nb_column = nb_row;
  Matrix new_matrix(new_matrix_nb_row,new_matrix_nb_column);

  IntegerUniqueArray new_matrix_rows_size(new_matrix_nb_row);

  // Calcul le nombre de colonnes de chaque ligne de la transposee.
  new_matrix_rows_size.fill(0);
  Integer nb_element = values.size();
  for( Integer i=0, is=columns.size(); i<is; ++i ){
    ++new_matrix_rows_size[ columns[i] ];
  }
  new_matrix.setRowsSize(new_matrix_rows_size);
  
  IntegerConstArrayView new_matrix_rows_index = new_matrix.rowsIndex();
  new_matrix_rows_size.fill(0);

  RealUniqueArray new_matrix_values(nb_element);
  IntegerUniqueArray new_matrix_columns(nb_element);

  for( Integer row=0, is=nb_row; row<is; ++row ){
    for( Integer j=rows_index[row]; j<rows_index[row+1]; ++j ){
      Integer col_index = columns[j];
      Integer pos = new_matrix_rows_index[col_index] + new_matrix_rows_size[col_index];
      //cout << "** CURRENT row=" << row << " col=" << col_index << " v=" << values[col_index]
      new_matrix_columns[pos] = row;
      new_matrix_values[pos] = values[j];
      ++new_matrix_rows_size[col_index];
    }
  }

  new_matrix.setValues(new_matrix_columns,new_matrix_values);
  return new_matrix;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Matrix MatrixOperation2::
applyGalerkinOperator(const Matrix& left_matrix,const Matrix& matrix,
                      const Matrix& right_matrix)
{
  Integer nb_original_row = matrix.nbRow();
  Integer nb_final_row = left_matrix.nbRow();
  IntegerUniqueArray p_marker(nb_final_row);
  IntegerUniqueArray a_marker(nb_original_row);
  p_marker.fill(-1);
  a_marker.fill(-1);

  IntegerConstArrayView left_matrix_rows = left_matrix.rowsIndex();
  IntegerConstArrayView left_matrix_columns = left_matrix.columns();
  RealConstArrayView left_matrix_values = left_matrix.values();

  IntegerConstArrayView right_matrix_rows = right_matrix.rowsIndex();
  IntegerConstArrayView right_matrix_columns = right_matrix.columns();
  RealConstArrayView right_matrix_values = right_matrix.values();

  IntegerConstArrayView matrix_rows = matrix.rowsIndex();
  IntegerConstArrayView matrix_columns = matrix.columns();
  RealConstArrayView matrix_values = matrix.values();

  Integer jj_counter = 0;
  Integer jj_row_begining = 0;

  IntegerUniqueArray new_matrix_rows_size(nb_final_row);

  // D'abord, détermine le nombre de colonnes de chaque ligne de la
  // matrice finale
  for( Integer ic = 0; ic<nb_final_row; ++ic ){
    // Ajoute la diagonale
    p_marker[ic] = jj_counter;
    jj_row_begining = jj_counter;
    ++jj_counter;

    // Boucle sur les colonnes de la ligne \a ic de \a matrix
    for( Integer jj1=left_matrix_rows[ic]; jj1<left_matrix_rows[ic+1]; ++jj1 ){
      Integer i1 = left_matrix_columns[jj1];

      // Boucle sur les colonnes de la ligne \a i1 de \a matrix
      for( Integer jj2=matrix_rows[i1]; jj2<matrix_rows[i1+1]; ++jj2 ){
        Integer i2 = matrix_columns[jj2];
        /*--------------------------------------------------------------
         *  Check A_marker to see if point i2 has been previously
         *  visited. New entries in RAP only occur from unmarked points.
         *--------------------------------------------------------------*/
        if (a_marker[i2]!=ic){
          a_marker[i2] = ic;
          /*-----------------------------------------------------------
           *  Loop over entries in row i2 of P.
           *-----------------------------------------------------------*/
          for( Integer jj3=right_matrix_rows[i2]; jj3<right_matrix_rows[i2+1]; ++jj3 ){
            Integer i3 = right_matrix_columns[jj3];
            /*--------------------------------------------------------
             *  Check P_marker to see that RAP_{ic,i3} has not already
             *  been accounted for. If it has not, mark it and increment
             *  counter.
             *--------------------------------------------------------*/
            if (p_marker[i3] < jj_row_begining){
              p_marker[i3] = jj_counter;
              ++jj_counter;
            }
          }
        }
      }
    }
    new_matrix_rows_size[ic] = jj_counter - jj_row_begining;
  }
  static Integer total_rap_size = 0;
  total_rap_size += jj_counter;

  cout << "** RAP_SIZE=" << jj_counter << " TOTAL=" << total_rap_size << '\n';
  Matrix new_matrix(nb_final_row,nb_final_row);
  new_matrix.setRowsSize(new_matrix_rows_size);
  
  //IntegerConstArrayView new_matrix_rows = new_matrix.rowsIndex();
  IntegerArrayView new_matrix_columns = new_matrix.columns();
  RealArrayView new_matrix_values = new_matrix.values();

  // Maintenant, remplit les coefficients de la matrice
  p_marker.fill(-1);
  a_marker.fill(-1);
  jj_counter = 0;
  for( Integer ic = 0; ic<nb_final_row; ++ic ){
    // Ajoute la diagonale
    p_marker[ic] = jj_counter;
    jj_row_begining = jj_counter;
    new_matrix_columns[jj_counter] = ic;
    new_matrix_values[jj_counter] = 0.0;
    ++jj_counter;
    // Boucle sur les colonnes de la ligne \a ic de \a matrix
    for( Integer jj1=left_matrix_rows[ic]; jj1<left_matrix_rows[ic+1]; ++jj1 ){
      Integer i1 = left_matrix_columns[jj1];
      Real r_entry = left_matrix_values[jj1];

      // Boucle sur les colonnes de la ligne \a i1 de \a matrix
      for( Integer jj2=matrix_rows[i1]; jj2<matrix_rows[i1+1]; ++jj2 ){
        Integer i2 = matrix_columns[jj2];
        Real r_a_product = r_entry * matrix_values[jj2];
        /*--------------------------------------------------------------
         *  Check A_marker to see if point i2 has been previously
         *  visited. New entries in RAP only occur from unmarked points.
         *--------------------------------------------------------------*/
        if (a_marker[i2] != ic){
          a_marker[i2] = ic;
          /*-----------------------------------------------------------
           *  Loop over entries in row i2 of P.
           *-----------------------------------------------------------*/
          for( Integer jj3=right_matrix_rows[i2]; jj3<right_matrix_rows[i2+1]; ++jj3 ){
            Integer i3 = right_matrix_columns[jj3];
            Real r_a_p_product = r_a_product * right_matrix_values[jj3];
            /*--------------------------------------------------------
             *  Check P_marker to see that RAP_{ic,i3} has not already
             *  been accounted for. If it has not, create a new entry.
             *  If it has, add new contribution.
             *--------------------------------------------------------*/
            if (p_marker[i3] < jj_row_begining){
              p_marker[i3] = jj_counter;
              new_matrix_values[jj_counter] = r_a_p_product;
              new_matrix_columns[jj_counter] = i3;
              ++jj_counter;
            }
            else{
              new_matrix_values[p_marker[i3]] += r_a_p_product;
            }
          }
        }
        else{
          /*--------------------------------------------------------------
           *  If i2 is previously visted ( A_marker[12]=ic ) it yields
           *  no new entries in RAP and can just add new contributions.
           *--------------------------------------------------------------*/
          for( Integer jj3=right_matrix_rows[i2]; jj3<right_matrix_rows[i2+1]; ++jj3 ){
            Integer i3 = right_matrix_columns[jj3];
            Real r_a_p_product = r_a_product * right_matrix_values[jj3];
            new_matrix_values[p_marker[i3]] += r_a_p_product;
          }
        }
      }
    }

  }
  return new_matrix;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Matrix MatrixOperation2::
applyGalerkinOperator2(const Matrix& left_matrix,const Matrix& matrix,
                       const Matrix& right_matrix)
{
  Integer nb_original_row = matrix.nbRow();
  Integer nb_final_row = left_matrix.nbRow();
  IntegerUniqueArray p_marker(nb_final_row);
  IntegerUniqueArray a_marker(nb_original_row);
  p_marker.fill(-1);
  a_marker.fill(-1);

  const Integer* left_matrix_rows = left_matrix.rowsIndex().data();
  const Integer* left_matrix_columns = left_matrix.columns().data();
  const Real* left_matrix_values = left_matrix.values().data();

  const Integer* right_matrix_rows = right_matrix.rowsIndex().data();
  const Integer* right_matrix_columns = right_matrix.columns().data();
  const Real* right_matrix_values = right_matrix.values().data();

  const Integer* matrix_rows = matrix.rowsIndex().data();
  const Integer* matrix_columns = matrix.columns().data();
  const Real* matrix_values = matrix.values().data();

  Integer jj_counter = 0;
  Integer jj_row_begining = 0;

  IntegerUniqueArray new_matrix_rows_size(nb_final_row);

  // D'abord, détermine le nombre de colonnes de chaque ligne de la
  // matrice finale
  for( Integer ic = 0; ic<nb_final_row; ++ic ){
    // Ajoute la diagonale
    p_marker[ic] = jj_counter;
    jj_row_begining = jj_counter;
    ++jj_counter;

    // Boucle sur les colonnes de la ligne \a ic de \a matrix
    for( Integer jj1=left_matrix_rows[ic]; jj1<left_matrix_rows[ic+1]; ++jj1 ){
      Integer i1 = left_matrix_columns[jj1];

      // Boucle sur les colonnes de la ligne \a i1 de \a matrix
      for( Integer jj2=matrix_rows[i1]; jj2<matrix_rows[i1+1]; ++jj2 ){
        Integer i2 = matrix_columns[jj2];
        /*--------------------------------------------------------------
         *  Check A_marker to see if point i2 has been previously
         *  visited. New entries in RAP only occur from unmarked points.
         *--------------------------------------------------------------*/
        if (a_marker[i2]!=ic){
          a_marker[i2] = ic;
          /*-----------------------------------------------------------
           *  Loop over entries in row i2 of P.
           *-----------------------------------------------------------*/
          for( Integer jj3=right_matrix_rows[i2]; jj3<right_matrix_rows[i2+1]; ++jj3 ){
            Integer i3 = right_matrix_columns[jj3];
            /*--------------------------------------------------------
             *  Check P_marker to see that RAP_{ic,i3} has not already
             *  been accounted for. If it has not, mark it and increment
             *  counter.
             *--------------------------------------------------------*/
            if (p_marker[i3] < jj_row_begining){
              p_marker[i3] = jj_counter;
              ++jj_counter;
            }
          }
        }
      }
    }
    new_matrix_rows_size[ic] = jj_counter - jj_row_begining;
  }
  
  Matrix new_matrix(nb_final_row,nb_final_row);
  new_matrix.setRowsSize(new_matrix_rows_size);
  
  //Integer* new_matrix_rows = new_matrix.rowsIndex();
  Integer* ARCANE_RESTRICT new_matrix_columns = new_matrix.columns().data();
  Real* ARCANE_RESTRICT new_matrix_values = new_matrix.values().data();
  
  // Maintenant, remplit les coefficients de la matrice
  p_marker.fill(-1);
  a_marker.fill(-1);
  jj_counter = 0;
  for( Integer ic = 0; ic<nb_final_row; ++ic ){
    // Ajoute la diagonale
    p_marker[ic] = jj_counter;
    jj_row_begining = jj_counter;
    new_matrix_columns[jj_counter] = ic;
    new_matrix_values[jj_counter] = 0.0;
    ++jj_counter;
    // Boucle sur les colonnes de la ligne \a ic de \a matrix
    for( Integer jj1=left_matrix_rows[ic]; jj1<left_matrix_rows[ic+1]; ++jj1 ){
      Integer i1 = left_matrix_columns[jj1];
      Real r_entry = left_matrix_values[jj1];

      // Boucle sur les colonnes de la ligne \a i1 de \a matrix
      for( Integer jj2=matrix_rows[i1]; jj2<matrix_rows[i1+1]; ++jj2 ){
        Integer i2 = matrix_columns[jj2];
        Real r_a_product = r_entry * matrix_values[jj2];
        /*--------------------------------------------------------------
         *  Check A_marker to see if point i2 has been previously
         *  visited. New entries in RAP only occur from unmarked points.
         *--------------------------------------------------------------*/
        if (a_marker[i2] != ic){
          a_marker[i2] = ic;
          /*-----------------------------------------------------------
           *  Loop over entries in row i2 of P.
           *-----------------------------------------------------------*/
          for( Integer jj3=right_matrix_rows[i2]; jj3<right_matrix_rows[i2+1]; ++jj3 ){
            Integer i3 = right_matrix_columns[jj3];
            Real r_a_p_product = r_a_product * right_matrix_values[jj3];
            /*--------------------------------------------------------
             *  Check P_marker to see that RAP_{ic,i3} has not already
             *  been accounted for. If it has not, create a new entry.
             *  If it has, add new contribution.
             *--------------------------------------------------------*/
            if (p_marker[i3] < jj_row_begining){
              p_marker[i3] = jj_counter;
              new_matrix_values[jj_counter] = r_a_p_product;
              new_matrix_columns[jj_counter] = i3;
              ++jj_counter;
            }
            else{
              new_matrix_values[p_marker[i3]] += r_a_p_product;
            }
          }
        }
        else{
          /*--------------------------------------------------------------
           *  If i2 is previously visted ( A_marker[12]=ic ) it yields
           *  no new entries in RAP and can just add new contributions.
           *--------------------------------------------------------------*/
          for( Integer jj3=right_matrix_rows[i2]; jj3<right_matrix_rows[i2+1]; ++jj3 ){
            Integer i3 = right_matrix_columns[jj3];
            Real r_a_p_product = r_a_product * right_matrix_values[jj3];
            new_matrix_values[p_marker[i3]] += r_a_p_product;
          }
        }
      }
    }

  }
  return new_matrix;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

enum{
  TYPE_UNDEFINED = 0,
  TYPE_COARSE = 1,
  TYPE_FINE = 2,
  TYPE_SPECIAL_FINE = 3
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AMGLevel
: public TraceAccessor
{
 public:
  AMGLevel(ITraceMng* tm,Integer level)
  : TraceAccessor(tm), m_level(level), m_is_verbose(false){}
  virtual ~AMGLevel(){}
 public:
  virtual void buildLevel(Matrix matrix,Real alpha);
 public:
  Matrix fineMatrix()
  {
    return m_fine_matrix;
  }
  Matrix coarseMatrix()
  {
    return m_coarse_matrix;
  }
  Matrix prolongationMatrix()
  {
    return m_prolongation_matrix;
  }
  Matrix restrictionMatrix()
  {
    return m_restriction_matrix;
  }
  Integer nbCoarsePoint() const
  {
    return m_coarse_matrix.nbRow();
  }
  Int32ConstArrayView pointsType() const
  {
    return m_points_type;
  }
  void printLevelInfo();

 private:
  Integer m_level;
  Matrix m_fine_matrix;
  Matrix m_coarse_matrix;
  Matrix m_prolongation_matrix;
  Matrix m_restriction_matrix;
  Int32UniqueArray m_points_type;

  bool m_is_verbose;
 private:
  void _buildCoarsePoints(Real alpha,
                          RealArray& rows_max_val,
                          UniqueArray< SharedArray<Integer> >& depends,
                          IntegerArray& weak_depends
                          );
  void _buildInterpolationMatrix(RealConstArrayView rows_max_val,
                                 UniqueArray< SharedArray<Integer> >& depends,
                                 IntegerArray& weak_depends
                                 );
  void _printLevelInfo(Matrix matrix);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AMG
: public TraceAccessor
{
 public:
  AMG(ITraceMng* tm) : TraceAccessor(tm) {}
  ~AMG();
 public:
  void build(Matrix matrix);
  void solve(const Vector& vector_b,Vector& vector_x);
 private:
  UniqueArray<AMGLevel*> m_levels;
  Matrix m_matrix;
 private:
  void _solve(const Vector& vector_b,Vector& vector_x,Integer level);
  void _relax(const Matrix& matrix,const Vector& vector_b,Vector& vector_x,
              Integer nb_relax);
  void _relax1(const Matrix& matrix,const Vector& vector_b,Vector& vector_x,
              Integer nb_relax);
  void _relaxJacobi(const Matrix& matrix,const Vector& vector_b,Vector& vector_x,
                    Real weight);
  void _relaxGaussSeidel(const Matrix& matrix,const Vector& vector_b,Vector& vector_x,
                         Integer point_type,Int32ConstArrayView points_type);
  void _relaxSymmetricGaussSeidel(const Matrix& matrix,const Vector& vector_b,Vector& vector_x);
  void _printResidualInfo(const Matrix& matrix,const Vector& vector_b,
                          const Vector& vector_x);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AMG::
~AMG()
{
  for( Integer i=0; i<m_levels.size(); ++i )
    delete m_levels[i];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMG::
build(Matrix matrix)
{
  Matrix current_matrix = matrix;
  m_matrix = matrix;
  for( Integer i=1; i<100; ++i ){
    AMGLevel* level = new AMGLevel(traceMng(),i);
    level->buildLevel(current_matrix,0.25);
    m_levels.add(level);
    Integer nb_coarse_point = level->nbCoarsePoint();
    if (nb_coarse_point<20)
      break;
    current_matrix = level->coarseMatrix();
  }
  //for( Integer i=0; i<m_levels.size(); ++i )
  //m_levels[i]->printLevelInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMG::
solve(const Vector& vector_b,Vector& vector_x)
{
  //info() << "AMG::solve";
  if (0){
    OStringStream ostr;
    RealConstArrayView v_values(vector_b.values());
    for( Integer i=0; i<20; ++i )
      ostr() << "VECTOR_F_"<< i << " = " << v_values[i] << " X=" << vector_x.values()[i] << '\n';
    for( Integer i=0; i<v_values.size(); ++i )
      if (math::abs(v_values[i])>1e-5)
        ostr() << "VECTOR_F_"<< i << " = " << v_values[i] << '\n';
    info() << "VECTOR_F\n" << ostr.str();
  }
  //_printResidualInfo(m_matrix,vector_b,vector_x);
  _solve(vector_b,vector_x,0);
  //info() << "END SOLVE";
  //_printResidualInfo(m_matrix,vector_b,vector_x);
  //info() << "END AMG::solve";
  /*{
    OStringStream ostr;
    ostr() << "\nVECTOR_X ";
    vector_x.dump(ostr());
    info() << ostr.str();
    }*/
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMG::
_relax1(const Matrix& matrix,const Vector& vector_b,Vector& vector_x,
       Integer nb_relax)
{
  Vector r(vector_x.size());
  MatrixOperation mat_op;
  for( Integer i=0; i<nb_relax; ++i ){
    // r = b - A * x
    mat_op.matrixVectorProduct(matrix,vector_x,r);
    mat_op.negateVector(r);
    mat_op.addVector(r,vector_b);
    
    // x = x + r
    mat_op.addVector(vector_x,r);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMG::
_relax(const Matrix& matrix,const Vector& vector_b,Vector& vector_x,
       Integer nb_relax)
{
  Real epsilon = 1.0e-10;
  DiagonalPreconditioner p(matrix);
  ConjugateGradientSolver solver;
  solver.setMaxIteration(nb_relax);
  //mat_op.matrixVectorProduct(restriction_matrix,vector_x,new_x);
  //new_x.values().fill(0.0);

  //OStringStream ostr;
  //vector_x.dump(ostr());
  //info() << " RELAX BEFORE VECTOR_X=" << ostr.str();

  solver.solve(matrix,vector_b,vector_x,epsilon,&p);
  //OStringStream ostr;
  //ostr() << " COARSE_B=";
  //new_b.dump(ostr());
  //ostr() << "\nCOARSE_X=";
  //new_x.dump(ostr());
  //info() << "SOLVE COARSE MATRIX nb_iter=" << solver.nbIteration()
  //       << ostr.str();      
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMG::
_relaxJacobi(const Matrix& matrix,const Vector& vector_b,Vector& vector_x,
             Real weight)
{
  IntegerConstArrayView rows = matrix.rowsIndex();
  IntegerConstArrayView columns = matrix.columns();
  RealConstArrayView mat_values = matrix.values();

  RealArrayView x_values = vector_x.values();
  RealConstArrayView b_values = vector_b.values();
  
  Integer nb_row = matrix.nbRow();
  RealConstArrayView cx_values(vector_x.values());
  RealUniqueArray tmp_values(cx_values);
  if (0){
    Integer v = math::min(40,nb_row);
    OStringStream ostr;
    for( Integer i=(nb_row-1); i>(nb_row-v); --i )
      ostr() << "BEFORE_B=" << i << "=" << b_values[i] << " U=" << x_values[i] << " T=" << tmp_values[i] <<'\n';
    info() << "B = X=" << x_values.data() << " T=" << tmp_values.data() << "\n" << ostr.str();
  }
  Real one_minus_weight = 1.0 - weight;
  for( Integer row=0; row<nb_row; ++row ){
    Real diag = mat_values[rows[row]];
    if (math::isZero(diag))
      continue;
    Real res = b_values[row];
    for( Integer j=rows[row]+1; j<rows[row+1]; ++j ){
      Integer col = columns[j];
      res -= mat_values[j] * tmp_values[col];
    }
    x_values[row] *= one_minus_weight;
    x_values[row] += (weight * res) / diag;
  }
  if (0){
    Integer v = math::min(40,nb_row);
    OStringStream ostr;
    for( Integer i=(nb_row-1); i>(nb_row-v); --i )
      ostr() << "AFTER_B=" << i << "=" << b_values[i] << " U=" << x_values[i] << '\n';
    info() << "B\n" << ostr.str();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMG::
_relaxGaussSeidel(const Matrix& matrix,const Vector& vector_b,Vector& vector_x,
                  Integer point_type,Int32ConstArrayView points_type2)
{
#if 1
  IntegerConstArrayView rows = matrix.rowsIndex();
  IntegerConstArrayView columns = matrix.columns();
  RealConstArrayView mat_values = matrix.values();

  RealArrayView x_values = vector_x.values();
  RealConstArrayView b_values = vector_b.values();
  Int32ConstArrayView points_type = points_type2;
#else
  const Integer* rows = matrix.rowsIndex().data();
  const Integer* columns = matrix.columns().data();
  const Real* mat_values = matrix.values().data();

  Real* ARCANE_RESTRICT x_values = vector_x.values().data();
  const Real* b_values = vector_b.values().data();
  const Integer* points_type = points_type2.data();
#endif

  Integer nb_row = matrix.nbRow();
  if (0){
    info() << " RELAX nb_relax=" << " nb_row=" << nb_row
           << " point_type=" << point_type;
    Integer v = math::min(40,nb_row);
    OStringStream ostr;
    for( Integer i=(nb_row-1); i>(nb_row-v); --i )
      ostr() << "BEFORE_B=" << i << "=" << b_values[i] << " U=" << x_values[i] << '\n';
    info() << "B\n" << ostr.str();
  }
  for( Integer row=0; row<nb_row; ++row ){
    Real diag = mat_values[rows[row]];
    if (points_type[row]!=point_type || math::isZero(diag))
      continue;
    Real res = b_values[row];
    for( Integer j=rows[row]+1; j<rows[row+1]; ++j ){
      Integer col = columns[j];
      res -= mat_values[j] * x_values[col];
    }
    x_values[row] = res / diag;
  }
  if (0){
    Integer v = math::min(40,nb_row);
    OStringStream ostr;
    for( Integer i=(nb_row-1); i>(nb_row-v); --i )
      ostr() << "AFTER_B=" << i << "=" << b_values[i] << " U=" << x_values[i] << '\n';
    info() << "B\n" << ostr.str();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMG::
_relaxSymmetricGaussSeidel(const Matrix& matrix,const Vector& vector_b,Vector& vector_x)
{
  IntegerConstArrayView rows = matrix.rowsIndex();
  IntegerConstArrayView columns = matrix.columns();
  RealConstArrayView mat_values = matrix.values();

  RealArrayView x_values = vector_x.values();
  RealConstArrayView b_values = vector_b.values();

  Integer nb_row = matrix.nbRow();
  //info() << " RELAX nb_relax=" << nb_relax << " nb_row=" << nb_row
  //       << " point_type=" << point_type;
  for( Integer row=0; row<nb_row; ++row ){
    Real diag = mat_values[rows[row]];
    if (math::isZero(diag))
      continue;
    Real res = b_values[row];
    for( Integer j=rows[row]+1; j<rows[row+1]; ++j ){
      Integer col = columns[j];
      res -= mat_values[j] * x_values[col];
    }
    x_values[row] = res / diag;
  }

  for( Integer row=nb_row-1; row>-1; --row ){
    Real diag = mat_values[rows[row]];
    if (math::isZero(diag))
      continue;
    Real res = b_values[row];
    for( Integer j=rows[row]+1; j<rows[row+1]; ++j ){
      Integer col = columns[j];
      res -= mat_values[j] * x_values[col];
    }
    x_values[row] = res / diag;
  }

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMG::
_solve(const Vector& vector_b,Vector& vector_x,Integer level)
{
  AMGLevel* current_level = m_levels[level];
  Integer vector_size = vector_b.size();
  Integer nb_coarse = current_level->nbCoarsePoint();
  Matrix fine_matrix = current_level->fineMatrix();
  Matrix restriction_matrix = current_level->restrictionMatrix();
  Matrix coarse_matrix = current_level->coarseMatrix();
  Matrix prolongation_matrix = current_level->prolongationMatrix();

  Integer new_nb_row = nb_coarse;
  Vector new_b(new_nb_row);
  Vector new_x(new_nb_row);
  Vector tmp(vector_size);

  MatrixOperation mat_op;

  bool is_final_level = (level+1) == m_levels.size();

  bool use_gauss_seidel = false;
  Integer nb_relax1 = 2;
  Real jacobi_weight = 2.0 / 3.0;
  if (use_gauss_seidel){
    for( Integer i=0; i<nb_relax1; ++i )
      _relaxGaussSeidel(fine_matrix,vector_b,vector_x,TYPE_FINE,current_level->pointsType());
    for( Integer i=0; i<nb_relax1; ++i )
      _relaxGaussSeidel(fine_matrix,vector_b,vector_x,TYPE_COARSE,current_level->pointsType());
  }
  else{
    //info() << "BEFORE SMOOTH";
    //_printResidualInfo(fine_matrix,vector_b,vector_x);
    for( Integer i=0; i<nb_relax1; ++i ){
      //_relaxSymmetricGaussSeidel(fine_matrix,vector_b,vector_x);
      _relaxJacobi(fine_matrix,vector_b,vector_x,jacobi_weight);
    }
    //info() << "AFTER SMOOTH";
    //_printResidualInfo(fine_matrix,vector_b,vector_x);
    //_relax(fine_matrix,vector_b,vector_x,nb_relax1);
  }
  
  // Restreint le nouveau b à partir du b actuel
  // b(k+1) = I * (b(k) - A * x)
  {
    OStringStream ostr;
    mat_op.matrixVectorProduct(fine_matrix,vector_x,tmp);
    //ostr() << "\nCOARSE_B TMP(A*x) level=" << level << " ";
    //tmp.dump(ostr());
    mat_op.negateVector(tmp);
    mat_op.addVector(tmp,vector_b);
    //ostr() << "\nCOARSE_B TMP(b-A*x) level=" << level << " ";
    //tmp.dump(ostr());
    mat_op.matrixVectorProduct(restriction_matrix,tmp,new_b);
    //ostr() << "\nCOARSE_B level=" << level << " ";
    //new_b.dump(ostr());
    info() << ostr.str();
  }

  //mat_op.matrixVectorProduct(transpose_prolongation_matrix,vector_x,tmp);
    
  // Si niveau final atteint, résoud la matrice.
  // Sinon, continue en restreignant à nouvea la matrice
  if (is_final_level){

    //info() << " SOLVE FINAL LEVEL";
    
    if (1){
      DirectSolver ds;
      ds.solve(coarse_matrix,new_b,new_x);
      //_printResidualInfo(coarse_matrix,new_b,new_x);
    }
    else{
      Real epsilon = 1.0e-14;
      DiagonalPreconditioner p(coarse_matrix);
      ConjugateGradientSolver solver;
      //mat_op.matrixVectorProduct(restriction_matrix,vector_x,new_x);
      new_x.values().fill(0.0);
      solver.solve(coarse_matrix,new_b,new_x,epsilon,&p);
      OStringStream ostr;
      //ostr() << " COARSE_B=";
      //new_b.dump(ostr());
      //ostr() << "\nCOARSE_X=";
      //new_x.dump(ostr());
      //if (m_is_verbose)
      info() << "SOLVE COARSE MATRIX nb_iter=" << solver.nbIteration();
      //       << ostr.str();      
      //_printResidualInfo(coarse_matrix,new_b,new_x);
    }
  }
  else{
    new_x.values().fill(0.0);
    _solve(new_b,new_x,level+1);
  }

  // Interpole le nouveau x à partir de la solution trouvée
  // x(k) = x(k) + tI * x(k+1)
  mat_op.matrixVectorProduct(prolongation_matrix,new_x,tmp);
  mat_op.addVector(vector_x,tmp);
  /*{
    OStringStream ostr;
    vector_x.dump(ostr());
    info() << "NEW_X level=" << level << " X=" << ostr.str();
    _printResidualInfo(fine_matrix,vector_b,vector_x);
    }*/

  // Relaxation de richardson
  if (use_gauss_seidel){
    for( Integer i=0; i<nb_relax1; ++i )
      _relaxGaussSeidel(fine_matrix,vector_b,vector_x,TYPE_FINE,current_level->pointsType());
    for( Integer i=0; i<nb_relax1; ++i )
      _relaxGaussSeidel(fine_matrix,vector_b,vector_x,TYPE_COARSE,current_level->pointsType());
  }
  else{
    //info() << "BEFORE SMOOTH 2";
    //_printResidualInfo(fine_matrix,vector_b,vector_x);
    for( Integer i=0; i<nb_relax1; ++i ){
      //_relaxSymmetricGaussSeidel(fine_matrix,vector_b,vector_x);
      _relaxJacobi(fine_matrix,vector_b,vector_x,jacobi_weight);
    }
    //info() << "AFTER SMOOTH 2";
    //_printResidualInfo(fine_matrix,vector_b,vector_x);
    //_relax(fine_matrix,vector_b,vector_x,nb_relax2);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMG::
_printResidualInfo(const Matrix& a,const Vector& b,const Vector& x)
{
  OStringStream ostr;
  Vector tmp(b.size());
  // tmp = b - Ax
  MatrixOperation mat_op;
  mat_op.matrixVectorProduct(a,x,tmp);
  //ostr() << "\nAX=";
  //tmp.dump(ostr());
  mat_op.negateVector(tmp);
  mat_op.addVector(tmp,b);
  Real r = mat_op.dot(tmp);
  if (0){
    Integer v = math::min(10,tmp.size());
    for( Integer i=0; i<v; ++i )
      info() << "R_" << i << " = " << tmp.values()[i];
  }
  info() << " AMG_RESIDUAL_NORM="  << r << " sqrt=" << math::sqrt(r);

  //ostr() << "\nR=";
  //tmp.dump(ostr());
  //info() << " AMG_RESIDUAL_NORM="  << r <<  " AMG_RESIDUAL=" << ostr.str();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class PointInfo
{
 public:
  PointInfo() : m_lambda(0), m_index(0) {}
  PointInfo(Integer lambda,Integer index) : m_lambda(lambda), m_index(index){}
  Integer m_lambda;
  Integer m_index;
  /*!
   * Le tri est fait pour que le premier point de la liste est celui
   * avec le lambda maximal et d'indice minimal.
   */
  bool operator<(const PointInfo& rhs) const
  {
    if (m_lambda==rhs.m_lambda)
      return m_index<rhs.m_index;
    return (m_lambda>rhs.m_lambda);    
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMGLevel::
printLevelInfo()
{
  _printLevelInfo(m_prolongation_matrix);
  _printLevelInfo(m_coarse_matrix);
}

void AMGLevel::
_printLevelInfo(Matrix matrix)
{
  OStringStream ostr;
  Integer nb_row = matrix.nbRow();
  Integer nb_column = matrix.nbColumn();
  
  IntegerConstArrayView rows = matrix.rowsIndex();
  //IntegerConstArrayView columns = matrix.columns();
  RealConstArrayView values = matrix.values();
  Integer nb_value = values.size();

  Real max_val = 0.0;
  Real min_val = 0.0;
  if (nb_value>0){
    max_val = values[0];
    min_val = values[0];
  }

  Real max_row_sum = 0.0;
  Real min_row_sum = 0.0;
  for( Integer row=0; row<nb_row; ++row ){
    Real row_sum = 0.0;
    for( Integer z=rows[row] ,zs=rows[row+1]; z<zs; ++z ){
      //Integer col = columns[z];
      Real v = values[z];
      if (v>max_val)
        max_val = v;
      if (v<min_val)
        min_val = v;
      row_sum += v;
    }
    if (row==0){
      max_row_sum = row_sum;
      min_row_sum = row_sum;
    }
    if (row_sum>max_row_sum)
      max_row_sum = row_sum;
    if (row_sum<max_row_sum)
      min_row_sum = row_sum;
  }

  Real sparsity = ((Real)nb_value) / ((Real)nb_row * (Real)nb_column);

  ostr() << "level=" << m_level
         << " nb_row=" << nb_row
         << " nb_col=" << nb_column
         << " nb_nonzero=" << nb_value
         << " sparsity=" << sparsity
         << " min=" << min_val
         << " max=" << max_val
         << " min_row=" << min_row_sum
         << " max_row=" << max_row_sum;
 
  info() << "INFO: " << ostr.str();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMGLevel::
_buildCoarsePoints(Real alpha,
                   RealArray& rows_max_val,
                   UniqueArray< SharedArray<Integer> >& depends,
                   IntegerArray& weak_depends
)
{
  IntegerConstArrayView rows_index = m_fine_matrix.rowsIndex();
  IntegerConstArrayView columns = m_fine_matrix.columns();
  RealConstArrayView mat_values = m_fine_matrix.values();
  Integer nb_row = m_fine_matrix.nbRow();

  Int32UniqueArray lambdas(nb_row);
  lambdas.fill(0);
  UniqueArray< SharedArray<Integer> > influences(nb_row);
  depends.resize(nb_row);
  //UniqueArray<IntegerUniqueArray> weak_depends(nb_row);
  m_points_type.resize(nb_row);
  m_points_type.fill(TYPE_UNDEFINED);

  weak_depends.resize(mat_values.size());
  weak_depends.fill(0);

  bool type_hypre = true;
  // Valeurs de chaque ligne qui influence
  rows_max_val.resize(nb_row);
  for( Integer row=0; row<nb_row; ++row ){
    Real max_val = 0.0;
    Real min_val = 0.0;
    Real diag_val = mat_values[rows_index[row]];
    // Cherche le max (en valeur absolue) de la colonne, autre que la diagonale
    for( Integer z=rows_index[row]+1 ,zs=rows_index[row+1]; z<zs; ++z ){
      //Real mv = math::abs(mat_values[z]);
      Real mv = mat_values[z];
      if (!type_hypre)
        mv = math::abs(mv);
      if (mv>max_val)
        max_val = mv;
      if (mv<min_val)
        min_val = mv;
    }
    // Prend tous les éléments supérieurs à alpha * max_val
    //rows_max_val[row] = max_val * alpha;
    if (type_hypre){
      if (diag_val<0.0)
        rows_max_val[row] = max_val * alpha;
      else
        rows_max_val[row] = min_val * alpha;
    }
    else
      rows_max_val[row] = max_val * alpha;
  }


  for( Integer row=0; row<nb_row; ++row ){
    // Prend tous les éléments supérieurs à max_val
    Real max_val = rows_max_val[row];
    Real diag_val = mat_values[rows_index[row]];
    for( Integer z=rows_index[row]+1 ,zs=rows_index[row+1]; z<zs; ++z ){
      //Real mv = math::abs(mat_values[z]);
      Real mv = mat_values[z];
      if (type_hypre){
        if (diag_val<0.0){
          if (mv>max_val){
            Integer column = columns[z];
            if (m_is_verbose)
              info() << " ADD INFLUENCE: ROW=" << row << " COL="  << column;
            ++lambdas[column];
            depends[row].add(column);
            influences[column].add(row);
            weak_depends[z] = 2;
          }
          else
            weak_depends[z] = 1;
        }
        else{
          if (mv<max_val){
            Integer column = columns[z];
            if (m_is_verbose)
              info() << " ADD INFLUENCE: ROW=" << row << " COL="  << column;
            ++lambdas[column];
            depends[row].add(column);
            influences[column].add(row);
            weak_depends[z] = 2;
          }
          else
            weak_depends[z] = 1;
        }
      }
      else{
        if (math::abs(mv)>max_val){
          Integer column = columns[z];
          if (m_is_verbose)
            info() << " ADD INFLUENCE: ROW=" << row << " COL="  << column;
          ++lambdas[column];
          depends[row].add(column);
          influences[column].add(row);
          weak_depends[z] = 2;
        }
        else{
          weak_depends[z] = 1;
        }
      }
      //else
      //weak_depends[row].add(column);
    }

  }
  
  if (0){
    OStringStream ostr;
    Integer index = 0;
    int n = math::min(nb_row,800);
    ostr() << "GRAPH\n";
    for( Integer i=0; i<n; ++i ){
      ostr() << " GRAPH I=" << i << " ";
      for( Integer j=0; j<depends[i].size(); ++j ){
        ++index;
        ostr() << " " << depends[i][j];
      }
      ostr() << " index=" << index << '\n';
    }
    ostr() << "\n MAXTRIX\n";
    index = 0;
    for( Integer i=0; i<n; ++i ){
      ostr() << "MATRIX I=" << i << " ";
      for( Integer j=rows_index[i]; j<rows_index[i+1]; ++j ){
        ++index;
        ostr() << " " << columns[j] << " " << mat_values[j];
      }
      ostr() << " index=" << index << '\n';
    }
    info() << ostr.str();
  }

  Integer nb_done = 0;
  Integer nb_iter = 0;
  Integer nb_fine = 0;
  Integer nb_coarse = 0;
  m_is_verbose = false;
  {
    // Marque comme point fin tous les points n'ayant aucune dépendance
    for( Integer row=0; row<nb_row; ++row ){
      if (depends[row].size()==0){
        m_points_type[row] = TYPE_FINE;
        ++nb_done;
        if (m_is_verbose)
          info() << "FIRST MARK FINE point=" << row;
      }
    }

    // Les points qui n'influencent personne sont forcément fins.
    for( Integer row=0; row<nb_row; ++row ){
      if (m_points_type[row]!=TYPE_FINE && lambdas[row]<=0){
        m_points_type[row]=TYPE_FINE;
        ++nb_done;
        if (m_is_verbose)
          info() << "INIT MARK FINE NULL MEASURE point=" << row << " measure=" << lambdas[row];
        for( Integer j=0, js=depends[row].size(); j<js; ++j ){
          Integer col = depends[row][j];
          if (m_points_type[col]!=TYPE_FINE)
            if (col<row){
              ++lambdas[col];
              if (m_is_verbose)
                printf("ADD MEASURE NULL point=%d measure=%d\n",(int)col,lambdas[col]);
            }
        }
      }
    }

    typedef std::set<PointInfo> PointSet;
    PointSet undefined_points;
    for( Integer i=0; i<nb_row; ++i ){
      if (m_points_type[i]==TYPE_UNDEFINED)
        undefined_points.insert(PointInfo(lambdas[i],i));
    }

    while(nb_done<nb_row && nb_iter<100000){
      ++nb_iter;
      //for( PointSet::const_iterator i(undefined_points.data()); i!=undefined_points.end(); ++i ){
        //info() << " SET index=" << i->m_index << " value=" << i->m_lambda;
      //}
      // Prend le lambda max et note le point C
      //Integer max_value = -1;
      //Integer max_value_index = -1;
      //for( Integer i=0; i<nb_row; ++i ){
      //if (lambdas[i]>max_value && points_type[i]==TYPE_UNDEFINED){
      //  max_value = lambdas[i];
      //  max_value_index = i;
      //}
      //}
      if (undefined_points.empty())
        fatal() << "Undefined points is empty";
      PointSet::iterator max_point = undefined_points.begin();
      Integer max_value_index = max_point->m_index;
      Integer max_value = max_point->m_lambda;
      m_points_type[max_value_index] = TYPE_COARSE;
      ++nb_done;
      ++nb_coarse;
      undefined_points.erase(max_point);
      if (m_is_verbose)
        cout << "MARK COARSE point=" << max_value_index
             << " measure=" << max_value
             << " left=" << (nb_row-nb_done)
             << "\n";
      IntegerConstArrayView point_influences = influences[max_value_index];
      for( Integer i=0, is=point_influences.size(); i<is; ++i ){
        //for( Integer i=0, is=depends[max_value_index].size(); i<is; ++i ){
        Integer pt = point_influences[i];
        //Integer pt = depends[max_value_index][i];
        if (m_points_type[pt]==TYPE_UNDEFINED){
          m_points_type[ pt ] = TYPE_FINE;
          ++nb_done;
          ++nb_fine;
          undefined_points.erase(PointInfo(lambdas[pt],pt));
          if (m_is_verbose)
            cout << "MARK FINE point=" << pt
                 << " measure=" << lambdas[pt]
                 << " left=" << (nb_row-nb_done)
                 << "\n";
          for( Integer z=0, zs=depends[pt].size(); z<zs; ++z ){
            Integer pt2 = depends[pt][z];
            //for( Integer z=0, zs=point_influences.size(); z<zs; ++z ){
            //Integer pt2 = point_influences[z];
            if (m_points_type[pt2]==TYPE_UNDEFINED){
              undefined_points.erase(PointInfo(lambdas[pt2],pt2));
              ++lambdas[pt2];
              undefined_points.insert(PointInfo(lambdas[pt2],pt2));
            }
          }
        }
      }
      for( Integer i=0, is=depends[max_value_index].size(); i<is; ++i ){
        Integer pt3 = depends[max_value_index][i];
        if (m_points_type[pt3]==TYPE_UNDEFINED){
          undefined_points.erase(PointInfo(lambdas[pt3],pt3));
          Integer n = lambdas[pt3];
          if (n<0)
            info() << "N < 0";
          --lambdas[pt3];
          undefined_points.insert(PointInfo(lambdas[pt3],pt3));
        }
      }
      if (m_is_verbose)
        info() << "LAMBDA MAX = " << max_value << " index=" << max_value_index << " nb_done=" << nb_done;
    }
  }

  if (m_is_verbose)
    info() << "NB ROW=" << nb_row << " nb_done=" << nb_done << " nb_fine=" << nb_fine
           << " nb_coarse=" << nb_coarse << " nb_iter=" << nb_iter;
  if (nb_done!=nb_row)
    fatal() << "Can not find all COARSE or FINE points nb_done=" << nb_done << " nb_point=" << nb_row;

  {
    // Maintenant, il faut etre certain que deux connections F-F ont au moins un
    // point C en commun. Si ce n'est pas le cas, le premier F est changé en C
    //info() << "SECOND PASS !!!";
    Int32UniqueArray points_marker(nb_row);
    points_marker.fill(-1);
    Integer ci_tilde_mark = -1;
    Integer ci_tilde = -1;
    bool C_i_nonempty = false;
    for( Integer row=0; row<nb_row; ++row ){
      if ( (ci_tilde_mark |= row) )
        ci_tilde = -1;
      if (m_points_type[row]==TYPE_FINE){
        for( Integer z=0, zs=depends[row].size(); z<zs; ++z ){
          //for( Integer z=rows_index[row] ,zs=rows_index[row+1]; z<zs; ++z ){
          //Integer col = columns[z];
          Integer col = depends[row][z];
          if (m_points_type[col]==TYPE_COARSE)
            points_marker[col] = row;
        }
        for( Integer z=0, zs=depends[row].size(); z<zs; ++z ){
          //for( Integer z=rows_index[row] ,zs=rows_index[row+1]; z<zs; ++z ){
          //Integer col = columns[z];
          Integer col = depends[row][z];
          if (m_points_type[col]==TYPE_FINE){
            bool set_empty = true;
            for( Integer z2=0, zs2=depends[row].size(); z2<zs2; ++z2 ){
              //for( Integer z2=rows_index[row] ,zs2=rows_index[row+1]; z2<zs2; ++z2 ){
              //Integer col2 = columns[z2];
              Integer col2 = depends[row][z2];
              if (points_marker[col2]==row){
                set_empty = false;
                break;
              }
            }
            if (set_empty){
              if (C_i_nonempty){
                m_points_type[row] = TYPE_COARSE;
                //printf("SECOND PASS MARK COARSE1 point=%d\n",row);
                if (ci_tilde>-1){
                  m_points_type[ci_tilde]= TYPE_FINE;
                  if (m_is_verbose)
                    printf("SECOND PASS MARK FINE point=%d\n",ci_tilde);
                  ci_tilde = -1;
                }
                C_i_nonempty = false;
              }
              else{
                ci_tilde = col;
                ci_tilde_mark = row;
                m_points_type[col] = TYPE_COARSE;
                if (m_is_verbose)
                  printf("SECOND PASS MARK COARSE2 point=%d\n",col);
                C_i_nonempty = true;
                --row;
                break;
              }
            }
          }
        }
      }
    }
  }


  if (0){
    // Lecture depuis Hypre
    static int matrix_number = 0;
    ++matrix_number;
    info() << "READ HYPRE CF_marker n=" << matrix_number;
    StringBuilder fname("CF_marker-");
    fname += matrix_number;
    std::ifstream ifile(fname.toString().localstr());
    Integer nb_read_point = 0;
    ifile >> ws >> nb_read_point >> ws;
    if (nb_read_point!=nb_row)
      fatal() << "Bad number of points for reading Hypre CF_marker read=" << nb_read_point
              << " expected=" << nb_row << " matrix_number=" << matrix_number;
    nb_coarse = 0;
    nb_fine = 0;
    for( Integer i=0; i<nb_row; ++i ){
      int pt = 0;
      ifile >> pt;
      if (!ifile)
        fatal() << "Can not read marker point number=" << i;
      if (pt==(-1) || pt==(-3)){
        m_points_type[i] = TYPE_FINE;
        ++nb_fine;
      }
      else if (pt==1){
        m_points_type[i] = TYPE_COARSE;
        ++nb_coarse;
      }
      else
        fatal() << "Bad value read=" << pt << " expected 1 or -1";
    }
  }

  // Vérifie que tous les points fins ont au moins un point qui influence
  nb_coarse = 0;
  for( Integer i=0; i<nb_row; ++i ){
    if (m_points_type[i]==TYPE_UNDEFINED)
      fatal() << " Point " << i << " is undefined";
    if (m_points_type[i]!=TYPE_FINE){
      ++nb_coarse;
      continue;
    }
#if 0
    bool is_ok = false;
    //info() << "CHECK POINT point=" << i
    //       << " depend_size=" << depends[i].size();
    for( Integer z=0, zs=depends[i].size(); z<zs; ++z ){
      if (m_points_type[depends[i][z]]==TYPE_COARSE){
        is_ok = true;
        break;
      }
    }
    //if (!is_ok)
    //  fatal() << " Point " << i << " has no coarse point";
#endif
  }
      
  if (m_is_verbose){
    OStringStream ostr;
    for( Integer i=0; i<nb_row; ++i ){
      ostr() << " POINT i=" << i << " type=" << m_points_type[i] << " depends=";
      for( Integer j=0, js=depends[i].size(); j<js; ++j )
        ostr() << depends[i][j] << ' ';
      ostr() << '\n';
    }
    info() << ostr.str();
  }

  nb_fine = nb_row - nb_coarse;
  Integer graph_size = 0;
  for( Integer i=0; i<nb_row; ++i )
    graph_size += depends[i].size();

  info() << " NB COARSE=" << nb_coarse << " NB FINE=" << nb_fine
         << " MAXTRIX NON_ZEROS=" << m_fine_matrix.rowsIndex()[nb_row]
         << " GRAPH_SIZE=" << graph_size;
  bool dump_matrix = false;
  bool has_error = false;
  if (nb_fine==0 || graph_size==0){
    has_error = true;
    dump_matrix = true;
  }

  if (dump_matrix){
    OStringStream ostr;
    Integer index = 0;
    int n = math::min(nb_row,40);
    if (0){
      ostr() << "GRAPH\n";
      for( Integer i=0; i<n; ++i ){
        ostr() << " GRAPH I=" << i << " ";
        for( Integer j=0; j<depends[i].size(); ++j ){
          ++index;
          ostr() << " " << depends[i][j];
        }
        ostr() << " index=" << index << '\n';
      }
    }
    ostr() << "\n MAXTRIX\n";
    index = 0;
    for( Integer i=0; i<n; ++i ){
      ostr() << "MATRIX I=" << i << " ";
      for( Integer j=rows_index[i]; j<rows_index[i+1]; ++j ){
        ++index;
        ostr() << " " << columns[j] << " " << mat_values[j];
      }
      ostr() << " index=" << index << '\n';
    }
    info() << ostr.str();
  }
  if (has_error)
    throw FatalErrorException("AMGLevel::_buildCoarsePoints");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMGLevel::
buildLevel(Matrix matrix,Real alpha)
{
  //Integer nb_row = matrix.nbRow();
  //if (nb_row<20)
  //return;

  m_fine_matrix = matrix;

  //bool is_verbose = false;
  matrix.sortDiagonale();

  IntegerUniqueArray points_type;
  RealUniqueArray rows_max_val;
  UniqueArray< SharedArray<Integer> > depends;
  IntegerUniqueArray weak_depends;

  _buildCoarsePoints(alpha,rows_max_val,depends,weak_depends);
  _buildInterpolationMatrix(rows_max_val,depends,weak_depends);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMGLevel::
_buildInterpolationMatrix(RealConstArrayView rows_max_val,
                          UniqueArray< SharedArray<Integer> >& depends,
                          IntegerArray& weak_depends)
{
  ARCANE_UNUSED(rows_max_val);

  IntegerConstArrayView rows_index = m_fine_matrix.rowsIndex();
  IntegerConstArrayView columns = m_fine_matrix.columns();
  RealConstArrayView mat_values = m_fine_matrix.values();
  Integer nb_row = m_fine_matrix.nbRow();

  IntegerUniqueArray points_in_coarse(nb_row);
  points_in_coarse.fill(-1);
  Integer nb_coarse = 0;
  {
    //Integer index = 0;
    nb_coarse = 0;
    for( Integer i=0; i<nb_row; ++i ){
      if (m_points_type[i]==TYPE_COARSE){
        points_in_coarse[i] = nb_coarse;
        ++nb_coarse;
      }
    }
  }
  bool type_hypre = true;

  // Maintenant,calcule les éléments de la matrice d'influence
  IntegerUniqueArray prolongation_matrix_columns;
  RealUniqueArray prolongation_matrix_values;
  IntegerUniqueArray prolongation_matrix_rows_size(nb_row);

  for( Integer row = 0; row<nb_row; ++row ){
    Integer nb_column = 0;
    if (m_points_type[row]==TYPE_FINE){
      Real weak_connect_sum = 0.0;
      //Real max_value = rows_max_val[row];
      Real diag = mat_values[rows_index[row]];
      Real sign = 1.0;
      if (diag<0.0)
        sign = -1.0;
      for( Integer z=rows_index[row]+1 ,zs=rows_index[row+1]; z<zs; ++z ){
        //Integer column = columns[z];
        if (weak_depends[z]==1){
          Real mv = mat_values[z];
          if (type_hypre)
            weak_connect_sum += mv;
          else{
            //weak_connect_sum += math::abs(mv);
            weak_connect_sum += mv;
          }
          //if (m_is_verbose || row<=5)
          //info() << "ADD WEAK_SUM mv=" << mv << " sum=" << weak_connect_sum
          //         << " row=" << row << " column=" << column;
        }
      }
      if (m_is_verbose)
        info() << "ROW row=" << row << " weak_connect_sum=" << weak_connect_sum;
      //for( Integer z=0, zs=depends[row].size(); z<zs; ++ z){
      //Integer j_column = dependcolumns[z];
      for( Integer z=rows_index[row]+1 ,zs=rows_index[row+1]; z<zs; ++z ){
        Integer j_column = columns[z];
        if (m_points_type[j_column]!=TYPE_COARSE)
          continue;
        if (weak_depends[z]!=2)
         continue;
        Real num_add = 0.0;
        Real mv = mat_values[z];
        for( Integer z2=0, zs2=depends[row].size(); z2<zs2; ++ z2){
          Integer k_column = depends[row][z2];
          //if (m_is_verbose || row<=5)
          //info() << "CHECK K_COLUMN row=" << row << " col=" << k_column << " val=" << mv;
          if (m_points_type[k_column]!=TYPE_FINE)
            continue;
          Real sum_coarse = 0.0;
          //if (m_is_verbose || row<=5)
          //info() << "CHECK COARSE row=" << row;
          //for( Integer z3=rows_index[row]+1 ,zs3=rows_index[row+1]; z3<zs3; ++z3 ){
          //Integer m_column = columns[z3];
          for( Integer z3=0 ,zs3=depends[row].size(); z3<zs3; ++z3 ){
            Integer m_column = depends[row][z3];
            //if (m_is_verbose || row<=5)
            //info() << "CHECK COLUMN column=" << m_column;
            if (m_points_type[m_column]==TYPE_COARSE){
              Real w = m_fine_matrix.value(k_column,m_column);
              //if (m_is_verbose || row<=5)
              //info() << "ADD SUM k=" << k_column << " m="<< m_column << " w=" << w;
              //if (math::isZero(w)){
              //fatal() << "WEIGHT is null k=" << k_column << " m=" << m_column << " row=" << row
              //        << " j=" << j_column;
              //}
              if (type_hypre){
                if (w*sign<0.0)
                  sum_coarse += w;
              }
              else{
                sum_coarse += math::abs(w);
                //if (w*sign<0.0)
                //sum_coarse += w;
              }
            }
          }
          Real to_add = 0.0;
          if (!math::isZero(sum_coarse)){
            Real akj = m_fine_matrix.value(k_column,j_column);
            bool do_add = false;
            if (type_hypre){
              if ((akj*sign)<0.0)
                do_add = true;
            }
            else{
              //if ((akj*sign)<0.0)
              //do_add = true;
              akj = math::abs(akj);
              do_add = true;
            }
            if (do_add)
            //fatal() << "SUM_WEIGHT is null k=" << k_column << " row=" << row << " j=" << j_column;
              to_add = math::divide(m_fine_matrix.value(row,k_column) * akj,sum_coarse);
          }
          num_add += to_add;
        }
        Real weight = - ( mv + num_add ) / ( diag + weak_connect_sum);
        Integer new_column = points_in_coarse[j_column];
        //if (m_is_verbose || row<=5)
        //info() << " ** WEIGHT row=" << row << " j_column=" << j_column
        //         << " weight=" << weight << " num_add=" << num_add << " mv=" << mv << " new_column=" << new_column
        //         << " diag=" << diag << " weak_sum=" << weak_connect_sum
        //         << " diag+wk=" << (diag+weak_connect_sum);
        if (new_column>=nb_coarse || new_column<0)
          fatal() << " BAD COLUMN for fine point column=" << new_column << " nb=" << nb_coarse
                  << " jcolumn=" << j_column;
        prolongation_matrix_columns.add(new_column);
        prolongation_matrix_values.add(weight);
        ++nb_column;
      }
    }
    else{
      // Point grossier, met 1.0 dans la diagonale
      Integer column = points_in_coarse[row];
      if (column>=nb_coarse || column<0)
        fatal() << " BAD COLUMN for coarse point j=" << column << " nb=" << nb_coarse
                << " row=" << row;
      prolongation_matrix_columns.add(column);
      prolongation_matrix_values.add(1.0);
      ++nb_column;
    }
    prolongation_matrix_rows_size[row] = nb_column;
  }

  m_prolongation_matrix = Matrix(nb_row,nb_coarse);
  m_prolongation_matrix.setRowsSize(prolongation_matrix_rows_size);
  //info() << "PROLONGATION_MATRIX_SIZE=" << m_prolongation_matrix.rowsIndex()[nb_row];
  m_prolongation_matrix.setValues(prolongation_matrix_columns,prolongation_matrix_values);

  if (0){
    OStringStream ostr;
    Integer index = 0;
    int n = math::min(nb_row,50);
    IntegerConstArrayView p_rows(m_prolongation_matrix.rowsIndex());
    IntegerConstArrayView p_columns(m_prolongation_matrix.columns());
    RealConstArrayView p_values(m_prolongation_matrix.values());
    for( Integer i=0; i<n; ++i ){
      ostr() << "PROLONG I=" << i << " ";
      for( Integer j=p_rows[i]; j<p_rows[i+1]; ++j ){
        ++index;
        ostr() << " " << p_columns[j] << " " << p_values[j];
      }
      ostr() << " index=" << index << '\n';
    }
    info() << "PROLONG\n" << ostr.str();
  }

  MatrixOperation2 mat_op2;
  if (1)
    m_restriction_matrix = mat_op2.transposeFast(m_prolongation_matrix);
  else
    m_restriction_matrix = mat_op2.transpose(m_prolongation_matrix);
  if (m_is_verbose){ 
    OStringStream ostr;
    ostr() << "PROLONGATION_MATRIX ";
    m_prolongation_matrix.dump(ostr());
    ostr() << '\n';
    ostr() << "RESTRICTION_MATRIX ";
    m_restriction_matrix.dump(ostr());
    info() << ostr.str();
  }
  //info() << " ** TOTAL SUM=" << total_sum;
  // Calcule la matrice grossiere Ak+1 = I * Ak * tI
  //MatrixOperation mat_op;
  bool old = false;
  if (old){
    Matrix n1 = mat_op2.matrixMatrixProductFast(m_fine_matrix,m_prolongation_matrix);
    if (m_is_verbose){
      OStringStream ostr;
      n1.dump(ostr());
      info() << "N1_MATRIX " << ostr.str();
    }
    m_coarse_matrix = mat_op2.matrixMatrixProductFast(m_restriction_matrix,n1);
  }
  else
    m_coarse_matrix = mat_op2.applyGalerkinOperator2(m_restriction_matrix,m_fine_matrix,m_prolongation_matrix);
  if (m_is_verbose){
    OStringStream ostr;
    m_coarse_matrix.dump(ostr());
    info() << "level= " << m_level << " COARSE_MATRIX=" << ostr.str();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AMGPreconditioner::
~AMGPreconditioner()
{
  delete m_amg;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMGPreconditioner::
apply(Vector& out_vec,const Vector& vec)
{
  m_amg->solve(vec,out_vec);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMGPreconditioner::
build(const Matrix& matrix)
{
  delete m_amg;
  m_amg = new AMG(m_trace_mng);
  m_amg->build(matrix);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AMGSolver::
~AMGSolver()
{
  delete m_amg;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMGSolver::
build(const Matrix& matrix)
{
  delete m_amg;
  m_amg = new AMG(m_trace_mng);
  m_amg->build(matrix);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMGSolver::
solve(const Vector& vector_b,Vector& vector_x)
{
  m_amg->solve(vector_b,vector_x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
