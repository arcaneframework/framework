// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*
 *  CNC: Concurrent Number Cruncher
 *  Copyright (C) 2008 GOCAD/ASGA, INRIA/ALICE
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 *  If you modify this software, you should include a notice giving the
 *  name of the person performing the modification, the date of modification,
 *  and the reason for such modification.
 *
 *  Contact: Luc Buatois
 *
 *     buatois@gocad.org
 *
 *     ASGA-INPL Bt. G
 *     Rue du Doyen Marcel Roubault - BP 40
 *     54501 VANDOEUVRE LES NANCY
 *     FRANCE
 *
 *  Note that the GNU General Public License does not permit incorporating
 *  the Software into proprietary programs. 
 */

#include "arcane/aleph/AlephArcane.h"
#include "arcane/aleph/cuda/AlephCuda.h"



ARCANE_BEGIN_NAMESPACE

//---------------------------------------------------------------------------//

CNC_Matrix::CNC_Matrix(long m, long n, Storage storage) {
  storage_ = NONE ;
  allocate(m,n,storage,false) ;
}

//---------------------------------------------------------------------------//

CNC_Matrix::CNC_Matrix(long n, Storage storage, bool symmetric_storage) {
  storage_ = NONE ;
  allocate(n,n,storage,symmetric_storage) ;
}

//---------------------------------------------------------------------------//

CNC_Matrix::CNC_Matrix( long n ) {
  m_ = 0 ;
  n_ = 0 ;
  diag_size_ = 0 ;
  
  row_ = NULL ;
  column_ = NULL ;
  diag_ = NULL ;
  
  storage_ = ROWS ;
  allocate(n,n,storage_,false) ;
}

//---------------------------------------------------------------------------//

CNC_Matrix::~CNC_Matrix() {
  deallocate() ;
}

//---------------------------------------------------------------------------//

CNC_Matrix::CNC_Matrix() {
  m_ = 0 ;
  n_ = 0 ;
  diag_size_ = 0 ;
  
  row_ = NULL ;
  column_ = NULL ;
  diag_ = NULL ;
  
  storage_ = NONE ;
  rows_are_stored_ = false ;
  columns_are_stored_ = false ;
  symmetric_storage_ = false ;
  symmetric_tag_ = false ;
}

//---------------------------------------------------------------------------//

long CNC_Matrix::m() const {return m_ ;  }

long CNC_Matrix::n() const {return n_ ;  }

long CNC_Matrix::diag_size() const {return diag_size_ ;}


long CNC_Matrix::nnz() const {
  long result = 0 ;
  if(rows_are_stored()) {
    for(long i=0; i<m(); i++) {
      result += row(i).nb_coeffs() ;
    }
  } else if(columns_are_stored()) {
    for(long j=0; j<n(); j++) {
      result += column(j).nb_coeffs() ;
    }
  } else {
  }
  return result ;
}


bool CNC_Matrix::rows_are_stored() const {
    return rows_are_stored_ ;
  }

bool CNC_Matrix::columns_are_stored() const {
    return columns_are_stored_ ;
  }

CNC_Matrix::Storage CNC_Matrix::storage() const {
    return storage_ ;
  }

bool CNC_Matrix::has_symmetric_storage() const {
    return symmetric_storage_ ;
  }

bool CNC_Matrix::is_square() const {
    return (m_ == n_) ;
  }

bool CNC_Matrix::is_symmetric() const {
    return (symmetric_storage_ || symmetric_tag_) ;
  }

  /**
   * For symmetric matrices that are not stored in symmetric mode,
   * one may want to give a hint that the matrix is symmetric.
   */
void CNC_Matrix::set_symmetric_tag(bool x) {
    symmetric_tag_ = x ;
  }

  /**
   * storage should be one of ROWS, ROWS_AND_COLUMNS
   * @param i index of the row, in the range [0, m-1]
   */
CNCSparseRowColumn& CNC_Matrix::row(long i) {
    return row_[i] ;
  }

  /**
   * storage should be one of ROWS, ROWS_AND_COLUMNS
   * @param i index of the row, in the range [0, m-1]
   */
const CNCSparseRowColumn& CNC_Matrix::row(long i) const {
    return row_[i] ;
  }

  /**
   * storage should be one of COLUMN, ROWS_AND_COLUMNS
   * @param i index of the column, in the range [0, n-1]
   */
CNCSparseRowColumn& CNC_Matrix::column(long j) {
    return column_[j] ;
  }

  /**
   * storage should be one of COLUMNS, ROWS_AND_COLUMNS
   * @param i index of the column, in the range [0, n-1]
   */
const CNCSparseRowColumn& CNC_Matrix::column(long j) const {
    return column_[j] ;
  }
        
  /**
   * returns aii.
   */
double CNC_Matrix::diag(long i) const {
    return diag_[i] ;
}

/**
 * aij <- aij + val
 */
void CNC_Matrix::add(long i, long j, double val) {
  if(symmetric_storage_ && j > i) {
    return ;
  }
  if(i == j) {
    diag_[i] += val ;
  } 
  if(rows_are_stored_) {
    row(i).add(j, val) ;
  }
  if(columns_are_stored_) {
    column(j).add(i, val) ;
  }
}

//---------------------------------------------------------------------------//

void CNC_Matrix::sort() {
  if(rows_are_stored_) {
    for(long i=0; i<m_; i++) {
      row(i).sort() ;
    }
  }
  if(columns_are_stored_) {
    for(long j=0; j<n_; j++) {
      column(j).sort() ;
    }
  }
}

//---------------------------------------------------------------------------//

void CNC_Matrix::zero() {
  if(rows_are_stored_) {
    for(long i=0; i<m_; i++) {
      row(i).zero() ;
    }
  }
  if(columns_are_stored_) {
    for(long j=0; j<n_; j++) {
      column(j).zero() ;
    }
  }
  for(long i=0; i<diag_size_; i++) {
    diag_[i] = 0.0 ;
  }
}

//---------------------------------------------------------------------------//

void CNC_Matrix::clear() {
  if(rows_are_stored_) {
    for(long i=0; i<m_; i++) {
      row(i).clear() ;
    }
  }
  if(columns_are_stored_) {
    for(long j=0; j<n_; j++) {
      column(j).clear() ;
    }
  }
  for(long i=0; i<diag_size_; i++) {
    diag_[i] = 0.0 ;
  }
}

//---------------------------------------------------------------------------//
        
void CNC_Matrix::deallocate() {
  m_ = 0 ;
  n_ = 0 ;
  diag_size_ = 0 ;
  
  if ( row_ != NULL ) delete[] row_ ;
  if ( column_ != NULL ) delete[] column_ ;
  if ( diag_ != NULL ) delete[] diag_ ;
  row_ = NULL ;
  column_ = NULL ;
  diag_ = NULL ;
  
  storage_ = NONE ;
  rows_are_stored_    = false ;
  columns_are_stored_ = false ;
  symmetric_storage_  = false ;
}

//---------------------------------------------------------------------------//

void CNC_Matrix::allocate(long m, long n, Storage storage, bool symmetric_storage){
  m_ = m ;
  n_ = n ;
  diag_size_ = (m<n)?(m):(n) ;
  symmetric_storage_ = symmetric_storage ;
  symmetric_tag_ = false ;
  storage_ = storage ;
  switch(storage) {
  case NONE:
    break ;
  case ROWS:
    rows_are_stored_    = true ;
    columns_are_stored_ = false ;
    break ;
  case COLUMNS:
    rows_are_stored_    = false ;
    columns_are_stored_ = true ;
    break ;
  case ROWS_AND_COLUMNS:
    rows_are_stored_    = true ;
    columns_are_stored_ = true ;
    break ;
  }
  diag_ = new double[diag_size_] ;
  for(long i=0; i<diag_size_; i++) {
    diag_[i] = 0.0 ;
  }

  if(rows_are_stored_) {
    row_ = new CNCSparseRowColumn[m] ;
  } else {
    row_ = NULL ;
  }

  if(columns_are_stored_) {
    column_ = new CNCSparseRowColumn[n] ;
  } else {
    column_ = NULL ;
  }
}



ARCANE_END_NAMESPACE


//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
