// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* cnc_matrix.h                                                (C) 2000-2012 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef _CNC_INTERFACE_MATRIX_H_
#define _CNC_INTERFACE_MATRIX_H_


ARCANE_BEGIN_NAMESPACE
  
/**
 * A coefficient of a SparseMatrix. (currently FLOAT data ONLY)
 */
class CNCCoeff {
public:
  double a ;
  long index ;
} ;

//---------------------------------------------------------------------------//

class CNCCoeffIndexCompare {
public:
  bool operator()(const CNCCoeff& c1, const CNCCoeff& c2) {
    return c1.index < c2.index ;
  }
};

//---------------------------------------------------------------------------//

/**
 * A row or a column of a SparseMatrix. SparseRowColumn is
 * compressed, and stored in the form of a list of
 * (value,index) couples.
 */
class CNCSparseRowColumn {
public:
  CNCSparseRowColumn() {
    coeff_ = CNCallocate<CNCCoeff>(2) ;
    nb_coeffs_ = 0 ;
    capacity_ = 2 ;
  }
  ~CNCSparseRowColumn() { CNCdeallocate<CNCCoeff>(coeff_) ;  }
  long nb_coeffs() const { return nb_coeffs_ ; }
  CNCCoeff& coeff(long ii) {return coeff_[ii] ; }
  const CNCCoeff& coeff(long ii) const {   return coeff_[ii] ;  }
  
  /** a_{index} <- a_{index} + val */
  void add(long index, double val){
    CNCCoeff* coeff = NULL ;
    // Search for a_{index}
    for(long ii=0; ii < nb_coeffs_; ii++) {
      if(coeff_[ii].index == index) {
        coeff = &(coeff_[ii]) ;
        break ;
      }
    }
    if(coeff != NULL) {
///////////////////////
//#warning add is set//
///////////////////////
//      coeff->a += val ;
      coeff->a = val ;
    } else {
      nb_coeffs_++ ;
      if(nb_coeffs_ > capacity_) {
        grow() ;
      }
      coeff = &(coeff_[nb_coeffs_ - 1]) ;
      coeff->a = val ;
      coeff->index = index ;
    }
  }
            
  /** sorts the coefficients by increasing index */
  void sort() {
    CNCCoeff* begin = coeff_ ;
    CNCCoeff* end   = coeff_ + nb_coeffs_ ;
    std::sort(begin, end, CNCCoeffIndexCompare()) ;
  }

  /** 
   * removes all the coefficients and frees the allocated
   * space.
   */
  void clear() { 
    CNCdeallocate<CNCCoeff>(coeff_) ;
    coeff_ = CNCallocate<CNCCoeff>(2) ;
    nb_coeffs_ = 0 ;
    capacity_ = 2 ;
  }

  /** 
   * removes all the coefficients, but keeps the
   * allocated space, that will be used by subsequent
   * calls to add().
   */
  void zero() { nb_coeffs_ = 0 ; }

protected:
  void grow(){
    long old_capacity = capacity_ ;
    capacity_ = capacity_ * 2 ;
    CNCreallocate<CNCCoeff>(coeff_, old_capacity, capacity_) ;
  }
  
private:
  CNCCoeff* coeff_ ;
  long nb_coeffs_ ;
  long capacity_ ;
} ;


//---------------------------------------------------------------------------//


class CNC_Matrix {
public:

  enum Storage {NONE, ROWS, COLUMNS, ROWS_AND_COLUMNS} ;

  // constructors / destructor

  /**
   * Constructs a m*n sparse matrix.
   * @param Storage can be one of ROWS, COLUMNS, ROWS_AND_COLUMNS
   */
  CNC_Matrix(long m, long n, Storage storage = ROWS) ;

  /**
   * Constructs a n*n sparse matrix, row storage is used,
   * Non symmetric storage is used
   */
  CNC_Matrix(long n ) ;

  /**
   * Constructs a square n*n sparse matrix.
   * @param Storage can be one of ROWS, COLUMNS, ROWS_AND_COLUMNS
   * @param symmetric_storage if set, only entries a_ij such
   *   that j <= i are stored.
   */
  CNC_Matrix(long n, Storage storage, bool symmetric_storage) ;

  CNC_Matrix() ;

  ~CNC_Matrix() ;

  // access

  long m() const;

  long n() const ;

  long diag_size() const ;

  /** number of non-zero coefficients */
  long nnz() const ;

  bool rows_are_stored() const;

  bool columns_are_stored() const ;

  Storage storage() const ;

  bool has_symmetric_storage() const ;

  bool is_square() const;

  bool is_symmetric() const ;

  /**
   * For symmetric matrices that are not stored in symmetric mode,
   * one may want to give a hint that the matrix is symmetric.
   */
  void set_symmetric_tag(bool x);
  
  CNCSparseRowColumn& row(long i) ;
  const CNCSparseRowColumn& row(long i) const ;
  CNCSparseRowColumn& column(long j) ;
  const CNCSparseRowColumn& column(long j) const ;


  /**
   * returns aii.
   */
  double diag(long i) const;

  /**
   * aij <- aij + val
   */
  void add(long i, long j, double val) ;
            

  /** sorts rows and columns by increasing coefficients */
  void sort() ;
            

  /**
   * removes all the coefficients and frees the allocated
   * space.
   */
  void clear() ;

  /**
   * removes all the coefficients, but keeps the allocated
   * storage, that will be used by subsequent calls to add().
   */
  void zero() ;

  void allocate(long m, long n, Storage storage, bool symmetric = false) ;

  void deallocate() ;

private:
  long m_ ;
  long n_ ;
  long diag_size_ ;

  CNCSparseRowColumn* row_ ;
  CNCSparseRowColumn* column_ ;
  double* diag_ ;

  Storage storage_ ;
  bool rows_are_stored_ ;
  bool columns_are_stored_ ;
  bool symmetric_storage_ ;
  bool symmetric_tag_ ;

  // SparseMatrix cannot be copied.
  CNC_Matrix(const CNC_Matrix& rhs) ;
  CNC_Matrix& operator=(const CNC_Matrix& rhs) ;
} ;


ARCANE_END_NAMESPACE

#endif
