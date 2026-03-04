// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef LAPACKUTILS_H
#define LAPACKUTILS_H

#define F77NAME(X) X ## _

////////////////////////////////////////////////////////////
// Schur decompositions and eigenvalue/vector computation

// sgees and dgees are external Fortran 77 routines from the LAPACK 
// library to be accessed following the "C" convention

//! Pointer to a boolean function of two floats
typedef bool (*bfnff_ptr)(float&, float&);
//! Pointer to a boolean function of two doubles
typedef bool (*bfndd_ptr)(double&, double&);

extern "C" {
  int F77NAME(sgees)(char&,      // JOBVS
                     char&,      // SORT
                     bfnff_ptr&, // SELECT
                     int&,       // N
                     float*,     // A
                     int&,       // LDA
                     int&,       // SDIM
                     float*,     // WR
                     float*,     // WI
                     float*,     // VS
                     int&,       // LDVS
                     float*,     // WORK
                     int&,       // LWORK
                     bool*,      // BWORK
                     int&);      // INFO
  
  int F77NAME(dgees)(char&,      // JOBVS
                     char&,      // SORT
                     bfndd_ptr&, // SELECT
                     int&,       // N
                     double*,    // A
                     int&,       // LDA
                     int&,       // SDIM
                     double*,    // WR
                     double*,    // WI
                     double*,    // VS
                     int&,       // LDVS
                     double*,    // WORK
                     int&,       // LWORK
                     bool*,      // BWORK
                     int&);      // INFO
}

////////////////////////////////////////////////////////////
// implicit QL or QR method for tridiagonal matrices 
// eigenvalue computation

extern "C" {
  int F77NAME(ssteqr)(char&,     // COMPZ
                      int&,      // N
                      float* ,   // D
                      float*,    // E
                      float*,    // Z
                      int&,      // LDZ
                      float*,    // WORK
                      int&);     // INFO

  int F77NAME(dsteqr)(char&,     // COMPZ
                      int&,      // N
                      double*,   // D
                      double*,   // E
                      double*,   // Z
                      int&,      // LDZ
                      double*,   // WORK
                      int&);     // INFO
}

////////////////////////////////////////////////////////////
// lu decomposition

extern "C" {
  int F77NAME(sgetrf)(int&,       // M
                      int&,       // N
                      float*,     // A
                      int&,       // LDA
                      int*,       // IPIV
                      int&);      // INFO

  int F77NAME(dgetrf)(int&,       // M
                      int&,       // N
                      double*,    // A
                      int&,       // LDA
                      int*,       // IPIV
                      int&);      // INFO

  int F77NAME(sgetrs)(char&,      // TRANS
                      int&,       // N
                      int&,       // NRHS
                      float*,     // A
                      int&,       // LDA
                      int*,       // IPIV
                      float*,     // B
                      int&,       // LDB
                      int&);      // INFO

  int F77NAME(dgetrs)(char&,      // TRANS
                      int&,       // N
                      int&,       // NRHS
                      double*,    // A
                      int&,       // LDA
                      int*,       // IPIV
                      double*,    // B
                      int&,       // LDB
                      int&);      // INFO
}

////////////////////////////////////////////////////////////
// Square root of a matrix

extern "C" {
  int F77NAME(ssyevr)(char&,      // JOBZ
                      char&,      // RANGE
                      char&,      // UPLO
                      int&,       // N
                      float*,     // A
                      int&,       // LDA
                      float&,     // VL
                      float&,     // VU
                      int&,       // IL
                      int&,       // IU
                      float&,     // ABSTOL
                      int&,       // M
                      float*,     // W
                      float*,     // Z
                      int&,       // LDZ
                      int*,       // ISUPPZ
                      float*,     // WORK
                      int&,       // LWORK
                      int*,       // IWORK
                      int&,       // LIWORK
                      int&);      // INFO

  int F77NAME(dsyevr)(char&,      // JOBZ
                      char&,      // RANGE
                      char&,      // UPLO
                      int&,       // N
                      double*,    // A
                      int&,       // LDA
                      double&,    // VL
                      double&,    // VU
                      int&,       // IL
                      int&,       // IU
                      double&,    // ABSTOL
                      int&,       // M
                      double*,    // W
                      double*,    // Z
                      int&,       // LDZ
                      int*,       // ISUPPZ
                      double*,    // WORK
                      int&,       // LWORK
                      int*,       // IWORK
                      int&,       // LIWORK
                      int&);      // INFO
}

////////////////////////////////////////////////////////////
// Eigenvalues / eigenvectors of symmetric matrix

extern "C" {
  int F77NAME(ssyev)(const char&,      // JOBZ
                     const char&,      // UPLO
                     const int&,       // N
                     float*,           // A
                     const int&,       // LDA
                     float*,           // W
                     float*,           // WORK
                     const int&,       // LWORK
                     int&);            // INFO

  int F77NAME(dsyev)(const char&,      // JOBZ
                     const char&,      // UPLO
                     const int&,       // N
                     double*,          // A
                     const int&,       // LDA
                     double*,          // W
                     double*,          // WORK
                     const int&,       // LWORK
                     int&);            // INFO
}
////////////////////////////////////////////////////////////
// Wrappers

/*!
  \struct XGEES
  \brief Wrap Lapack routines sgees and dgees
  \author Daniele A. Di Pietro
*/

template<class T>
struct XGEES {
  typedef bool (*bfnTT_ptr)(T&, T&);
  virtual void apply(char&, char&, bfnTT_ptr&, int&, T*, int&,
                     int&, T*, T*, T*, int&, T*, int&, bool*, int&) = 0;
};

template<>
struct XGEES<float> {
  typedef bool (*bfnTT_ptr)(float&, float&);
  static void apply(char& JOBVS, char& SORT, bfnTT_ptr& SELECT, 
                    int& N, float* A, int& LDA, int& SDIM, 
                    float* WR, float* WI, float* VS, int& LDVS, 
                    float* WORK, int& LWORK, bool* BWORK, int& INFO) {
    F77NAME(sgees)(JOBVS, SORT, SELECT, N, A, LDA, SDIM,
                   WR, WI, VS, LDVS, WORK, LWORK, BWORK, INFO);
  }
};

template<>
struct XGEES<double> {
  typedef bool (*bfnTT_ptr)(double&, double&);
  static void apply(char& JOBVS, char& SORT, bfnTT_ptr& SELECT, 
                    int& N, double* A, int& LDA, int& SDIM, 
                    double* WR, double* WI, double* VS, int& LDVS, 
                    double* WORK, int& LWORK, bool* BWORK, int& INFO) {
    F77NAME(dgees)(JOBVS, SORT, SELECT, N, A, LDA, SDIM,
                   WR, WI, VS, LDVS, WORK, LWORK, BWORK, INFO);
  }
};

/*!
  \struct XGETRF
  \brief Wrap Lapack routines sgetrf and dgetrf
  \author Daniele A. Di Pietro
*/

template<class T>
struct XGETRF{
  virtual void apply(int&, int&, T*, int&, int*, int&) = 0;
};
  
template<>
struct XGETRF<float> {
  static inline void apply(int& M, int& N, float* A, int& LDA, 
                           int* IPIV, int& INFO) {
    F77NAME(sgetrf)(M, N, A, LDA, IPIV, INFO);      
  }
};

template<>
struct XGETRF<double> {
  static inline void apply(int& M, int& N, double* A, int& LDA, 
                           int* IPIV, int& INFO) {
    F77NAME(dgetrf)(M, N, A, LDA, IPIV, INFO);
      
  }
};

/*!
  \struct XGETRS
  \brief Wrap Lapack routines sgetrs and dgetrs
  \author Daniele A. Di Pietro
*/

template<class T>
struct XGETRS {
  virtual void apply(char&, int&, int&,
                     T*, int&, int*, 
                     T*, int&, int&) = 0;
};
  
template<>
struct XGETRS<float> {
  static inline void apply(char& TRANS, int& N, int& NRHS, 
                           float* A, int& LDA, int* IPIV, 
                           float* B, int& LDB, int& INFO) {
    F77NAME(sgetrs)(TRANS, N, NRHS, A, LDA, IPIV, B, LDB, INFO);	      
  }
};

template<>
struct XGETRS<double> {
  static inline void apply(char& TRANS, int& N, int& NRHS, 
                           double* A, int& LDA, int* IPIV, 
                           double* B, int& LDB, int& INFO) {
    F77NAME(dgetrs)(TRANS, N, NRHS, A, LDA, IPIV, B, LDB, INFO);	      
  }
};

/*!
  \struct XSTEQR
  \brief Wrap Lapack routines ssteqr and dsteqr
  \author Daniele A. Di Pietro
*/
  
template<class T>
struct XSTEQR {
  virtual void apply(char&, int&, T*, T*, T*, int&, T*, int&);
};

template<>
struct XSTEQR<float> {
  static inline void apply(char& COMPZ, int& N, 
                           float* D, float* E, float* Z,
                           int& LDZ, float* WORK, int& INFO) {
    F77NAME(ssteqr)(COMPZ, N, D, E, Z, LDZ, WORK, INFO);
  }
};

template<>
struct XSTEQR<double> {
  static inline void apply(char& COMPZ, int& N, 
                           double* D, double* E, double* Z,
                           int& LDZ, double* WORK, int& INFO) {
    F77NAME(dsteqr)(COMPZ, N, D, E, Z, LDZ, WORK, INFO);
  }
};

/*!
  \struct XSYEVR
  \brief Wrap Lapack routines ssyevr and dsyevr
  \author Daniele A. Di Pietro
*/

template<class T>
struct XSYEVR {
    
};

template<>
struct XSYEVR<float> {
  static inline void apply(char&   JOBZ,
                           char&   RANGE,
                           char&   UPLO,
                           int&    N,
                           float*  A,
                           int&    LDA,
                           float&  VL,
                           float&  VU,
                           int&    IL,  
                           int&    IU,
                           float&  ABSTOL,
                           int&    M,
                           float*  W,
                           float*  Z,
                           int&    LDZ,
                           int*    ISUPPZ,
                           float*  WORK,
                           int&    LWORK,
                           int*    IWORK,
                           int&    LIWORK,
                           int&    INFO) {
    F77NAME(ssyevr)(JOBZ, RANGE, UPLO, N, A, LDA, VL, VU, IL, IU, ABSTOL,
                    M, W, Z, LDZ, ISUPPZ, WORK, LWORK, IWORK, LIWORK, INFO);

  }
};

template<>
struct XSYEVR<double> {
  static inline void apply(char&   JOBZ,
                           char&   RANGE,
                           char&   UPLO,
                           int&    N,
                           double* A,
                           int&    LDA,
                           double& VL,
                           double& VU,
                           int&    IL,  
                           int&    IU,
                           double& ABSTOL,
                           int&    M,
                           double* W,
                           double* Z,
                           int&    LDZ,
                           int*    ISUPPZ,
                           double* WORK,
                           int&    LWORK,
                           int*    IWORK,
                           int&    LIWORK,
                           int&    INFO) {
    F77NAME(dsyevr)(JOBZ, RANGE, UPLO, N, A, LDA, VL, VU, IL, IU, ABSTOL,
                    M, W, Z, LDZ, ISUPPZ, WORK, LWORK, IWORK, LIWORK, INFO);

  }
};

/*!
  \struct XSYEV
  \brief Wrap Lapack routines ssyev and dsyev
  \author Pascal Hav�
*/

template<class T>
struct XSYEV {
    
};

template<>
struct XSYEV<float> {
  static inline void apply(const char&   JOBZ,
                           const char&   UPLO,
                           const int&    N,
                           float*  A,
                           const int&    LDA,
                           float*  W,
                           float*  WORK,
                           const int&    LWORK,
                           int&    INFO) {
    F77NAME(ssyev)(JOBZ, UPLO, N, A, LDA,
                   W, WORK, LWORK, INFO);

  }
};

template<>
struct XSYEV<double> {
  static inline void apply(const char&   JOBZ,
                           const char&   UPLO,
                           const int&    N,
                           double* A,
                           const int&    LDA,
                           double* W,
                           double* WORK,
                           const int&    LWORK,
                           int&    INFO) {
    F77NAME(dsyev)(JOBZ, UPLO, N, A, LDA,
                   W, WORK, LWORK, INFO);

  }
};


#endif
