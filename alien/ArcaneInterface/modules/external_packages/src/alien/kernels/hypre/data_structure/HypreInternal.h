// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

#define MPICH_SKIP_MPICXX 1
#include "mpi.h"
#include <_hypre_utilities.h>
#include <HYPRE_utilities.h>
#include <HYPRE.h>
#include <HYPRE_parcsr_mv.h>

#include <HYPRE_IJ_mv.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_parcsr_mv.h>

//! Internal struct for Hypre implementation
/*! Separate data from header;
 *  can be only included by HypreLinearSystem and HypreLinearSolver
 */
#include <vector>

#include <alien/utils/Precomp.h>

#include <alien/core/backend/BackEnd.h>
#include <alien/AlienExternalPackagesPrecomp.h>
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien::Internal {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MatrixInternal
{
 public:
  typedef Arccore::Real ValueType ;
  typedef int           IndexType ;
  MatrixInternal(const MPI_Comm comm,
                 Alien::BackEnd::Memory::eType memory_type,
                 Alien::BackEnd::Exec::eSpaceType exec_space)
  : m_internal(nullptr)
  , m_comm(comm)
  , m_memory_type(memory_type)
  , m_exec_space(exec_space)
  {
  }

  virtual ~MatrixInternal();

  typedef HYPRE_IJMatrix matrix_type;
  matrix_type& internal() { return m_internal; }

  matrix_type const& internal() const { return m_internal; }

  bool init(const int ilower, const int iupper, const int jlower, const int jupper,
      const Arccore::ConstArrayView<Arccore::Integer>& lineSizes);

  bool addMatrixValues(const int nrow, const int* rows, const int* ncols, const int* cols,
      const Arccore::Real* values);

  bool setMatrixValues(const int nrow, const int* rows, const int* ncols, const int* cols,
      const Arccore::Real* values);

  bool assemble();

  BackEnd::Memory::eType getMemoryType() const {
    return m_memory_type ;
  }

  BackEnd::Exec::eSpaceType getExecSpace() const {
    return m_exec_space  ;
  }

  void allocateHostPointers(std::size_t nrows,
                            std::size_t nnz,
                            IndexType** rows,
                            IndexType** ncols,
                            IndexType** cols,
                            ValueType** values) ;

  void freeHostPointers(IndexType* rows,
                        IndexType* ncols,
                        IndexType* cols,
                        ValueType* values) ;

  void allocateDevicePointers(std::size_t nrows,
                              std::size_t nnz,
                              IndexType** rows,
                              IndexType** ncols,
                              IndexType** cols,
                              ValueType** values) ;

  void copyHostToDevicePointers(std::size_t nrows,
                                std::size_t nnz,
                                const IndexType* rows_h,
                                const IndexType* ncols_h,
                                const IndexType* cols_h,
                                const ValueType* values_h,
                                IndexType* rows_d,
                                IndexType* ncols_d,
                                IndexType* cols_d,
                                ValueType* values_d) ;

  void freeDevicePointers(IndexType* rows,
                          IndexType* ncols,
                          IndexType* cols,
                          ValueType* values) ;

 private:
  matrix_type m_internal;
  MPI_Comm m_comm;
  Alien::BackEnd::Memory::eType m_memory_type = BackEnd::Memory::Host ;
  Alien::BackEnd::Exec::eSpaceType m_exec_space = BackEnd::Exec::Host ;
};

/*---------------------------------------------------------------------------*/

class VectorInternal
{
 public:
  using ValueType = Arccore::Real;
  using IndexType = int;

  VectorInternal(const MPI_Comm comm,
                 BackEnd::Memory::eType memory_type,
                 BackEnd::Exec::eSpaceType exec_space)
  : m_internal(nullptr)
  , m_comm(comm)
  , m_memory_type(memory_type)
  , m_exec_space(exec_space)
  {
  }

  virtual ~VectorInternal();

  typedef HYPRE_IJVector vector_type;
  vector_type& internal() { return m_internal; }

  vector_type const& internal() const { return m_internal; }

  bool init(const int ilower, const int iupper);

  void setRows(std::size_t nrow, IndexType const* rows) ;

  bool addValues(const int nrow, const int* rows, const Arccore::Real* values);

  bool setValues(const int nrow, const int* rows, const Arccore::Real* values);

  bool setValuesOnDevice(const int nrow, const int* rows, const Arccore::Real* values);

  bool setInitValues(const int nrow, const int* rows, const Arccore::Real* values);

  bool setValuesToZeros(const int nrow, const int* rows) ;

  bool assemble();

  bool getValues(const int nrow, const int* rows, Arccore::Real* values);
  bool getValuesToDevice(const int nrow, const int* rows, Arccore::Real* values);
  bool getValuesToHost(const int nrow, const int* rows, Arccore::Real* values);

  BackEnd::Memory::eType getMemoryType() const {
    return m_memory_type ;
  }

  BackEnd::Exec::eSpaceType getExecSpace() const {
    return m_exec_space  ;
  }

  static void allocateDevicePointers(std::size_t local_size, ValueType** values);

  static void freeDevicePointers(ValueType* values) ;

 private:
  vector_type m_internal;
  MPI_Comm m_comm;
  IndexType* m_rows = nullptr ;
  std::vector<ValueType> m_zeros ;
  ValueType* m_zeros_device = nullptr ;
  BackEnd::Memory::eType m_memory_type = BackEnd::Memory::Host ;
  BackEnd::Exec::eSpaceType m_exec_space = BackEnd::Exec::Host ;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien::Internal

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
