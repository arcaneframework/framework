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

#include <alien/utils/Precomp.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien::Internal {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MatrixInternal
{
 public:
  MatrixInternal(const MPI_Comm comm)
  : m_internal(nullptr)
  , m_comm(comm)
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

 private:
  matrix_type m_internal;
  MPI_Comm m_comm;
};

/*---------------------------------------------------------------------------*/

class VectorInternal
{
 public:
  VectorInternal(const MPI_Comm comm)
  : m_internal(nullptr)
  , m_comm(comm)
  {
  }

  virtual ~VectorInternal();

  typedef HYPRE_IJVector vector_type;
  vector_type& internal() { return m_internal; }

  vector_type const& internal() const { return m_internal; }

  bool init(const int ilower, const int iupper);

  bool addValues(const int nrow, const int* rows, const Arccore::Real* values);

  bool setValues(const int nrow, const int* rows, const Arccore::Real* values);

  bool setInitValues(const int nrow, const int* rows, const Arccore::Real* values);

  bool assemble();

  bool getValues(const int nrow, const int* rows, Arccore::Real* values);

 private:
  vector_type m_internal;
  MPI_Comm m_comm;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien::Internal

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
