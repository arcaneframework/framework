// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <alien/kernels/petsc/PETScBackEnd.h>
#include <alien/kernels/petsc/data_structure/PETScInternal.h>
#include <alien/kernels/petsc/data_structure/PETScVector.h>
#include <alien/kernels/petsc/data_structure/PETScMatrix.h>

#include <alien/kernels/petsc/linear_solver/PETScInternalLinearSolver.h>

#include <alien/core/impl/MultiMatrixImpl.h>

#include <arccore/message_passing/Communicator.h>
#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>

/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/

PETScMatrix::PETScMatrix(const MultiMatrixImpl* multi_impl)
: IMatrixImpl(multi_impl, AlgebraTraits<BackEnd::tag::petsc>::name())
{
  const auto& row_space = multi_impl->rowSpace();
  const auto& col_space = multi_impl->colSpace();
  if (row_space.size() != col_space.size())
    throw Arccore::FatalErrorException("PETSc matrix must be square");

  m_pm = multi_impl->distribution().parallelMng();
}

/*---------------------------------------------------------------------------*/

PETScMatrix::~PETScMatrix()
{
  //delete m_internal;
}

PETScMatrix::MatrixInternal* PETScMatrix::internal() {
  return m_internal.get();
}

const PETScMatrix::MatrixInternal*
PETScMatrix::internal() const {
  return m_internal.get();
}


BackEnd::Memory::eType PETScMatrix::getMemoryType() const {
  return  PETScInternalLinearSolver::m_library_plugin->getMemoryType() ;
}

BackEnd::Exec::eSpaceType PETScMatrix::getExecSpace() const {
  return  PETScInternalLinearSolver::m_library_plugin->getExecSpace() ;
}

/*---------------------------------------------------------------------------*/

bool
PETScMatrix::initMatrix(const int local_size,
                        const int local_offset,
                        const int global_size,
                        const int block_size,
                        Arccore::ConstArrayView<Arccore::Integer> diag_lineSizes,
                        Arccore::ConstArrayView<Arccore::Integer> offdiag_lineSizes,
                        const bool parallel)
{
  int ierr = 0; // code d'erreur de retour

  auto memory_type = PETScInternalLinearSolver::m_library_plugin->getMemoryType() ;
  auto exec_space = PETScInternalLinearSolver::m_library_plugin->getExecSpace() ;

  m_internal.reset(new MatrixInternal{local_offset,
                                      local_size,
                                      block_size,
                                      parallel,
                                      memory_type,
                                      exec_space});

  Arccore::Integer max_diag_size = 0, max_offdiag_size = 0;

// -- Matrix --
#ifdef PETSC_HAVE_MATVALID
  PetscTruth valid_flag;
  ierr += MatValid(m_internal->m_internal, &valid_flag);
  if (valid_flag == PETSC_TRUE)
    return (ierr == 0);
#endif /* PETSC_HAVE_MATVALID */


  auto pm = m_pm->communicator();
  MPI_Comm mpi_comm = (pm.isValid()) ? static_cast<MPI_Comm>(pm) : (parallel ? PETSC_COMM_WORLD : PETSC_COMM_SELF) ;

  switch(exec_space)
  {
    case BackEnd::Exec::Device:
    {
#if PETSC_VERSION_GE(3, 20, 0)
      ierr += MatCreate(mpi_comm, &m_internal->m_internal);
      ierr += MatSetSizes(m_internal->m_internal, local_size, local_size, global_size, global_size);
#ifdef PETSC_HAVE_MATSETBLOCKSIZE
      if(block_size>1)
        ierr += MatSetBlockSize(m_internal->m_internal, block_size) ;
#endif
      ierr += MatSetType(m_internal->m_internal, m_internal->m_type);
      if(parallel)
      {
         ierr += MatMPIAIJSetPreallocation(m_internal->m_internal, max_diag_size,
             diag_lineSizes.unguardedBasePointer(), max_offdiag_size,
             offdiag_lineSizes.unguardedBasePointer());
       } else { // Use sequential structures
         ierr += MatSeqAIJSetPreallocation(
             m_internal->m_internal, max_diag_size, diag_lineSizes.unguardedBasePointer());
       }
#else
       throw Arccore::FatalErrorException(A_FUNCINFO, "PETSC Matrix Type for Device Execution is not available");
       return false ;
#endif
    }
    break;
    case BackEnd::Exec::Host:
    default:
    {
      ierr += MatCreate(mpi_comm, &m_internal->m_internal);
      ierr += MatSetSizes(
          m_internal->m_internal, local_size, local_size, global_size, global_size);
#ifdef PETSC_HAVE_MATSETBLOCKSIZE
      if(block_size>1)
        ierr += MatSetBlockSize(m_internal->m_internal, block_size) ;
#endif
      ierr += MatSetType(m_internal->m_internal, m_internal->m_type);

       if(parallel)
       {
          ierr += MatMPIAIJSetPreallocation(m_internal->m_internal, max_diag_size,
              diag_lineSizes.unguardedBasePointer(), max_offdiag_size,
              offdiag_lineSizes.unguardedBasePointer());
        } else { // Use sequential structures
          ierr += MatSeqAIJSetPreallocation(
              m_internal->m_internal, max_diag_size, diag_lineSizes.unguardedBasePointer());
        }
    }
  }

  // Offsets are implicit with PETSc, we can to check that
  // they are compatible with the expected behaviour
  PetscInt low;
  ierr += MatGetOwnershipRange(m_internal->m_internal, &low, nullptr);
  if (low != local_offset)
    return false;

  return (ierr == 0);
}

bool PETScMatrix::initMatrix(const int local_size,
                        const int local_offset,
                        const int global_size,
                        const int block_size,
                        const int nb_dofs,
                        const int* dof_uids,
                        const int nnz,
                        int* rows,
                        int* cols,
                        const bool parallel)
{
  int ierr = 0; // code d'erreur de retour

  auto memory_type = PETScInternalLinearSolver::m_library_plugin->getMemoryType() ;
  auto exec_space = PETScInternalLinearSolver::m_library_plugin->getExecSpace() ;

  m_internal.reset(new MatrixInternal{local_offset,
                                      local_size,
                                      block_size,
                                      parallel,
                                      memory_type,
                                      exec_space});
  auto pm = m_pm->communicator();
  MPI_Comm mpi_comm = (pm.isValid()) ? static_cast<MPI_Comm>(pm) : (parallel ? PETSC_COMM_WORLD : PETSC_COMM_SELF) ;

  ISLocalToGlobalMapping m_petsc_map;
  if(parallel)
    ierr += ISLocalToGlobalMappingCreate(mpi_comm, 1, nb_dofs, dof_uids, PETSC_COPY_VALUES, &m_petsc_map);

  ierr += MatCreate(mpi_comm, &m_internal->m_internal);
  ierr += MatSetSizes(m_internal->m_internal, local_size, local_size, global_size, global_size);

#ifdef PETSC_HAVE_MATSETBLOCKSIZE
  if(block_size>1)
    ierr += MatSetBlockSize(m_internal->m_internal, block_size) ;
#endif
  ierr += MatSetType(m_internal->m_internal, m_internal->m_type);

  if(parallel)
  {
    ierr += MatSetLocalToGlobalMapping(m_internal->m_internal, m_petsc_map, m_petsc_map);
    ierr += ISLocalToGlobalMappingDestroy(&m_petsc_map);
  }

  ierr += MatSetPreallocationCOO(m_internal->m_internal, nnz, rows, cols);

  return (ierr == 0);
}

/*---------------------------------------------------------------------------*/

bool
PETScMatrix::addMatrixValues(
    const int row, const int ncols, const int* cols, const Arccore::Real* values)
{
  assert(m_internal.get()) ;
  int ierr =
      MatSetValues(m_internal->m_internal, 1, &row, ncols, cols, values, ADD_VALUES);
  return (ierr == 0);
}

/*---------------------------------------------------------------------------*/

bool
PETScMatrix::setMatrixValues(
    const int row, const int ncols, const int* cols, const Arccore::Real* values)
{
  assert(m_internal.get()) ;
  int ierr =
      MatSetValues(m_internal->m_internal, 1, &row, ncols, cols, values, INSERT_VALUES);
  return (ierr == 0);
}

bool
PETScMatrix::setMatrixValuesFromCSR(const Arccore::Real* values)
{
  assert(m_internal.get()) ;

  int ierr = MatSetValuesCOO(m_internal->m_internal, values, INSERT_VALUES);

  return (ierr == 0);
}


#ifdef PETSC_HAVE_MATSETBLOCKSIZE
bool
PETScMatrix::addMatrixBlockValues(
    const int row, const int ncols, const int* cols, const Arccore::Real* values)
{
  assert(m_internal.get()) ;
  int ierr =
      MatSetValuesBlocked(m_internal->m_internal, 1, &row, ncols, cols, values, ADD_VALUES);
  return (ierr == 0);
}

bool
PETScMatrix::setMatrixBlockValues(
    const int row, const int ncols, const int* cols, const Arccore::Real* values)
{
  assert(m_internal.get()) ;
  int ierr =
      MatSetValuesBlocked(m_internal->m_internal, 1, &row, ncols, cols, values, INSERT_VALUES);
  return (ierr == 0);
}
#endif
/*---------------------------------------------------------------------------*/
void PETScMatrix::setMatrixCoordinate(Vector const& x, Vector const& y, Vector const& z)
{
  assert(m_internal.get()) ;
  m_internal->m_coordinates_dim = 3;
  PetscScalar *c;
  VecCreate(MPI_COMM_WORLD, &m_internal->m_coordinates);
#ifdef PETSC_HAVE_VECSETBLOCKSIZE
  VecSetBlockSize(m_internal->m_coordinates, m_internal->m_coordinates_dim);
#endif
  VecSetSizes(m_internal->m_coordinates, m_internal->m_local_size, PETSC_DECIDE);
  VecSetUp(m_internal->m_coordinates);
  VecGetArray(m_internal->m_coordinates, &c);

  Alien::LocalVectorReader x_view(x);
  Alien::LocalVectorReader y_view(y);
  Alien::LocalVectorReader z_view(z);

  {
    Integer offset = 0 ;
    for(int i=0;i<m_internal->m_local_size/m_internal->m_block_size;++i)
    {
      c[offset    ] = x_view[i] ;
      c[offset + 1] = y_view[i] ;
      c[offset + 2] = z_view[i] ;
      offset += m_internal->m_coordinates_dim ;
    }
  }
  VecRestoreArray(m_internal->m_coordinates, &c);
  m_internal->m_has_coordinates = true ;
}

void PETScMatrix::setMatrixCoordinate(Vector const& x, Vector const& y)
{
  assert(m_internal.get()) ;
  m_internal->m_coordinates_dim = 2 ;
  PetscScalar *c;
  VecCreate(MPI_COMM_WORLD, &m_internal->m_coordinates);
#ifdef PETSC_HAVE_VECSETBLOCKSIZE
  VecSetBlockSize(m_internal->m_coordinates, m_internal->m_coordinates_dim);
#endif
  VecSetSizes(m_internal->m_coordinates, m_internal->m_local_size, PETSC_DECIDE);
  VecSetUp(m_internal->m_coordinates);
  VecGetArray(m_internal->m_coordinates, &c);

  Alien::LocalVectorReader x_view(x);
  Alien::LocalVectorReader y_view(y);

  {
    Integer offset = 0 ;
    for(int i=0;i<m_internal->m_local_size/m_internal->m_block_size;++i)
    {
      c[offset    ] = x_view[i] ;
      c[offset + 1] = y_view[i] ;
      offset += m_internal->m_coordinates_dim ;
    }
  }
  VecRestoreArray(m_internal->m_coordinates, &c);
  m_internal->m_has_coordinates = true ;
}
/*---------------------------------------------------------------------------*/

bool
PETScMatrix::assemble()
{
  int ierr = 0;
  ierr += MatAssemblyBegin(m_internal->m_internal, MAT_FINAL_ASSEMBLY);
  ierr += MatAssemblyEnd(m_internal->m_internal, MAT_FINAL_ASSEMBLY);
  return (ierr == 0);
}

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//#pragma clang diagnostic pop
