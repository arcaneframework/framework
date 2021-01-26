#include "mpi.h"
#include "MCGCompositeMatrix.h"

#include <ALIEN/Kernels/MCG/DataStructure/MCGInternal.h>

#include <ALIEN/Core/Impl/MultiMatrixImpl.h>

/*---------------------------------------------------------------------------*/

BEGIN_MCGINTERNAL_NAMESPACE

END_MCGINTERNAL_NAMESPACE

BEGIN_NAMESPACE(Alien)

/*---------------------------------------------------------------------------*/

MCGCompositeMatrix::MCGCompositeMatrix(const MultiMatrixImpl* multi_impl)
: IMatrixImpl(multi_impl, AlgebraTraits<BackEnd::tag::mcgsolver_composite>::name())
{
  m_internal = new MatrixInternal();
}

/*---------------------------------------------------------------------------*/

MCGCompositeMatrix::~MCGCompositeMatrix()
{
  delete m_internal;
}

/*---------------------------------------------------------------------------*/

bool
MCGCompositeMatrix::initDiagMatrix(const int i, const int block_size, const int nrow,
    const int* row_offset, const int* cols)
{
  // Diag matrix in composite must be square
  assert(i < 2);

  return _initMatrix(i, i, block_size, block_size, nrow, nrow, row_offset, cols);
}

bool
MCGCompositeMatrix::initOffDiagMatrix(const int i, const int j, const int block_size,
    const int block_size2, const int nrow, const int ncol, const int* row_offset,
    const int* cols)
{
  assert(i != j);
  assert(i < 2);
  assert(j < 2);

  return _initMatrix(i, j, block_size, block_size2, nrow, ncol, row_offset, cols);
}

bool
MCGCompositeMatrix::initOffDiagSymProfileMatrix(const int i, const int j,
    const int block_size, const int block_size2, const int nrow, const int ncol,
    const int* row_offset, const int* cols, const bool trans)
{
  // init both matrices j,i and i,j
  // the profile of matrix j,i is the transposed profile of matrix i,j
  assert(i != j);
  assert(i < 2);
  assert(j < 2);

  bool r = false;

  r = _initMatrix(i, j, block_size, block_size2, nrow, ncol, row_offset, cols);
  if (trans) {
    r &= _initTransMatrix(j, i, block_size, block_size2, nrow, ncol, row_offset, cols);
  } else {
    r &= _initMatrix(j, i, block_size, block_size2, nrow, ncol, row_offset, cols);
  }

  return r;
}

bool
MCGCompositeMatrix::_initMatrix(const int i, const int j, const int block_size,
    const int block_size2, const int nrow, const int ncol, const int* row_offset,
    const int* cols)
{
  int nnz = row_offset[nrow];
  std::shared_ptr<MCGInternal::MatrixInternal::ProfileType> profile(
      new MCGInternal::MatrixInternal::ProfileType(nrow, ncol, nnz));

  auto& dst_kcol = profile->getKColv();
  auto& dst_cols = profile->getColsv();

  for (int i = 0; i < nrow + 1; ++i) {
    dst_kcol[i] = row_offset[i];
  }
  for (int i = 0; i < nnz; ++i) {
    dst_cols[i] = cols[i];
  }

  m_internal->m_matrix[i][j] =
      new MCGInternal::MatrixInternal::MatrixType(block_size, block_size2, profile);

  return true;
}

bool
MCGCompositeMatrix::_initTransMatrix(const int i, const int j, const int block_size,
    const int block_size2, const int nrow, const int ncol, const int* row_offset,
    const int* cols)
{
  int nnz = row_offset[nrow];
  std::shared_ptr<MCGInternal::MatrixInternal::ProfileType> profile(
      new MCGInternal::MatrixInternal::ProfileType(nrow, ncol, nnz));

  auto dst_kcol = profile->getKCol();
  auto dst_cols = profile->getCols();

  for (int i = 0; i < nrow + 1; ++i) {
    dst_kcol[i] = row_offset[i];
  }
  for (int i = 0; i < nnz; ++i) {
    dst_cols[i] = cols[i];
  }

  // transpose profile
  const auto& trans_profile_info = profile->transposeProfileInfo();

  std::shared_ptr<MCGInternal::MatrixInternal::ProfileType> trans_profile(
      new MCGInternal::MatrixInternal::ProfileType(trans_profile_info));

  std::vector<int> elem_perm;

  trans_profile->transposeInit(*profile, elem_perm);
  m_internal->m_matrix[i][j] =
      new MCGInternal::MatrixInternal::MatrixType(block_size2, block_size, trans_profile);

  return true;
}

bool
MCGCompositeMatrix::initMatrixValues(const int i, const int j, Real const* values)
{
  assert(i < 2);
  assert(j < 2);

  m_internal->m_matrix[i][j]->setValues(values);
  return true;
}

/*---------------------------------------------------------------------------*/

END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
