#include <alien/core/impl/MultiMatrixImpl.h>

#include "alien/kernels/mcg/data_structure/MCGInternal.h"
#include "alien/kernels/mcg/data_structure/MCGMatrix.h"

BEGIN_MCGINTERNAL_NAMESPACE

END_MCGINTERNAL_NAMESPACE

namespace Alien {

MCGMatrix::MCGMatrix(const MultiMatrixImpl* multi_impl)
: IMatrixImpl(multi_impl, AlgebraTraits<BackEnd::tag::mcgsolver>::name())
{
  m_internal = new MatrixInternal();
}

MCGMatrix::~MCGMatrix()
{
  delete m_internal;
}

bool
MCGMatrix::initMatrix(const int block_size, const int block_size2, const int nrow,
    int const* row_offset, int const* cols)
{
  Integer nblocks = row_offset[nrow];
  std::shared_ptr<MCGInternal::MatrixInternal::ProfileType> profile(
      new MCGInternal::MatrixInternal::ProfileType(nrow, nrow, nblocks));

  auto dst_kcol = profile->getKCol();
  auto dst_cols = profile->getCols();

  for (int i = 0; i < nrow + 1; ++i) {
    dst_kcol[i] = row_offset[i];
  }
  for (int i = 0; i < nblocks; ++i) {
    dst_cols[i] = cols[i];
  }

  m_internal->m_matrix[0][0].reset(
      new MCGInternal::MatrixInternal::MatrixType(block_size, block_size2, profile));

  m_is_init = true;
  return true;
}

bool
MCGMatrix::initMatrixValues(Real const* values)
{
  m_internal->m_matrix[0][0]->setValues(values);
  return true;
}

bool
MCGMatrix::isInit() const
{
  return m_is_init;
}
} // namespace Alien
