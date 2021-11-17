// -*- C++ -*-
#ifndef ALIEN_MCGIMPL_MCGCOMPOSITEMATRIX_H
#define ALIEN_MCGIMPL_MCGCOMPOSITEMATRIX_H

#include <alien/core/impl/IMatrixImpl.h>
#include <alien/core/block/Block.h>
#include <alien/data/Space.h>

#include "alien/kernels/mcg/MCGPrecomp.h"
#include "alien/kernels/mcg/MCGBackEnd.h"

BEGIN_MCGINTERNAL_NAMESPACE

class MatrixInternal;

END_MCGINTERNAL_NAMESPACE

namespace Alien {

class MCGCompositeMatrix : public IMatrixImpl
{
 public:
  typedef MCGInternal::MatrixInternal MatrixInternal;

 public:
  MCGCompositeMatrix(const MultiMatrixImpl* multi_impl);
  virtual ~MCGCompositeMatrix();

 public:
  void init(
      const ISpace& row_space, const ISpace& col_space, const MatrixDistribution& dist)
  {
    std::cout << "init MCGCompositeMatrix with m_domain_offset = "
                 "dist.rowOffset()/m_equations_num "
              << std::endl;
  }

  void initSpace0(const Space& space) { m_space0 = &space; }

  void initSpace1(const Space& space) { m_space1 = &space; }

  const ISpace& space() const
  {
    if (m_space0)
      return *m_space0;
    else
      return IMatrixImpl::rowSpace();
  }

  const Space& space0() const { return *m_space0; }

  const Space& space1() const { return *m_space1; }

  void clear() {}

 public:
  bool initDiagMatrix(const int i, const int block_size, const int nrow,
      const int* row_offset, const int* cols);

  bool initOffDiagMatrix(const int i, const int j, const int block_size,
      const int block_size2, const int nrow, const int ncol, const int* row_offset,
      const int* cols);

  bool initOffDiagSymProfileMatrix(const int i, const int j, const int block_size,
      const int block_size2, const int nrow, const int ncol, const int* row_offset,
      const int* cols, const bool trans = false);

  bool initMatrixValues(const int i, const int j, Real const* values);

  Space m_row_space1;
  Space m_col_space1;

 public:
  MatrixInternal* internal() { return m_internal; }
  const MatrixInternal* internal() const { return m_internal; }

 private:
  MatrixInternal* m_internal = nullptr;
  Space const* m_space0 = nullptr;
  Space const* m_space1 = nullptr;

  bool _initMatrix(const int i, const int j, const int block_size, const int block_size2,
      const int nrow, const int ncol, const int* row_offset, const int* cols);
  bool _initTransMatrix(const int i, const int j, const int block_size,
      const int block_size2, const int nrow, const int ncol, const int* row_offset,
      const int* cols);
};
} // namespace Alien

#endif /* ALIEN_MCGIMPL_MCGCOMPOSITEMATRIX_H */
