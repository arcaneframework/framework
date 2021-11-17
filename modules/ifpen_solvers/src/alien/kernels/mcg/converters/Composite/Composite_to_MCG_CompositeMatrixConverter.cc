#include <iostream>

#include "alien/core/backend/IMatrixConverter.h"
#include "alien/core/backend/MatrixConverterRegisterer.h"

#include <alien/kernels/simple_csr/CSRStructInfo.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/composite/CompositeMatrix.h>
#include <alien/kernels/composite/CompositeBackEnd.h>

#include "alien/kernels/mcg/data_structure/MCGVector.h"
#include "alien/kernels/mcg/data_structure/MCGCompositeMatrix.h"
#include "alien/kernels/mcg/MCGBackEnd.h"

using namespace Alien;
using namespace Alien::SimpleCSRInternal;

/*---------------------------------------------------------------------------*/

class Composite_to_MCG_MatrixConverter : public IMatrixConverter
{
 public:
  Composite_to_MCG_MatrixConverter();
  virtual ~Composite_to_MCG_MatrixConverter() {}

 public:
  BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::composite>::name();
  }
  BackEndId targetBackend() const
  {
    return AlgebraTraits<BackEnd::tag::mcgsolver_composite>::name();
  }
  void convert(const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const;
  void convert(
      const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl, Integer i, Integer j) const;

  void _build(
      const SimpleCSRMatrix<Real>& sourceImpl, MCGCompositeMatrix& targetImpl) const;
  void _buildSubMatrix01(
      const SimpleCSRMatrix<Real>& sourceImpl, MCGCompositeMatrix& targetImpl) const;
  void _buildSubMatrix10(
      const SimpleCSRMatrix<Real>& sourceImpl, MCGCompositeMatrix& targetImpl) const;
  void _buildSubMatrix11(
      const SimpleCSRMatrix<Real>& sourceImpl, MCGCompositeMatrix& targetImpl) const;
};

/*---------------------------------------------------------------------------*/

Composite_to_MCG_MatrixConverter::Composite_to_MCG_MatrixConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
Composite_to_MCG_MatrixConverter::convert(
    const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const
{
  const auto& v = cast<CompositeKernel::Matrix>(sourceImpl, sourceBackend());

  alien_debug([&] { cout() << "Converting CompositeMatrix: " << &v << " to MCGMatrix"; });

  for (Integer i = 0; i < v.size(); ++i) {
    for (Integer j = 0; j < v.size(); ++j) {
      this->convert(sourceImpl, targetImpl, i, j);
    }
  }
}

void
Composite_to_MCG_MatrixConverter::convert(
    const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl, int i, int j) const
{
  const auto& compo = cast<CompositeKernel::Matrix>(sourceImpl, sourceBackend());
  const SimpleCSRMatrix<Real>& v = compo(i, j).impl()->get<BackEnd::tag::simplecsr>();
  MCGCompositeMatrix& v2 = cast<MCGCompositeMatrix>(targetImpl, targetBackend());

  if (i == 0) {
    if (j == 0) {
      const ISpace& space = v.rowSpace();
      const MatrixDistribution& dist = v.distribution();
      v2.init(space, space, dist);
      // v2.initDistribution(dist);
      if (sourceImpl->vblock())
        throw FatalErrorException(
            A_FUNCINFO, "Block sizes are variable - builds not yet implemented");
      else
        _build(v, v2);
    } else {
      _buildSubMatrix01(v, v2);
    }
  } else {
    if (j == 0) {
      _buildSubMatrix10(v, v2);
    } else {
      const ISpace& space = v.rowSpace();
      const MatrixDistribution& dist = v.distribution();
      v2.init(space, space, dist);
      _buildSubMatrix11(v, v2);
    }
  }
}

void
Composite_to_MCG_MatrixConverter::_build(
    const SimpleCSRMatrix<Real>& sourceImpl, MCGCompositeMatrix& targetImpl) const
{
  const CSRStructInfo& profile = sourceImpl.getCSRProfile();
  const Integer localSize = profile.getNRow();
  const SimpleCSRMatrix<Real>::MatrixInternal& matrixInternal = sourceImpl.internal();

  Integer block_size = 1;
  const Block* block = sourceImpl.block();
  if (block) {
    assert(block->sizeX() == block->sizeY());
    block_size = block->sizeX();
  }
  ConstArrayView<Integer> row_offset = profile.getRowOffset();
  ConstArrayView<Integer> cols = profile.getCols();
  ConstArrayView<Real> values = matrixInternal.getValues();
  if (not targetImpl.initDiagMatrix(0, block_size, localSize,
          row_offset.unguardedBasePointer(), cols.unguardedBasePointer())) {
    throw FatalErrorException(A_FUNCINFO, "MCGSolver Initialisation failed");
  }

  const bool success = targetImpl.initMatrixValues(0, 0, values.unguardedBasePointer());

  if (not success) {
    throw FatalErrorException(A_FUNCINFO, "Cannot set MCGSolver Matrix Values");
  }
}

void
Composite_to_MCG_MatrixConverter::_buildSubMatrix01(
    const SimpleCSRMatrix<Real>& sourceImpl, MCGCompositeMatrix& targetImpl) const
{
  const MatrixDistribution& dist = sourceImpl.distribution();
  const CSRStructInfo& profile = sourceImpl.getCSRProfile();
  const SimpleCSRMatrix<Real>::MatrixInternal& matrixInternal = sourceImpl.internal();

  ConstArrayView<Integer> row_offset = profile.getRowOffset();
  ConstArrayView<Integer> cols = profile.getCols();
  ConstArrayView<Real> values = matrixInternal.getValues();
  Integer block_size = 1;
  Integer block_size2 = 1;
  const Block* block = sourceImpl.block();
  if (block) {
    block_size = block->sizeX();
    block_size2 = block->sizeY();
  }
  Integer nrows = dist.localRowSize() / block_size; // TODO: check this line

  if (not targetImpl.initOffDiagMatrix(0, 1, block_size, block_size2, nrows,
          dist.localRowSize(), row_offset.unguardedBasePointer(),
          cols.unguardedBasePointer())) {
    throw FatalErrorException(A_FUNCINFO, "MCGSolver Initialisation failed");
  }

  const bool success = targetImpl.initMatrixValues(0, 1, values.unguardedBasePointer());
  if (not success) {
    throw FatalErrorException(A_FUNCINFO, "Cannot set MCGSolver Matrix Values");
  }
}

void
Composite_to_MCG_MatrixConverter::_buildSubMatrix10(
    const SimpleCSRMatrix<Real>& sourceImpl, MCGCompositeMatrix& targetImpl) const
{
  const MatrixDistribution& dist = sourceImpl.distribution();
  const CSRStructInfo& profile = sourceImpl.getCSRProfile();
  const SimpleCSRMatrix<Real>::MatrixInternal& matrixInternal = sourceImpl.internal();

  ConstArrayView<Integer> row_offset = profile.getRowOffset();
  ConstArrayView<Integer> cols = profile.getCols();
  ConstArrayView<Real> values = matrixInternal.getValues();

  Integer block_size = 1;
  Integer block_size2 = 1;
  const Block* block = sourceImpl.block();
  if (block) {
    block_size = block->sizeX();
    block_size2 = block->sizeY();
  }

  Integer nrows = dist.localRowSize() / block_size;
  if (not targetImpl.initOffDiagMatrix(1, 0, block_size, block_size2, nrows,
          dist.localRowSize(), row_offset.unguardedBasePointer(),
          cols.unguardedBasePointer())) {
    throw FatalErrorException(A_FUNCINFO, "MCGSolver Initialisation failed");
  }

  const bool success = targetImpl.initMatrixValues(1, 0, values.unguardedBasePointer());
  if (not success) {
    throw FatalErrorException(A_FUNCINFO, "Cannot set MCGSolver Matrix Values");
  }
}

void
Composite_to_MCG_MatrixConverter::_buildSubMatrix11(
    const SimpleCSRMatrix<Real>& sourceImpl, MCGCompositeMatrix& targetImpl) const
{
  const CSRStructInfo& profile = sourceImpl.getCSRProfile();
  const Integer localSize = profile.getNRow();

  const SimpleCSRMatrix<Real>::MatrixInternal& matrixInternal = sourceImpl.internal();
  ConstArrayView<Integer> row_offset = profile.getRowOffset();
  ConstArrayView<Integer> cols = profile.getCols();
  ConstArrayView<Real> values = matrixInternal.getValues();

  Integer block_size = 1;
  const Block* block = sourceImpl.block();
  if (block) {
    assert(block->sizeX() == block->sizeY());
    block_size = block->sizeX();
  }

  if (not targetImpl.initDiagMatrix(1, block_size, localSize,
          row_offset.unguardedBasePointer(), cols.unguardedBasePointer())) {
    throw FatalErrorException(A_FUNCINFO, "MCGSolver Initialisation failed");
  }
  const bool success = targetImpl.initMatrixValues(1, 1, values.unguardedBasePointer());
  if (not success) {
    throw FatalErrorException(A_FUNCINFO, "Cannot set MCGSolver Matrix Values");
  }
}

/*---------------------------------------------------------------------------*/

REGISTER_MATRIX_CONVERTER(Composite_to_MCG_MatrixConverter);
