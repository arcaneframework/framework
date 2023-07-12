#include <alien/core/backend/IMatrixConverter.h>
#include <alien/core/backend/MatrixConverterRegisterer.h>

#include <iostream>
#include <alien/kernels/ifp/data_structure/IFPMatrix.h>

#include <alien/kernels/ifp/IFPSolverBackEnd.h>
#include <alien/kernels/simple_csr/CSRStructInfo.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>

using namespace Alien;
using namespace Alien::SimpleCSRInternal;

/*---------------------------------------------------------------------------*/

class SimpleCSR_to_IFP_MatrixConverter : public IMatrixConverter
{
 public:
  SimpleCSR_to_IFP_MatrixConverter();
  virtual ~SimpleCSR_to_IFP_MatrixConverter() {}
 public:
  BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }
  BackEndId targetBackend() const
  {
    return AlgebraTraits<BackEnd::tag::ifpsolver>::name();
  }
  void convert(const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const;
  void _build(const SimpleCSRMatrix<Real>& sourceImpl, IFPMatrix& targetImpl) const;
  void _buildBlock(const SimpleCSRMatrix<Real>& sourceImpl, IFPMatrix& targetImpl) const;
};

/*---------------------------------------------------------------------------*/

SimpleCSR_to_IFP_MatrixConverter::SimpleCSR_to_IFP_MatrixConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
SimpleCSR_to_IFP_MatrixConverter::convert(
    const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const
{
  const SimpleCSRMatrix<Real>& v =
      cast<SimpleCSRMatrix<Real>>(sourceImpl, sourceBackend());
  IFPMatrix& v2 = cast<IFPMatrix>(targetImpl, targetBackend());
  if (sourceImpl->block())
    _buildBlock(v, v2);
  else if (sourceImpl->vblock())
    throw FatalErrorException(
        A_FUNCINFO, "Block sizes are variable - builds not yet implemented");
  else
    _build(v, v2);
}

void
SimpleCSR_to_IFP_MatrixConverter::_build(
    const SimpleCSRMatrix<Real>& sourceImpl, IFPMatrix& targetImpl) const
{
  const MatrixDistribution& dist = sourceImpl.distribution();
  const CSRStructInfo& profile = sourceImpl.getCSRProfile();
  const SimpleCSRMatrix<Real>::MatrixInternal& matrixInternal = sourceImpl.internal();
  const Integer localSize = profile.getNRow();
  const Integer globalSize = dist.globalRowSize();
  const Integer localOffset = dist.rowOffset();
  Int64 profile_timestamp = profile.timestamp();

  ConstArrayView<Integer> row_offset = profile.getRowOffset();
  ConstArrayView<Integer> cols = profile.getCols();
  ConstArrayView<Real> m_values = matrixInternal.getValues();

  targetImpl.setSymmetricProfile(profile.getSymmetric());
  if (not targetImpl.initMatrix(1, 1, globalSize, localSize, localOffset, row_offset,
          cols, profile_timestamp)) {
    throw FatalErrorException(A_FUNCINFO, "IFPSolver Initialisation failed");
  }

  const bool success = targetImpl.setMatrixValues(m_values.unguardedBasePointer());
  if (not success) {
    throw FatalErrorException(A_FUNCINFO, "Cannot set IFPSolver Matrix Values");
  }
}

void
SimpleCSR_to_IFP_MatrixConverter::_buildBlock(
    const SimpleCSRMatrix<Real>& sourceImpl, IFPMatrix& targetImpl) const
{
  const MatrixDistribution& dist = sourceImpl.distribution();
  const CSRStructInfo& profile = sourceImpl.getCSRProfile();
  const SimpleCSRMatrix<Real>::MatrixInternal& matrixInternal = sourceImpl.internal();

  const Integer block_size = sourceImpl.block()->size();
  const Integer localSize = profile.getNRow();
  const Integer globalSize = dist.globalRowSize();
  const Integer localOffset = dist.rowOffset();
  Int64 profile_timestamp = profile.timestamp();

  ConstArrayView<Integer> row_offset = profile.getRowOffset();
  ConstArrayView<Integer> cols = profile.getCols();
  ConstArrayView<Real> m_values = matrixInternal.getValues();

  targetImpl.setSymmetricProfile(profile.getSymmetric());

  if (not targetImpl.initMatrix(block_size, block_size, globalSize, localSize,
          localOffset, row_offset, cols, profile_timestamp)) {
    throw FatalErrorException(A_FUNCINFO, "IFPSolver Initialisation failed");
  }

  const bool success = targetImpl.setMatrixValues(m_values.unguardedBasePointer());
  if (not success) {
    throw FatalErrorException(A_FUNCINFO, "Cannot set IFPSolver Matrix Values");
  }
}

/*---------------------------------------------------------------------------*/

REGISTER_MATRIX_CONVERTER(SimpleCSR_to_IFP_MatrixConverter);
