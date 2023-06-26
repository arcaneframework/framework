#include <alien/core/backend/IMatrixConverter.h>
#include <alien/core/backend/MatrixConverterRegisterer.h>

#include <iostream>
#include <alien/kernels/trilinos/data_structure/TrilinosMatrix.h>
//#include <alien/core/block/ComputeBlockOffsets.h>

#include <alien/kernels/trilinos/TrilinosBackEnd.h>
#include <alien/kernels/simple_csr/CSRStructInfo.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>

using namespace Alien;
using namespace Alien::SimpleCSRInternal;

/*---------------------------------------------------------------------------*/

template <typename TagT>
class SimpleCSR_to_Trilinos_MatrixConverter : public IMatrixConverter
{
 public:
  SimpleCSR_to_Trilinos_MatrixConverter() {}
  virtual ~SimpleCSR_to_Trilinos_MatrixConverter() {}
 public:
  BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }

  BackEndId targetBackend() const { return AlgebraTraits<TagT>::name(); }

  void convert(const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const;

  void _build(const SimpleCSRMatrix<Real>& sourceImpl,
      TrilinosMatrix<Real, TagT>& targetImpl) const;

  void _buildBlock(const SimpleCSRMatrix<Real>& sourceImpl,
      TrilinosMatrix<Real, TagT>& targetImpl) const;
};

/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/

template <typename TagT>
void
SimpleCSR_to_Trilinos_MatrixConverter<TagT>::convert(
    const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const
{
  const SimpleCSRMatrix<Real>& v =
      cast<SimpleCSRMatrix<Real>>(sourceImpl, sourceBackend());
  TrilinosMatrix<Real, TagT>& v2 =
      cast<TrilinosMatrix<Real, TagT>>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting SimpleCSRMatrix: " << &v << " to TrilinosMatrix " << &v2;
  });

  if (targetImpl->block())
    _buildBlock(v, v2);
  else if (targetImpl->vblock())
    throw FatalErrorException(
        A_FUNCINFO, "Block sizes are variable - builds not yet implemented");
  else
    _build(v, v2);
}

template <typename TagT>
void
SimpleCSR_to_Trilinos_MatrixConverter<TagT>::_build(
    const SimpleCSRMatrix<Real>& sourceImpl, TrilinosMatrix<Real, TagT>& targetImpl) const
{
  typedef SimpleCSRMatrix<Real>::MatrixInternal CSRMatrixType;

  const MatrixDistribution& dist = targetImpl.distribution();
  const CSRStructInfo& profile = sourceImpl.getCSRProfile();
  const Integer localSize = profile.getNRow();
  const Integer localOffset = dist.rowOffset();
  const Integer globalSize = dist.globalRowSize();

  auto const& matrixInternal = sourceImpl.internal();
  const Integer myRank = dist.parallelMng()->commRank();
  const Integer nProc = dist.parallelMng()->commSize();

  auto const& matrix_profile = sourceImpl.internal().getCSRProfile();
  int nrows = matrix_profile.getNRow();
  int const* kcol = matrix_profile.getRowOffset().unguardedBasePointer();
  int const* cols = matrix_profile.getCols().unguardedBasePointer();
  int block_size = sourceImpl.block() ? sourceImpl.block()->size() : 1;

  if (not targetImpl.initMatrix(dist.parallelMng(), localOffset, globalSize, nrows, kcol,
          cols, block_size, matrixInternal.getDataPtr())) {
    throw FatalErrorException(A_FUNCINFO, "Trilinos Initialisation failed");
  }
}

template <typename TagT>
void
SimpleCSR_to_Trilinos_MatrixConverter<TagT>::_buildBlock(
    const SimpleCSRMatrix<Real>& sourceImpl, TrilinosMatrix<Real, TagT>& targetImpl) const
{
}

#ifdef KOKKOS_ENABLE_SERIAL
template class SimpleCSR_to_Trilinos_MatrixConverter<Alien::BackEnd::tag::tpetraserial>;
typedef SimpleCSR_to_Trilinos_MatrixConverter<Alien::BackEnd::tag::tpetraserial>
    ConverterSerial;

REGISTER_MATRIX_CONVERTER(ConverterSerial);
#endif

#ifdef KOKKOS_ENABLE_OPENMP
template class SimpleCSR_to_Trilinos_MatrixConverter<Alien::BackEnd::tag::tpetraomp>;
typedef SimpleCSR_to_Trilinos_MatrixConverter<Alien::BackEnd::tag::tpetraomp>
    ConverterOMP;

REGISTER_MATRIX_CONVERTER(ConverterOMP);
#endif

#ifdef KOKKOS_ENABLE_THREADS
template class SimpleCSR_to_Trilinos_MatrixConverter<Alien::BackEnd::tag::tpetrapth>;
typedef SimpleCSR_to_Trilinos_MatrixConverter<Alien::BackEnd::tag::tpetrapth>
    ConverterPTH;

REGISTER_MATRIX_CONVERTER(ConverterPTH);
#endif

#ifdef KOKKOS_ENABLE_CUDA
template class SimpleCSR_to_Trilinos_MatrixConverter<Alien::BackEnd::tag::tpetracuda>;
typedef SimpleCSR_to_Trilinos_MatrixConverter<Alien::BackEnd::tag::tpetracuda>
    ConverterCUDA;

REGISTER_MATRIX_CONVERTER(ConverterCUDA);
#endif
