#include <ALIEN/Core/Backend/IMatrixConverter.h>
#include <ALIEN/Core/Backend/MatrixConverterRegisterer.h>

#include <iostream>
#include <ALIEN/Kernels/HTS/DataStructure/HTSMatrix.h>
#include <ALIEN/Core/Block/ComputeBlockOffsets.h>

#include <ALIEN/Kernels/HTS/HTSBackEnd.h>
#include <ALIEN/Kernels/SimpleCSR/DataStructure/CSRStructInfo.h>
#include <ALIEN/Kernels/SimpleCSR/DataStructure/SimpleCSRMatrix.h>
#include <ALIEN/Kernels/SimpleCSR/SimpleCSRBackEnd.h>

using namespace Alien;
using namespace Alien::SimpleCSRInternal;

/*---------------------------------------------------------------------------*/

class SimpleCSR_to_HTS_MatrixConverter : public IMatrixConverter
{
 public:
  SimpleCSR_to_HTS_MatrixConverter();
  virtual ~SimpleCSR_to_HTS_MatrixConverter() {}
 public:
  BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }
  BackEndId targetBackend() const { return AlgebraTraits<BackEnd::tag::hts>::name(); }
  void convert(const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const;
  void _build(const SimpleCSRMatrix<Arccore::Real>& sourceImpl, HTSMatrix<Arccore::Real>& targetImpl) const;
  void _buildBlock(const SimpleCSRMatrix<Arccore::Real>& sourceImpl, HTSMatrix<Arccore::Real>& targetImpl) const;
};

/*---------------------------------------------------------------------------*/

SimpleCSR_to_HTS_MatrixConverter::SimpleCSR_to_HTS_MatrixConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
SimpleCSR_to_HTS_MatrixConverter::convert(
    const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const
{
  const SimpleCSRMatrix<Arccore::Real>& v =
      cast<SimpleCSRMatrix<Arccore::Real>>(sourceImpl, sourceBackend());
  HTSMatrix<Arccore::Real>& v2 = cast<HTSMatrix<Arccore::Real>>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting SimpleCSRMatrix: " << &v << " to HTSMatrix " << &v2;
  });

  if(targetImpl->block())
    _buildBlock(v,v2);
  else if(targetImpl->vblock())
    throw Arccore::FatalErrorException(A_FUNCINFO,"Block sizes are variable - builds not yet implemented");
  else
    _build(v,v2);
}

void
SimpleCSR_to_HTS_MatrixConverter::_build(
    const SimpleCSRMatrix<Arccore::Real>& sourceImpl, HTSMatrix<Arccore::Real>& targetImpl) const
{
  typedef SimpleCSRMatrix<Arccore::Real>::MatrixInternal CSRMatrixType ;

  const MatrixDistribution& dist = targetImpl.distribution();
  const CSRStructInfo & profile = sourceImpl.getCSRProfile();
  const Arccore::Integer localSize = profile.getNRow();
  const Arccore::Integer localOffset = dist.rowOffset();
  auto const& matrixInternal = sourceImpl.internal();
  const Arccore::Integer myRank = dist.parallelMng()->commRank();
  const Arccore::Integer nProc = dist.parallelMng()->commSize();


  {

    auto const& matrix_profile = sourceImpl.internal().getCSRProfile() ;
    int nrows = matrix_profile.getNRow() ;
    int const* kcol = matrix_profile.getRowOffset().unguardedBasePointer() ;
    int const* cols = matrix_profile.getCols().unguardedBasePointer() ;
    int block_size = sourceImpl.block() ? sourceImpl.block()->size() : 1 ;

    if (not targetImpl.initMatrix(dist.parallelMng(),nrows,kcol,cols,block_size))
    {
      throw Arccore::FatalErrorException(A_FUNCINFO, "HTS Initialisation failed");
    }

    if (not targetImpl.setMatrixValues(matrixInternal.getValues().data()))
    {
      throw Arccore::FatalErrorException(A_FUNCINFO, "Cannot set HTS Matrix Values");
    }

    if(not targetImpl.computeDDMatrix())
    {
      throw Arccore::FatalErrorException(A_FUNCINFO, "Cannot set HTS DDMatrix Values");
    }
  }
}

void
SimpleCSR_to_HTS_MatrixConverter::_buildBlock(
    const SimpleCSRMatrix<Arccore::Real>& sourceImpl, HTSMatrix<Arccore::Real>& targetImpl) const
{

}

/*---------------------------------------------------------------------------*/

REGISTER_MATRIX_CONVERTER(SimpleCSR_to_HTS_MatrixConverter);
