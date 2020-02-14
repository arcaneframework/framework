#include <ALIEN/Core/Backend/IMatrixConverter.h>
#include <ALIEN/Core/Backend/MatrixConverterRegisterer.h>

#include <iostream>
#include <ALIEN/Kernels/IFP/DataStructure/IFPMatrix.h>

#include <ALIEN/Kernels/IFP/IFPSolverBackEnd.h>
#include <ALIEN/Kernels/SimpleCSR/DataStructure/CSRStructInfo.h>
#include <ALIEN/Kernels/SimpleCSR/DataStructure/SimpleCSRMatrix.h>
#include <ALIEN/Kernels/SimpleCSR/SimpleCSRBackEnd.h>

using namespace Alien;
using namespace Alien::SimpleCSRInternal;

/*---------------------------------------------------------------------------*/

class SimpleCSR_to_IFP_MatrixConverter : public IMatrixConverter 
{
public:
  SimpleCSR_to_IFP_MatrixConverter();
  virtual ~SimpleCSR_to_IFP_MatrixConverter() { }
public:
  BackEndId sourceBackend() const { return AlgebraTraits<BackEnd::tag::simplecsr>::name(); }
  BackEndId targetBackend() const { return AlgebraTraits<BackEnd::tag::ifpsolver>::name(); }
  void convert(const IMatrixImpl * sourceImpl, IMatrixImpl * targetImpl) const;
  void _build(const SimpleCSRMatrix<Arccore::Real> & sourceImpl, IFPMatrix & targetImpl) const;
  void _buildBlock(const SimpleCSRMatrix<Arccore::Real> & sourceImpl, IFPMatrix & targetImpl) const;
};

/*---------------------------------------------------------------------------*/

SimpleCSR_to_IFP_MatrixConverter::
SimpleCSR_to_IFP_MatrixConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
SimpleCSR_to_IFP_MatrixConverter::
convert(const IMatrixImpl * sourceImpl, IMatrixImpl * targetImpl) const
{
  const SimpleCSRMatrix<Arccore::Real> & v = cast<SimpleCSRMatrix<Arccore::Real> >(sourceImpl, sourceBackend());
  IFPMatrix & v2 = cast<IFPMatrix>(targetImpl, targetBackend());
  if(sourceImpl->block())
    _buildBlock(v,v2);
  else if(sourceImpl->vblock())
    throw Arccore::FatalErrorException(A_FUNCINFO,"Block sizes are variable - builds not yet implemented");
  else
    _build(v,v2);
}

void
SimpleCSR_to_IFP_MatrixConverter::
_build(const SimpleCSRMatrix<Arccore::Real> & sourceImpl, IFPMatrix & targetImpl) const
{
  const MatrixDistribution& dist = sourceImpl.distribution();
  const CSRStructInfo & profile = sourceImpl.getCSRProfile();
  const SimpleCSRMatrix<Arccore::Real>::MatrixInternal& matrixInternal = sourceImpl.internal();
  const Arccore::Integer localSize = profile.getNRow();
  const Arccore::Integer globalSize = dist.globalRowSize();
  const Arccore::Integer localOffset = dist.rowOffset();
  Arccore::Int64 profile_timestamp = profile.timestamp() ;

  Arccore::ConstArrayView<Arccore::Integer> row_offset = profile.getRowOffset() ;
  Arccore::ConstArrayView<Arccore::Integer> cols = profile.getCols() ;
  Arccore::ConstArrayView<Arccore::Real> m_values = matrixInternal.getValues() ;
  
  targetImpl.setSymmetricProfile(profile.getSymmetric());
  if (not targetImpl.initMatrix(1,1,
                            globalSize,
                            localSize,
                            localOffset,
                            row_offset,
                            cols,
                            profile_timestamp
                            ))
  {
    throw Arccore::FatalErrorException(A_FUNCINFO,"IFPSolver Initialisation failed");
  }
  
  const bool success = targetImpl.setMatrixValues(m_values.unguardedBasePointer());
  if (not success)
  {
    throw Arccore::FatalErrorException(A_FUNCINFO,"Cannot set IFPSolver Matrix Values");
  }
}

void
SimpleCSR_to_IFP_MatrixConverter::
_buildBlock(const SimpleCSRMatrix<Arccore::Real> & sourceImpl, IFPMatrix & targetImpl) const
{
  const MatrixDistribution& dist = sourceImpl.distribution();
  const CSRStructInfo & profile = sourceImpl.getCSRProfile();
  const SimpleCSRMatrix<Arccore::Real>::MatrixInternal& matrixInternal = sourceImpl.internal();

  const Arccore::Integer block_size = sourceImpl.block()->size();
  const Arccore::Integer localSize = profile.getNRow();
  const Arccore::Integer globalSize = dist.globalRowSize();
  const Arccore::Integer localOffset = dist.rowOffset();
  Arccore::Int64 profile_timestamp = profile.timestamp() ;

  Arccore::ConstArrayView<Arccore::Integer> row_offset = profile.getRowOffset() ;
  Arccore::ConstArrayView<Arccore::Integer> cols = profile.getCols() ;
  Arccore::ConstArrayView<Arccore::Real> m_values = matrixInternal.getValues() ;
  
  targetImpl.setSymmetricProfile(profile.getSymmetric());
  
  
  if (not targetImpl.initMatrix(block_size,block_size,
                            globalSize,
                            localSize,
                            localOffset,
                            row_offset,
                            cols,
                            profile_timestamp
                            ))
  {
    throw Arccore::FatalErrorException(A_FUNCINFO,"IFPSolver Initialisation failed");
  }
  
  const bool success = targetImpl.setMatrixValues(m_values.unguardedBasePointer());
  if (not success)
  {
    throw Arccore::FatalErrorException(A_FUNCINFO,"Cannot set IFPSolver Matrix Values");
  }
}

/*---------------------------------------------------------------------------*/

REGISTER_MATRIX_CONVERTER(SimpleCSR_to_IFP_MatrixConverter);
