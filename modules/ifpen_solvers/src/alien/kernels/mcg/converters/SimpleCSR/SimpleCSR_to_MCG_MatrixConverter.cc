#include "alien/core/backend/IMatrixConverter.h"
#include "alien/core/backend/MatrixConverterRegisterer.h"

#include <iostream>

#include <alien/kernels/simple_csr/data_structure/CSRStructInfo.h>
#include <alien/kernels/simple_csr/data_structure/SimpleCSRMatrix.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>
#include <alien/kernels/mcg/data_structure/MCGVector.h>
#include <alien/kernels/mcg/data_structure/MCGMatrix.h>

#include <alien/kernels/mcg/MCGBackEnd.h>

using namespace Alien;
using namespace Alien::SimpleCSRInternal;

/*---------------------------------------------------------------------------*/

class SimpleCSR_to_MCG_MatrixConverter : public IMatrixConverter
//: public ICompositeMatrixConverter<2>
{
public:
  SimpleCSR_to_MCG_MatrixConverter();
  virtual ~SimpleCSR_to_MCG_MatrixConverter() { }
public:
  BackEndId sourceBackend() const { return AlgebraTraits<BackEnd::tag::simplecsr>::name(); }
  BackEndId targetBackend() const { return AlgebraTraits<BackEnd::tag::mcgsolver>::name(); }
  void convert(const IMatrixImpl * sourceImpl, IMatrixImpl * targetImpl) const;
  //void convert(const IMatrixImpl * sourceImpl, IMatrixImpl * targetImpl, int i, int j) const;

  void _build(const SimpleCSRMatrix<Real> & sourceImpl, MCGMatrix & targetImpl) const;
  void _buildBlock(const SimpleCSRMatrix<Real> & sourceImpl, MCGMatrix & targetImpl) const;
  void _buildSubMatrix01(const SimpleCSRMatrix<Real> & sourceImpl, MCGMatrix & targetImpl) const;
  void _buildSubMatrix10(const SimpleCSRMatrix<Real> & sourceImpl, MCGMatrix & targetImpl) const;
  void _buildSubMatrix11(const SimpleCSRMatrix<Real> & sourceImpl, MCGMatrix & targetImpl) const;
};

/*---------------------------------------------------------------------------*/

SimpleCSR_to_MCG_MatrixConverter::
SimpleCSR_to_MCG_MatrixConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
SimpleCSR_to_MCG_MatrixConverter::
convert(const IMatrixImpl * sourceImpl, IMatrixImpl * targetImpl) const
{
  const SimpleCSRMatrix<Real> & v = cast<SimpleCSRMatrix<Real> >(sourceImpl, sourceBackend());
  MCGMatrix & v2 = cast<MCGMatrix>(targetImpl, targetBackend());

  alien_debug([&] {
      cout() << "Converting SimpleCSRMatrix: " << &v << " to MCGMatrix " << &v2;
    });

  if(sourceImpl->block())
    _buildBlock(v,v2);
  else if(sourceImpl->vblock())
    throw FatalErrorException(A_FUNCINFO,"Block sizes are variable - builds not yet implemented");
  else
    _build(v,v2);
}

void
SimpleCSR_to_MCG_MatrixConverter::
_build(const SimpleCSRMatrix<Real> & sourceImpl, MCGMatrix & targetImpl) const
{
  const CSRStructInfo & profile = sourceImpl.getCSRProfile();
  const Integer localSize = profile.getNRow();
  const SimpleCSRMatrix<Real>::MatrixInternal& matrixInternal = sourceImpl.internal();

  targetImpl.setBlockSize(1,1);

  ConstArrayView<Integer> row_offset = profile.getRowOffset();
  ConstArrayView<Integer> cols = profile.getCols();
  ConstArrayView<Real> values = matrixInternal.getValues();
  if (not targetImpl.initMatrix(localSize,
                                row_offset.unguardedBasePointer(),
                                cols.unguardedBasePointer()))
  {
    throw FatalErrorException(A_FUNCINFO,"GPUSolver Initialisation failed");
  }


  const bool success = targetImpl.initMatrixValues(localSize,
						   row_offset.unguardedBasePointer(),
						   cols.unguardedBasePointer(),
						   values.unguardedBasePointer());

  if (not success)
  {
    throw FatalErrorException(A_FUNCINFO,"Cannot set GPUSolver Matrix Values");
  }
}

void
SimpleCSR_to_MCG_MatrixConverter::
_buildBlock(const SimpleCSRMatrix<Real> & sourceImpl, MCGMatrix & targetImpl) const
{
  const CSRStructInfo & profile = sourceImpl.getCSRProfile();
  const Integer local_size = profile.getNRow();
  const Integer block_size = sourceImpl.block()->size();
  targetImpl.setBlockSize(block_size,block_size);
  ConstArrayView<Integer> row_offset = profile.getRowOffset();
  ConstArrayView<Integer> cols = profile.getCols();

  const SimpleCSRMatrix<Real>::MatrixInternal& matrixInternal = sourceImpl.internal();
  ConstArrayView<Real> values = matrixInternal.getValues();
  if (not targetImpl.initMatrix(local_size,
				row_offset.unguardedBasePointer(),
				cols.unguardedBasePointer()))
  {
    throw FatalErrorException(A_FUNCINFO,"GPUSolver Initialisation failed");
  }

  const bool success = targetImpl.initMatrixValues(local_size,
                                                   row_offset.unguardedBasePointer(),
                                                   cols.unguardedBasePointer(),
                                                   values.unguardedBasePointer());

  if (not success)
  {
    throw FatalErrorException(A_FUNCINFO,"Cannot set GPUSolver Matrix Values");
  }
}

/*---------------------------------------------------------------------------*/

REGISTER_MATRIX_CONVERTER(SimpleCSR_to_MCG_MatrixConverter);
