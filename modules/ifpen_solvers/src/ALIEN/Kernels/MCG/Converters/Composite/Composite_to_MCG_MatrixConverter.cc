#include "alien/core/backend/IMatrixConverter.h"
#include "alien/core/backend/MatrixConverterRegisterer.h"

#include <iostream>

#include <alien/kernels/simple_csr/data_structure/CSRStructInfo.h>
#include <alien/kernels/simple_csr/data_structure/SimpleCSRMatrix.h>
#include <ALIEN/Kernels/Composite/DataStructure/CompositeMatrix.h>
#include <ALIEN/Kernels/MCG/DataStructure/MCGVector.h>
#include <ALIEN/Kernels/MCG/DataStructure/MCGMatrix.h>

#include "ALIEN/Kernels/Composite/CompositeBackEnd.h"
#include "ALIEN/Kernels/MCG/MCGBackEnd.h"

using namespace Alien;
using namespace Alien::SimpleCSRInternal;

/*---------------------------------------------------------------------------*/

class Composite_to_MCG_MatrixConverter : public IMatrixConverter
{
public:
  Composite_to_MCG_MatrixConverter();
  virtual ~Composite_to_MCG_MatrixConverter() { }
public:
  BackEndId sourceBackend() const { return AlgebraTraits<BackEnd::tag::composite>::name(); }
  BackEndId targetBackend() const { return AlgebraTraits<BackEnd::tag::mcgsolver>::name(); }
  void convert(const IMatrixImpl * sourceImpl, IMatrixImpl * targetImpl) const;
  void convert(const IMatrixImpl * sourceImpl, IMatrixImpl * targetImpl, Integer i, Integer j) const;
  
  void _build(const SimpleCSRMatrix<Real> & sourceImpl, MCGMatrix & targetImpl) const;
  void _buildBlock(const SimpleCSRMatrix<Real> & sourceImpl, MCGMatrix & targetImpl) const;
  void _buildSubMatrix01(const SimpleCSRMatrix<Real> & sourceImpl, MCGMatrix & targetImpl) const;
  void _buildSubMatrix10(const SimpleCSRMatrix<Real> & sourceImpl, MCGMatrix & targetImpl) const;
  void _buildSubMatrix11(const SimpleCSRMatrix<Real> & sourceImpl, MCGMatrix & targetImpl) const;
};

/*---------------------------------------------------------------------------*/

Composite_to_MCG_MatrixConverter::
Composite_to_MCG_MatrixConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
Composite_to_MCG_MatrixConverter::
convert(const IMatrixImpl * sourceImpl, IMatrixImpl * targetImpl) const
{
  const auto& v = cast<CompositeKernel::Matrix >(sourceImpl, sourceBackend());

  alien_debug([&] {
      cout() << "Converting CompositeMatrix: " << &v << " to MCGMatrix";
    });

  for(Integer i=0;i<v.size();++i)
  {
    for(Integer j=0;j<v.size();++j)
    {
      this->convert(sourceImpl,targetImpl,i,j);
    }
  }
}

void
Composite_to_MCG_MatrixConverter::
convert(const IMatrixImpl * sourceImpl, IMatrixImpl * targetImpl,int i, int j) const
{
  const auto& compo = cast<CompositeKernel::Matrix >(sourceImpl, sourceBackend());
  const SimpleCSRMatrix<Real>& v = compo(i,j).impl()->get<BackEnd::tag::simplecsr>();
  MCGMatrix & v2 = cast<MCGMatrix>(targetImpl, targetBackend());

  if(i==0)
  {
    if(j==0)
    {
      const ISpace & space = v.rowSpace();
      const MatrixDistribution& dist = v.distribution();
      v2.init(space, space, dist) ;
      //v2.initDistribution(dist);
      if(sourceImpl->block())
          _buildBlock(v,v2);
      else if(sourceImpl->vblock())
        throw FatalErrorException(A_FUNCINFO,"Block sizes are variable - builds not yet implemented");
      else
        _build(v,v2);
    }
    else
    {
      _buildSubMatrix01(v,v2);
    }
  }
  else
  {
    if(j==0)
    {
      _buildSubMatrix10(v,v2);
    }
    else
    {
      const ISpace & space = v.rowSpace();
      const MatrixDistribution& dist = v.distribution();
      v2.init(space, space, dist);
      _buildSubMatrix11(v,v2);
    }
  }
}

void
Composite_to_MCG_MatrixConverter::
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
Composite_to_MCG_MatrixConverter::
_buildSubMatrix01(const SimpleCSRMatrix<Real> & sourceImpl, MCGMatrix & targetImpl) const
{
  const MatrixDistribution& dist = sourceImpl.distribution();
  const CSRStructInfo & profile = sourceImpl.getCSRProfile();
  const SimpleCSRMatrix<Real>::MatrixInternal& matrixInternal = sourceImpl.internal();

  ConstArrayView<Integer> row_offset = profile.getRowOffset();
  ConstArrayView<Integer> cols = profile.getCols();
  ConstArrayView<Real> values = matrixInternal.getValues();
  Integer block_size = 1;
  const Block* block = sourceImpl.block();
  if(block)
    block_size = block->size();
  Integer nrows = dist.localRowSize() / block_size;
  if (not targetImpl.initSubMatrix01(nrows,
				     dist.localRowSize(),
				     row_offset.unguardedBasePointer(),
				     cols.unguardedBasePointer()))
  {
    throw FatalErrorException(A_FUNCINFO,"GPUSolver Initialisation failed");
  }

  const bool success = targetImpl.initSubMatrix01Values(values.unguardedBasePointer());
  if (not success)
  {
    throw FatalErrorException(A_FUNCINFO,"Cannot set GPUSolver Matrix Values");
  }
}

void
Composite_to_MCG_MatrixConverter::
_buildSubMatrix10(const SimpleCSRMatrix<Real> & sourceImpl, MCGMatrix & targetImpl) const
{
  const SimpleCSRMatrix<Real>::MatrixInternal& matrixInternal = sourceImpl.internal();
  ConstArrayView<Real> values = matrixInternal.getValues();
  const bool success = targetImpl.initSubMatrix10Values(values.unguardedBasePointer());
  if (not success)
  {
    throw FatalErrorException(A_FUNCINFO,"Cannot set GPUSolver Matrix Values");
  }
}

void
Composite_to_MCG_MatrixConverter::
_buildSubMatrix11(const SimpleCSRMatrix<Real> & sourceImpl, MCGMatrix & targetImpl) const
{
  const CSRStructInfo & profile = sourceImpl.getCSRProfile();
  const Integer localSize = profile.getNRow();

  const SimpleCSRMatrix<Real>::MatrixInternal& matrixInternal = sourceImpl.internal();
  ConstArrayView<Integer> row_offset = profile.getRowOffset();
  ConstArrayView<Integer> cols = profile.getCols();
  ConstArrayView<Real> values = matrixInternal.getValues();
  if (not targetImpl.initSubMatrix11(localSize,
				     row_offset.unguardedBasePointer(),
				     cols.unguardedBasePointer()))
  {
    throw FatalErrorException(A_FUNCINFO,"GPUSolver Initialisation failed");
  }
  const bool success = targetImpl.initSubMatrix11Values(values.unguardedBasePointer());
  if (not success)
  {
    throw FatalErrorException(A_FUNCINFO,"Cannot set GPUSolver Matrix Values");
  }
}

void
Composite_to_MCG_MatrixConverter::
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

REGISTER_MATRIX_CONVERTER(Composite_to_MCG_MatrixConverter);
