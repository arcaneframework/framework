#include <alien/Kernels/Trilinos/TrilinosBackEnd.h>
#include <alien/Kernels/Trilinos/DataStructure/TrilinosInternal.h>

#include <alien/core/impl/MultiMatrixImpl.h>
#include "TrilinosMatrix.h"

/*---------------------------------------------------------------------------*/
BEGIN_TRILINOSINTERNAL_NAMESPACE


template<typename ValueT,typename TagT>
bool
MatrixInternal<ValueT,TagT>::initMatrix(int local_offset,
                                    int nrows,
                                    int const* kcol,
                                    int const* cols,
                                    int block_size,
                                    ValueT const* values)
{
  m_local_offset           = local_offset ;
  m_local_size             = nrows ;
  int const* cols_ptr      = cols ;
  ValueT const* values_ptr = values;
  auto& csr_matrix = *m_internal ;
  for(int irow=0;irow<nrows;++irow)
  {
    int row_size = kcol[irow+1] - kcol[irow] ;
    csr_matrix.insertGlobalValues (local_offset+irow,row_size,values_ptr,cols_ptr) ;
    cols_ptr += row_size ;
    values_ptr += row_size ;
  }
  csr_matrix.fillComplete () ;

  //Tpetra::MatrixMarket::Writer<matrix_type>::writeSparseFile("MatrixA.txt",csr_matrix);
  return true ;
}

template<typename ValueT,typename TagT>
bool
MatrixInternal<ValueT,TagT>::setMatrixValues(Real const* values)
{
  using Teuchos::Array;
  using Teuchos::ArrayView;
  Real const* values_ptr = values ;
  auto& csr_matrix = *m_internal ;
  for(int irow=0;irow<m_local_size;++irow)
  {
    size_t row_size = csr_matrix.getNumEntriesInLocalRow (m_local_offset+irow);

    Array<scalar_type>            rowvals(row_size) ;
    Array<global_ordinal_type>    cols(row_size) ;
    csr_matrix.getGlobalRowCopy (irow,cols(),rowvals(),row_size);
    for(std::size_t k=0;k<row_size;++k)
      rowvals[k] = values_ptr[k] ;
    csr_matrix.replaceGlobalValues (m_local_offset+irow, cols,rowvals());
    values_ptr += row_size ;
  }
  return true ;
}


END_TRILINOSINTERNAL_NAMESPACE

namespace Alien {
/*---------------------------------------------------------------------------*/
template <typename ValueT, typename TagT>
TrilinosMatrix<ValueT, TagT>::TrilinosMatrix(const MultiMatrixImpl* multi_impl)
: IMatrixImpl(multi_impl, AlgebraTraits<TagT>::name())
{
  const auto& row_space = multi_impl->rowSpace();
  const auto& col_space = multi_impl->colSpace();

  if (row_space.size() != col_space.size())
    throw FatalErrorException("Trilinos matrix must be square");
}

/*---------------------------------------------------------------------------*/

template<typename ValueT,typename TagT>
TrilinosMatrix<ValueT,TagT>::~TrilinosMatrix()
{}

/*---------------------------------------------------------------------------*/

template<typename ValueT,typename TagT>
bool
TrilinosMatrix<ValueT,TagT>::initMatrix( IParallelMng const* parallel_mng,
                                    int local_offset,
                                    int global_size,
                                    int nrows,
                                    int const* kcol,
                                    int const* cols,
                                    int block_size,
                                    ValueT const* values)
{
  MPI_Comm* comm = static_cast<MPI_Comm*>(const_cast<IParallelMng*>(parallel_mng)->getMPICommunicator()) ;
  m_internal.reset(new MatrixInternal(local_offset,global_size,nrows,*comm));

  return m_internal->initMatrix(local_offset,nrows,kcol,cols,block_size,values) ;
}

template<typename ValueT,typename TagT>
bool
TrilinosMatrix<ValueT,TagT>::setMatrixValues(Real const* values)
{
  return m_internal->setMatrixValues(values) ;
}


template<typename ValueT,typename TagT>
void
TrilinosMatrix<ValueT,TagT>::mult(ValueT const* x, ValueT* y) const
{
}


template class TrilinosMatrix<double,BackEnd::tag::tpetraserial>;
#ifdef KOKKOS_ENABLE_OPENMP
template class TrilinosMatrix<double,BackEnd::tag::tpetraomp>;
#endif
#ifdef KOKKOS_ENABLE_THREADS
template class TrilinosMatrix<double,BackEnd::tag::tpetrapth>;
#endif
} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
