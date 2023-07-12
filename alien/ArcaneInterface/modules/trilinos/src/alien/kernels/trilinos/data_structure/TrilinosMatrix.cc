#include <alien/kernels/trilinos/TrilinosBackEnd.h>
#include <alien/kernels/trilinos/data_structure/TrilinosInternal.h>

#include <alien/core/impl/MultiMatrixImpl.h>
#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>
#include "TrilinosMatrix.h"
#include "TrilinosVector.h"

/*---------------------------------------------------------------------------*/
BEGIN_TRILINOSINTERNAL_NAMESPACE

template <typename ValueT, typename TagT>
bool
MatrixInternal<ValueT, TagT>::initMatrix(int local_offset, int nrows, int const* kcol,
    int const* cols, int block_size, ValueT const* values)
{
  m_local_offset = local_offset;
  m_local_size = nrows;
  int const* cols_ptr = cols;
  ValueT const* values_ptr = values;
  auto& csr_matrix = *m_internal;
  for (int irow = 0; irow < nrows; ++irow) {
    int row_size = kcol[irow + 1] - kcol[irow];
    csr_matrix.insertGlobalValues(local_offset + irow, row_size, values_ptr, cols_ptr);
    cols_ptr += row_size;
    values_ptr += row_size;
  }
  csr_matrix.fillComplete();

  return true;
}

template <typename ValueT, typename TagT>
bool
MatrixInternal<ValueT, TagT>::setMatrixValues(Real const* values)
{
  using Teuchos::Array;
  using Teuchos::ArrayView;
  Real const* values_ptr = values;
  auto& csr_matrix = *m_internal;
  for (int irow = 0; irow < m_local_size; ++irow) {
    size_t row_size = csr_matrix.getNumEntriesInLocalRow(m_local_offset + irow);

    Array<scalar_type> rowvals(row_size);
    Array<global_ordinal_type> cols(row_size);
    csr_matrix.getGlobalRowCopy(irow, cols(), rowvals(), row_size);
    for (std::size_t k = 0; k < row_size; ++k)
      rowvals[k] = values_ptr[k];
    csr_matrix.replaceGlobalValues(m_local_offset + irow, cols, rowvals());
    values_ptr += row_size;
  }
  return true;
}

template <typename ValueT, typename TagT>
void
MatrixInternal<ValueT, TagT>::mult(vector_type const& x, vector_type& y) const
{
  m_internal->apply(x, y);
}

template <typename ValueT, typename TagT>
void
MatrixInternal<ValueT, TagT>::mult(ValueT const* x, ValueT* y) const
{
  // m_internal->apply(x,y) ;
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

template <typename ValueT, typename TagT> TrilinosMatrix<ValueT, TagT>::~TrilinosMatrix()
{
}

/*---------------------------------------------------------------------------*/

template <typename ValueT, typename TagT>
bool
TrilinosMatrix<ValueT, TagT>::initMatrix(IMessagePassingMng const* parallel_mng,
    int local_offset, int global_size, int nrows, int const* kcol, int const* cols,
    int block_size, ValueT const* values)
{
  using namespace Arccore::MessagePassing::Mpi;
  auto* parallel_mng_ = const_cast<IMessagePassingMng*>(parallel_mng);
  const auto* mpi_mng = dynamic_cast<const MpiMessagePassingMng*>(parallel_mng_);
  const MPI_Comm* comm = static_cast<const MPI_Comm*>(mpi_mng->getMPIComm());
  m_internal.reset(new MatrixInternal(local_offset, global_size, nrows, *comm));

  return m_internal->initMatrix(local_offset, nrows, kcol, cols, block_size, values);
}

template <typename ValueT, typename TagT>
bool
TrilinosMatrix<ValueT, TagT>::setMatrixValues(Real const* values)
{
  return m_internal->setMatrixValues(values);
}

template<typename ValueT,typename TagT>
void TrilinosMatrix<ValueT,TagT>::setMatrixCoordinate(Vector const& x, Vector const& y, Vector const& z)
{
  typedef typename MatrixInternal::coord_vector_type CoordVectorType ;
  typedef typename MatrixInternal::real_type         RealType ;
  assert(m_internal.get()) ;
  m_internal->m_coordinates = rcp( new CoordVectorType(m_internal->m_map, 3, false) );

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<RealType> > Coord(3);
  Coord[0] = m_internal->m_coordinates->getDataNonConst(0);
  Coord[1] = m_internal->m_coordinates->getDataNonConst(1);
  Coord[2] = m_internal->m_coordinates->getDataNonConst(2);

  Alien::LocalVectorReader x_view(x);
  Alien::LocalVectorReader y_view(y);
  Alien::LocalVectorReader z_view(z);

  for(int i=0;i<m_internal->m_local_size;++i)
  {
    Coord[0][i] = x_view[i] ;
    Coord[1][i] = y_view[i] ;
    Coord[2][i] = z_view[i] ;
  }
}

template <typename ValueT, typename TagT>
void
TrilinosMatrix<ValueT, TagT>::mult(
    TrilinosVector<ValueT, TagT> const& x, TrilinosVector<ValueT, TagT>& y) const
{
  m_internal->mult(*x.internal()->m_internal, *y.internal()->m_internal);
}

template <typename ValueT, typename TagT>
void
TrilinosMatrix<ValueT, TagT>::mult(ValueT const* x, ValueT* y) const
{
  m_internal->mult(x, y);
}

template <typename ValueT, typename TagT>
void
TrilinosMatrix<ValueT, TagT>::dump(std::string const& filename) const
{
  Tpetra::MatrixMarket::Writer<typename MatrixInternal::matrix_type>::writeSparseFile(
      filename, *m_internal->m_internal);
}

#ifdef KOKKOS_ENABLE_SERIAL
template class TrilinosMatrix<double, BackEnd::tag::tpetraserial>;
#endif
#ifdef KOKKOS_ENABLE_OPENMP
template class TrilinosMatrix<double, BackEnd::tag::tpetraomp>;
#endif
#ifdef KOKKOS_ENABLE_THREADS
template class TrilinosMatrix<double, BackEnd::tag::tpetrapth>;
#endif
#ifdef KOKKOS_ENABLE_CUDA
template class TrilinosMatrix<double, BackEnd::tag::tpetracuda>;
#endif

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
