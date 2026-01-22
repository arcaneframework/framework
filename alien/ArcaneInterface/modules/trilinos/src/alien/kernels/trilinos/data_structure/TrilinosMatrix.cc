// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
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
    int const* cols, int block_size, int max_row_size, ValueT const* values)
{
  m_local_offset = local_offset;
  m_local_size = nrows;
  int const* cols_ptr = cols;
  ValueT const* values_ptr = values;
  auto& csr_matrix = *m_internal;
  if(block_size>1)
  {
    m_local_offset *= block_size ;
    m_local_size *= block_size ;
    int block2_size = block_size*block_size;

    Arcane::UniqueArray<int> row_indices(max_row_size*block_size) ;
    Arcane::UniqueArray2<ValueT> row_values(block_size,max_row_size*block_size) ;
    for (int irow = 0; irow < nrows; ++irow)
    {
       int row_size = kcol[irow + 1] - kcol[irow];
       int jcol = 0 ;
       for(int k=kcol[irow];k<kcol[irow + 1];++k)
         for(int j=0;j<block_size;++j)
         {
           row_indices[jcol] = cols[k] * block_size + j ;
           for( int i=0;i<block_size;++i)
           {
             row_values[i][jcol] = values[ k * block2_size + i * block_size + j] ;
           }
            ++jcol;
         }
       for( int i=0;i<block_size;++i)
       {
          csr_matrix.insertGlobalValues(m_local_offset + irow*block_size + i,
                                        row_size * block_size,
                                        row_values[i].data(),
                                        row_indices.data());
       }
    }
  }
  else
  {
    for (int irow = 0; irow < nrows; ++irow) {
      int row_size = kcol[irow + 1] - kcol[irow];
      csr_matrix.insertGlobalValues(local_offset + irow, row_size, values_ptr, cols_ptr);
      cols_ptr += row_size;
      values_ptr += row_size;
    }
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

#if (TRILINOS_MAJOR_VERSION < 15)
    Array<scalar_type> rowvals(row_size);
    Array<global_ordinal_type> cols(row_size);
    csr_matrix.getGlobalRowCopy(irow, cols(), rowvals(), row_size);
    for (std::size_t k = 0; k < row_size; ++k)
      rowvals[k] = values_ptr[k];
    csr_matrix.replaceGlobalValues(m_local_offset + irow, cols, rowvals());
#else
    typename MatrixInternal::matrix_type::nonconst_global_inds_host_view_type cols("Inds",row_size);
    typename MatrixInternal::matrix_type::nonconst_values_host_view_type rowvals("Vals",row_size);
    csr_matrix.getGlobalRowCopy(irow, cols, rowvals, row_size);
    for (std::size_t k = 0; k < row_size; ++k)
      rowvals[k] = values_ptr[k];
    csr_matrix.replaceGlobalValues(m_local_offset + irow, cols, rowvals);
#endif
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
MatrixInternal<ValueT, TagT>::mult([[maybe_unused]] ValueT const* x,[[maybe_unused]]  ValueT* y) const
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
  auto* pm = dynamic_cast<MpiMessagePassingMng*>(const_cast<IMessagePassingMng*>(parallel_mng));

  int max_row_size = 0 ;
  std::vector<std::size_t> row_size(nrows*block_size) ;
  for (int irow = 0; irow < nrows; ++irow) {
    int size = (kcol[irow + 1] - kcol[irow])*block_size ;
    for(int k=0;k<block_size;++k)
      row_size[irow*block_size+k] = size ;
    max_row_size = std::max(max_row_size,size) ;
  }

  if(pm && *static_cast<const MPI_Comm*>(pm->getMPIComm()) != MPI_COMM_NULL)
    m_internal.reset(new MatrixInternal(local_offset*block_size, global_size*block_size, nrows*block_size,row_size.data(),*static_cast<const MPI_Comm*>(pm->getMPIComm())));
  else
    m_internal.reset(new MatrixInternal(local_offset*block_size, global_size*block_size, nrows*block_size, row_size.data(), MPI_COMM_WORLD));

  return m_internal->initMatrix(local_offset, nrows, kcol, cols, block_size, max_row_size, values);
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
template class ALIEN_TRILINOS_EXPORT TrilinosMatrix<double, BackEnd::tag::tpetraserial>;
#endif
#ifdef KOKKOS_ENABLE_OPENMP
template class ALIEN_TRILINOS_EXPORT TrilinosMatrix<double, BackEnd::tag::tpetraomp>;
#endif
#ifdef KOKKOS_ENABLE_THREADS
template class ALIEN_TRILINOS_EXPORT TrilinosMatrix<double, BackEnd::tag::tpetrapth>;
#endif
#ifdef KOKKOS_ENABLE_CUDA
template class ALIEN_TRILINOS_EXPORT TrilinosMatrix<double, BackEnd::tag::tpetracuda>;
#endif

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
