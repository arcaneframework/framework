#include <alien/Kernels/Trilinos/TrilinosBackEnd.h>
#include <alien/Kernels/Trilinos/DataStructure/TrilinosInternal.h>

#include <alien/core/block/Block.h>
#include "TrilinosVector.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
template <typename ValueT, typename TagT>
TrilinosVector<ValueT, TagT>::TrilinosVector(const MultiVectorImpl* multi_impl)
: IVectorImpl(multi_impl, AlgebraTraits<TagT>::name())
, m_local_offset(0)
{}

/*---------------------------------------------------------------------------*/
template<typename ValueT,typename TagT>
TrilinosVector<ValueT,TagT>::~TrilinosVector()
{}

/*---------------------------------------------------------------------------*/
template<typename ValueT,typename TagT>
void
TrilinosVector<ValueT,TagT>::init(const VectorDistribution& dist, const bool need_allocate)
{
  if (need_allocate)
    allocate();
}

/*---------------------------------------------------------------------------*/
template<typename ValueT,typename TagT>
void
TrilinosVector<ValueT,TagT>::allocate()
{
  const VectorDistribution& dist = this->distribution();
  m_local_offset = dist.offset();
  const Integer globalSize = dist.globalSize() ;
  MPI_Comm* comm = static_cast<MPI_Comm*>(const_cast<IParallelMng*>(dist.parallelMng())->getMPICommunicator()) ;

  m_internal.reset(new VectorInternal(dist.offset(),globalSize,this->scalarizedLocalSize(),*comm));

  //m_internal->m_internal = 0.;
}

/*---------------------------------------------------------------------------*/

template<typename ValueT,typename TagT>
void
TrilinosVector<ValueT,TagT>::setValues(const int nrow, const ValueT* values)
{
  auto& x = *m_internal->m_internal ;
  x.sync_host ();
  auto x_2d = x.getLocalViewHost ();
  auto x_1d = Kokkos::subview (x_2d, Kokkos::ALL (), 0);
  const size_t localLength = x.getLocalLength ();
  x.modify_host ();
  for (int i= 0; i < nrow; ++i)
  {
    x_1d(i) = values[i] ;
  }
  using memory_space = typename VectorInternal::vector_type::device_type::memory_space;
  x.template sync<memory_space> ();
}

/*---------------------------------------------------------------------------*/
template<typename ValueT,typename TagT>
void
TrilinosVector<ValueT,TagT>::getValues(const int nrow, ValueT* values) const
{
  auto& x = *m_internal->m_internal ;
  x.sync_host ();
  auto x_2d = x.getLocalViewHost ();
  auto x_1d = Kokkos::subview (x_2d, Kokkos::ALL (), 0);
  const size_t localLength = x.getLocalLength ();
  for (int i = 0; i < nrow; ++i)
    values[i] = x_1d(i);
}

template<typename ValueT,typename TagT>
void
TrilinosVector<ValueT,TagT>::
dump() const
{
  auto& x = *m_internal->m_internal ;
  x.sync_host ();
  auto x_2d = x.getLocalViewHost ();
  auto x_1d = Kokkos::subview (x_2d, Kokkos::ALL (), 0);
  const size_t localLength = x.getLocalLength ();
  for (int i = 0; i < localLength; ++i)
    std::cout << x_1d(i)<<" "<<std::endl ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
template class TrilinosVector<double,BackEnd::tag::tpetraserial> ;
#ifdef KOKKOS_ENABLE_OPENMP
template class TrilinosVector<double,BackEnd::tag::tpetraomp> ;
#endif
#ifdef KOKKOS_ENABLE_THREADS
template class TrilinosVector<double,BackEnd::tag::tpetrapth> ;
#endif
} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
