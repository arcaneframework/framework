#include <alien/core/backend/IVectorConverter.h>
#include <alien/core/backend/VectorConverterRegisterer.h>

#include <iostream>
#include <alien/kernels/trilinos/TrilinosBackEnd.h>
#include <alien/kernels/trilinos/data_structure/TrilinosVector.h>

#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>
using namespace Alien;

/*---------------------------------------------------------------------------*/
template <typename TagT>
class SimpleCSR_to_Trilinos_VectorConverter : public IVectorConverter
{
 public:
  SimpleCSR_to_Trilinos_VectorConverter();
  virtual ~SimpleCSR_to_Trilinos_VectorConverter() {}
 public:
  Alien::BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }
  Alien::BackEndId targetBackend() const { return AlgebraTraits<TagT>::name(); }
  void convert(const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const;
};

/*---------------------------------------------------------------------------*/
template <typename TagT>
SimpleCSR_to_Trilinos_VectorConverter<TagT>::SimpleCSR_to_Trilinos_VectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/
template <typename TagT>
void
SimpleCSR_to_Trilinos_VectorConverter<TagT>::convert(
    const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const SimpleCSRVector<double>& v =
      cast<SimpleCSRVector<double>>(sourceImpl, sourceBackend());
  TrilinosVector<double, TagT>& v2 =
      cast<TrilinosVector<double, TagT>>(targetImpl, targetBackend());

  alien_info([&] {
    cout() << "Converting SimpleCSRVector: " << &v << " to TrilinosVector " << &v2;
  });

  /*{
    std::cout<<  " nrows : "<<v.scalarizedLocalSize()<<std::endl ;
    for(int i=0;i<v.scalarizedLocalSize();++i)
      std::cout<<" CSR V["<<i<<"]"<<v.values()[i]<<std::endl ;
  }*/
  ConstArrayView<Real> values = v.values();
  v2.setValues(v.scalarizedLocalSize(), values.data());
}

/*---------------------------------------------------------------------------*/
#ifdef KOKKOS_ENABLE_SERIAL
template class SimpleCSR_to_Trilinos_VectorConverter<BackEnd::tag::tpetraserial>;
typedef SimpleCSR_to_Trilinos_VectorConverter<BackEnd::tag::tpetraserial>
    SimpleCSR_to_Trilinos_VectorConverterSerial;
REGISTER_VECTOR_CONVERTER(SimpleCSR_to_Trilinos_VectorConverterSerial);
#endif

#ifdef KOKKOS_ENABLE_OPENMP
template class SimpleCSR_to_Trilinos_VectorConverter<BackEnd::tag::tpetraomp>;
typedef SimpleCSR_to_Trilinos_VectorConverter<BackEnd::tag::tpetraomp>
    SimpleCSR_to_Trilinos_VectorConverterOMP;
REGISTER_VECTOR_CONVERTER(SimpleCSR_to_Trilinos_VectorConverterOMP);
#endif

#ifdef KOKKOS_ENABLE_THREADS
template class SimpleCSR_to_Trilinos_VectorConverter<BackEnd::tag::tpetrapth>;
typedef SimpleCSR_to_Trilinos_VectorConverter<BackEnd::tag::tpetrapth>
    SimpleCSR_to_Trilinos_VectorConverterPTH;
REGISTER_VECTOR_CONVERTER(SimpleCSR_to_Trilinos_VectorConverterPTH);
#endif

#ifdef KOKKOS_ENABLE_CUDA
template class SimpleCSR_to_Trilinos_VectorConverter<BackEnd::tag::tpetracuda>;
typedef SimpleCSR_to_Trilinos_VectorConverter<BackEnd::tag::tpetracuda>
    SimpleCSR_to_Trilinos_VectorConverterCUDA;
REGISTER_VECTOR_CONVERTER(SimpleCSR_to_Trilinos_VectorConverterCUDA);
#endif
