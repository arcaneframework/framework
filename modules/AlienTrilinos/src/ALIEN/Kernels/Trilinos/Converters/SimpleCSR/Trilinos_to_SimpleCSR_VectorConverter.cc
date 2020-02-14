#include <ALIEN/Core/Backend/IVectorConverter.h>
#include <ALIEN/Core/Backend/VectorConverterRegisterer.h>

#include <iostream>
#include <ALIEN/Kernels/Trilinos/DataStructure/TrilinosVector.h>

#include <ALIEN/Kernels/Trilinos/TrilinosBackEnd.h>
#include <ALIEN/Kernels/SimpleCSR/DataStructure/SimpleCSRVector.h>
#include <ALIEN/Kernels/SimpleCSR/SimpleCSRBackEnd.h>
using namespace Alien;

/*---------------------------------------------------------------------------*/
template<typename TagT>
class Trilinos_to_SimpleCSR_VectorConverter : public IVectorConverter
{
 public:
  Trilinos_to_SimpleCSR_VectorConverter();
  virtual ~Trilinos_to_SimpleCSR_VectorConverter() { }
public:
  Alien::BackEndId sourceBackend() const { return AlgebraTraits<TagT>::name(); }
  Alien::BackEndId targetBackend() const { return AlgebraTraits<BackEnd::tag::simplecsr>::name(); }
  void convert(const IVectorImpl * sourceImpl, IVectorImpl * targetImpl) const;
};

/*---------------------------------------------------------------------------*/
template<typename TagT>
Trilinos_to_SimpleCSR_VectorConverter<TagT>::Trilinos_to_SimpleCSR_VectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/
template<typename TagT>
void
Trilinos_to_SimpleCSR_VectorConverter<TagT>::convert(
    const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const TrilinosVector<double,TagT> & v = cast<TrilinosVector<double,TagT>>(sourceImpl, sourceBackend());
  SimpleCSRVector<double> & v2 = cast<SimpleCSRVector<double> >(targetImpl, targetBackend());
  
  alien_debug([&] {
    cout() << "Converting TrilinosVector: " << &v << " to SimpleCSRVector " << &v2;
  });

  v.getValues(v2.values().size(), v2.getDataPtr());
}

/*---------------------------------------------------------------------------*/
template class Trilinos_to_SimpleCSR_VectorConverter<Alien::BackEnd::tag::tpetraserial> ;
typedef Trilinos_to_SimpleCSR_VectorConverter<Alien::BackEnd::tag::tpetraserial> Trilinos_to_SimpleCSR_VectorConverterSerial ;
REGISTER_VECTOR_CONVERTER(Trilinos_to_SimpleCSR_VectorConverterSerial);
#ifdef KOKKOS_ENABLE_OPENMP
template class Trilinos_to_SimpleCSR_VectorConverter<Alien::BackEnd::tag::tpetraomp> ;
typedef Trilinos_to_SimpleCSR_VectorConverter<Alien::BackEnd::tag::tpetraomp> Trilinos_to_SimpleCSR_VectorConverterOMP ;
REGISTER_VECTOR_CONVERTER(Trilinos_to_SimpleCSR_VectorConverterOMP);
#endif
#ifdef KOKKOS_ENABLE_THREADS
template class Trilinos_to_SimpleCSR_VectorConverter<Alien::BackEnd::tag::tpetrapth> ;
typedef Trilinos_to_SimpleCSR_VectorConverter<Alien::BackEnd::tag::tpetrapth> Trilinos_to_SimpleCSR_VectorConverterPTH ;
REGISTER_VECTOR_CONVERTER(Trilinos_to_SimpleCSR_VectorConverterPTH);
#endif
#ifdef KOKKOS_ENABLE_CUDA
template class Trilinos_to_SimpleCSR_VectorConverter<Alien::BackEnd::tag::tpetracuda> ;
typedef Trilinos_to_SimpleCSR_VectorConverter<Alien::BackEnd::tag::tpetracuda> Trilinos_to_SimpleCSR_VectorConverterCUDA ;
REGISTER_VECTOR_CONVERTER(Trilinos_to_SimpleCSR_VectorConverterCUDA);
#endif
