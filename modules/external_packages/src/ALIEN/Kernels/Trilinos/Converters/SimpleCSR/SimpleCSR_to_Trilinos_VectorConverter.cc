#include <alien/core/backend/IVectorConverter.h>
#include <alien/core/backend/VectorConverterRegisterer.h>

#include <iostream>
#include <ALIEN/Kernels/Trilinos/DataStructure/TrilinosVector.h>

#include <ALIEN/Kernels/Trilinos/TrilinosBackEnd.h>
#include <alien/kernels/simple_csr/data_structure/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>
using namespace Alien;

/*---------------------------------------------------------------------------*/
template<typename TagT>
class SimpleCSR_to_Trilinos_VectorConverter : public IVectorConverter
{
 public:
  SimpleCSR_to_Trilinos_VectorConverter();
  virtual ~SimpleCSR_to_Trilinos_VectorConverter() { }
public:
  Alien::BackEndId sourceBackend() const { return AlgebraTraits<BackEnd::tag::simplecsr>::name(); }
  Alien::BackEndId targetBackend() const { return AlgebraTraits<TagT>::name(); }
  void convert(const IVectorImpl * sourceImpl, IVectorImpl * targetImpl) const;
};

/*---------------------------------------------------------------------------*/
template<typename TagT>
SimpleCSR_to_Trilinos_VectorConverter<TagT>::SimpleCSR_to_Trilinos_VectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/
template<typename TagT>
void
SimpleCSR_to_Trilinos_VectorConverter<TagT>::convert(
    const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const SimpleCSRVector<double>& v =
      cast<SimpleCSRVector<double>>(sourceImpl, sourceBackend());
  TrilinosVector<double,TagT>& v2 = cast<TrilinosVector<double,TagT>>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting SimpleCSRVector: " << &v << " to TrilinosVector " << &v2;
  });

  ConstArrayView<Real> values = v.values();
  v2.setValues(v.scalarizedLocalSize(), dataPtr(values));
}

/*---------------------------------------------------------------------------*/
template class SimpleCSR_to_Trilinos_VectorConverter<BackEnd::tag::tpetraserial> ;
typedef SimpleCSR_to_Trilinos_VectorConverter<BackEnd::tag::tpetraserial> SimpleCSR_to_Trilinos_VectorConverterSerial ;
REGISTER_VECTOR_CONVERTER(SimpleCSR_to_Trilinos_VectorConverterSerial);

#ifdef KOKKOS_ENABLE_OPENMP
template class SimpleCSR_to_Trilinos_VectorConverter<BackEnd::tag::tpetraomp> ;
typedef SimpleCSR_to_Trilinos_VectorConverter<BackEnd::tag::tpetraomp> SimpleCSR_to_Trilinos_VectorConverterOMP ;
REGISTER_VECTOR_CONVERTER(SimpleCSR_to_Trilinos_VectorConverterOMP);
#endif

#ifdef KOKKOS_ENABLE_THREADS
template class SimpleCSR_to_Trilinos_VectorConverter<BackEnd::tag::tpetrapth> ;
typedef SimpleCSR_to_Trilinos_VectorConverter<BackEnd::tag::tpetrapth> SimpleCSR_to_Trilinos_VectorConverterPTH ;
REGISTER_VECTOR_CONVERTER(SimpleCSR_to_Trilinos_VectorConverterPTH);
#endif
