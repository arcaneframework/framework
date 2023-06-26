#include <alien/core/backend/IVectorConverter.h>
#include <alien/core/backend/VectorConverterRegisterer.h>

#include <iostream>
#include <alien/kernels/hts/data_structure/HTSVector.h>

#include <alien/kernels/hts/HTSBackEnd.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>
using namespace Alien;

/*---------------------------------------------------------------------------*/

class SimpleCSR_to_HTS_VectorConverter : public IVectorConverter
{
 public:
  SimpleCSR_to_HTS_VectorConverter();
  virtual ~SimpleCSR_to_HTS_VectorConverter() {}
 public:
  Alien::BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }
  Alien::BackEndId targetBackend() const
  {
    return AlgebraTraits<BackEnd::tag::hts>::name();
  }
  void convert(const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const;
};

/*---------------------------------------------------------------------------*/

SimpleCSR_to_HTS_VectorConverter::SimpleCSR_to_HTS_VectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
SimpleCSR_to_HTS_VectorConverter::convert(
    const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const SimpleCSRVector<double>& v =
      cast<SimpleCSRVector<double>>(sourceImpl, sourceBackend());
  HTSVector<double, true>& v2 =
      cast<HTSVector<double, true>>(targetImpl, targetBackend());

  alien_debug(
      [&] { cout() << "Converting SimpleCSRVector: " << &v << " to HTSVector " << &v2; });

  ConstArrayView<Real> values = v.values();
  v2.setValues(v.scalarizedLocalSize(), values.data());
}

/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(SimpleCSR_to_HTS_VectorConverter);
