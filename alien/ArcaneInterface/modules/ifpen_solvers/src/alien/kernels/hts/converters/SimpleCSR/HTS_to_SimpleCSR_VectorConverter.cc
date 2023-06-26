#include <alien/core/backend/IVectorConverter.h>
#include <alien/core/backend/VectorConverterRegisterer.h>

#include <iostream>
#include <alien/kernels/hts/data_structure/HTSVector.h>

#include <alien/kernels/hts/HTSBackEnd.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>
using namespace Alien;

/*---------------------------------------------------------------------------*/

class HTS_to_SimpleCSR_VectorConverter : public IVectorConverter
{
 public:
  HTS_to_SimpleCSR_VectorConverter();
  virtual ~HTS_to_SimpleCSR_VectorConverter() {}
 public:
  Alien::BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::hts>::name();
  }
  Alien::BackEndId targetBackend() const
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }
  void convert(const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const;
};

/*---------------------------------------------------------------------------*/

HTS_to_SimpleCSR_VectorConverter::HTS_to_SimpleCSR_VectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
HTS_to_SimpleCSR_VectorConverter::convert(
    const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const HTSVector<double, true>& v =
      cast<HTSVector<double, true>>(sourceImpl, sourceBackend());
  SimpleCSRVector<double>& v2 =
      cast<SimpleCSRVector<double>>(targetImpl, targetBackend());

  alien_debug(
      [&] { cout() << "Converting HTSVector: " << &v << " to SimpleCSRVector " << &v2; });

  v.getValues(v2.values().size(), v2.getDataPtr());
}

/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(HTS_to_SimpleCSR_VectorConverter);
