#include <ALIEN/Core/Backend/IVectorConverter.h>
#include <ALIEN/Core/Backend/VectorConverterRegisterer.h>

#include <iostream>
#include <ALIEN/Kernels/HTS/DataStructure/HTSVector.h>

#include <ALIEN/Kernels/HTS/HTSBackEnd.h>
#include <ALIEN/Kernels/SimpleCSR/DataStructure/SimpleCSRVector.h>
#include <ALIEN/Kernels/SimpleCSR/SimpleCSRBackEnd.h>
using namespace Alien;

/*---------------------------------------------------------------------------*/

class HTS_to_SimpleCSR_VectorConverter : public IVectorConverter
{
 public:
  HTS_to_SimpleCSR_VectorConverter();
  virtual ~HTS_to_SimpleCSR_VectorConverter() { }
public:
  Alien::BackEndId sourceBackend() const { return AlgebraTraits<BackEnd::tag::hts>::name(); }
  Alien::BackEndId targetBackend() const { return AlgebraTraits<BackEnd::tag::simplecsr>::name(); }
  void convert(const IVectorImpl * sourceImpl, IVectorImpl * targetImpl) const;
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
  const HTSVector<double,true> & v = cast<HTSVector<double,true>>(sourceImpl, sourceBackend());
  SimpleCSRVector<double> & v2 = cast<SimpleCSRVector<double> >(targetImpl, targetBackend());
  
  alien_debug([&] {
    cout() << "Converting HTSVector: " << &v << " to SimpleCSRVector " << &v2;
  });

  v.getValues(v2.values().size(), v2.getDataPtr());
}

/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(HTS_to_SimpleCSR_VectorConverter);
