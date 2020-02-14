#include <ALIEN/Core/Backend/IVectorConverter.h>
#include <ALIEN/Core/Backend/VectorConverterRegisterer.h>

#include <iostream>
#include <ALIEN/Kernels/MTL/DataStructure/MTLVector.h>

#include <ALIEN/Kernels/MTL/MTLBackEnd.h>
#include <ALIEN/Kernels/SimpleCSR/DataStructure/SimpleCSRVector.h>
#include <ALIEN/Kernels/SimpleCSR/SimpleCSRBackEnd.h>
using namespace Alien;

/*---------------------------------------------------------------------------*/

class MTL_to_SimpleCSR_VectorConverter : public IVectorConverter
{
 public:
  MTL_to_SimpleCSR_VectorConverter();
  virtual ~MTL_to_SimpleCSR_VectorConverter() { }
public:
  Alien::BackEndId sourceBackend() const { return AlgebraTraits<BackEnd::tag::mtl>::name(); }
  Alien::BackEndId targetBackend() const { return AlgebraTraits<BackEnd::tag::simplecsr>::name(); }
  void convert(const IVectorImpl * sourceImpl, IVectorImpl * targetImpl) const;
};

/*---------------------------------------------------------------------------*/

MTL_to_SimpleCSR_VectorConverter::MTL_to_SimpleCSR_VectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
MTL_to_SimpleCSR_VectorConverter::convert(
    const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const MTLVector & v = cast<MTLVector>(sourceImpl, sourceBackend());
  SimpleCSRVector<double> & v2 = cast<SimpleCSRVector<double> >(targetImpl, targetBackend());
  
  alien_debug([&] {
    cout() << "Converting MTLVector: " << &v << " to SimpleCSRVector " << &v2;
  });

  v.getValues(v2.values().size(), v2.getDataPtr());
}

/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(MTL_to_SimpleCSR_VectorConverter);
