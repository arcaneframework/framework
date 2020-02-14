#include <ALIEN/Core/Backend/IVectorConverter.h>
#include <ALIEN/Core/Backend/VectorConverterRegisterer.h>

#include <iostream>
#include <ALIEN/Kernels/MTL/DataStructure/MTLVector.h>

#include <ALIEN/Kernels/MTL/MTLBackEnd.h>
#include <ALIEN/Kernels/SimpleCSR/DataStructure/SimpleCSRVector.h>
#include <ALIEN/Kernels/SimpleCSR/SimpleCSRBackEnd.h>
using namespace Alien;

/*---------------------------------------------------------------------------*/

class SimpleCSR_to_MTL_VectorConverter : public IVectorConverter
{
 public:
  SimpleCSR_to_MTL_VectorConverter();
  virtual ~SimpleCSR_to_MTL_VectorConverter() { }
public:
  Alien::BackEndId sourceBackend() const { return AlgebraTraits<BackEnd::tag::simplecsr>::name(); }
  Alien::BackEndId targetBackend() const { return AlgebraTraits<BackEnd::tag::mtl>::name(); }
  void convert(const IVectorImpl * sourceImpl, IVectorImpl * targetImpl) const;
};

/*---------------------------------------------------------------------------*/

SimpleCSR_to_MTL_VectorConverter::SimpleCSR_to_MTL_VectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
SimpleCSR_to_MTL_VectorConverter::convert(
    const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const SimpleCSRVector<double>& v =
      cast<SimpleCSRVector<double>>(sourceImpl, sourceBackend());
  MTLVector& v2 = cast<MTLVector>(targetImpl, targetBackend());

  alien_debug(
      [&] { cout() << "Converting SimpleCSRVector: " << &v << " to MTLVector " << &v2; });

  Arccore::ConstArrayView<Arccore::Real> values = v.values();
  v2.setValues(v.scalarizedLocalSize(), values.data());
}

/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(SimpleCSR_to_MTL_VectorConverter);
