#include <alien/core/backend/IVectorConverter.h>
#include <alien/core/backend/VectorConverterRegisterer.h>

#include <iostream>
#include <alien/kernels/mtl/data_structure/MTLVector.h>

#include <alien/kernels/mtl/MTLBackEnd.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>
using namespace Alien;

/*---------------------------------------------------------------------------*/

class SimpleCSR_to_MTL_VectorConverter : public IVectorConverter
{
 public:
  SimpleCSR_to_MTL_VectorConverter();
  virtual ~SimpleCSR_to_MTL_VectorConverter() {}
 public:
  Alien::BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }
  Alien::BackEndId targetBackend() const
  {
    return AlgebraTraits<BackEnd::tag::mtl>::name();
  }
  void convert(const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const;
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
