#include <alien/core/backend/IVectorConverter.h>
#include <alien/core/backend/VectorConverterRegisterer.h>

#include <iostream>
#include <alien/kernels/ifp/data_structure/IFPVector.h>

#include <alien/kernels/ifp/IFPSolverBackEnd.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>
using namespace Alien;

/*---------------------------------------------------------------------------*/

class SimpleCSR_to_IFP_VectorConverter : public IVectorConverter
{
 public:
  SimpleCSR_to_IFP_VectorConverter();
  virtual ~SimpleCSR_to_IFP_VectorConverter() {}
 public:
  Alien::BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }
  Alien::BackEndId targetBackend() const
  {
    return AlgebraTraits<BackEnd::tag::ifpsolver>::name();
  }
  void convert(const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const;
};

/*---------------------------------------------------------------------------*/

SimpleCSR_to_IFP_VectorConverter::SimpleCSR_to_IFP_VectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
SimpleCSR_to_IFP_VectorConverter::convert(
    const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  if (IFPSolverInternal::VectorInternal::hasRepresentationSwitch())
    return;

  const SimpleCSRVector<double>& v =
      cast<SimpleCSRVector<double>>(sourceImpl, sourceBackend());
  IFPVector& v2 = cast<IFPVector>(targetImpl, targetBackend());

  alien_debug(
      [&] { cout() << "Converting SimpleCSRVector: " << &v << " to IFPVector " << &v2; });

  ConstArrayView<Real> values = v.values();
  v2.setValues(values.unguardedBasePointer());
}

/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(SimpleCSR_to_IFP_VectorConverter);
