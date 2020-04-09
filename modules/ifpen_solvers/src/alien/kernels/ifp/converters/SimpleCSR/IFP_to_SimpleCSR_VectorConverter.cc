#include <alien/core/backend/IVectorConverter.h>
#include <alien/core/backend/VectorConverterRegisterer.h>

#include <iostream>
#include <alien/kernels/ifp/data_structure/IFPVector.h>

#include <alien/kernels/ifp/IFPSolverBackEnd.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>

using namespace Alien;

/*---------------------------------------------------------------------------*/

class IFP_to_SimpleCSR_VectorConverter : public IVectorConverter
{
 public:
  IFP_to_SimpleCSR_VectorConverter();
  virtual ~IFP_to_SimpleCSR_VectorConverter() {}
 public:
  Alien::BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::ifpsolver>::name();
  }
  Alien::BackEndId targetBackend() const
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }
  void convert(const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const;
};

/*---------------------------------------------------------------------------*/

IFP_to_SimpleCSR_VectorConverter::IFP_to_SimpleCSR_VectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
IFP_to_SimpleCSR_VectorConverter::convert(
    const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const IFPVector& v = cast<IFPVector>(sourceImpl, sourceBackend());
  SimpleCSRVector<double>& v2 =
      cast<SimpleCSRVector<double>>(targetImpl, targetBackend());

  alien_debug(
      [&] { cout() << "Converting IFPVector: " << &v << " to SimpleCSRVector " << &v2; });

  ArrayView<Real> values = v2.values();

  v.getValues(values.size(), v2.getDataPtr());
}

/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(IFP_to_SimpleCSR_VectorConverter);
