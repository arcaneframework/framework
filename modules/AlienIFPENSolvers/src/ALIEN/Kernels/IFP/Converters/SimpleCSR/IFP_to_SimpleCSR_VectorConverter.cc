#include <ALIEN/Core/Backend/IVectorConverter.h>
#include <ALIEN/Core/Backend/VectorConverterRegisterer.h>

#include <iostream>
#include <ALIEN/Kernels/IFP/DataStructure/IFPVector.h>

#include <ALIEN/Kernels/IFP/IFPSolverBackEnd.h>
#include <ALIEN/Kernels/SimpleCSR/DataStructure/SimpleCSRVector.h>
#include <ALIEN/Kernels/SimpleCSR/SimpleCSRBackEnd.h>


using namespace Alien;

/*---------------------------------------------------------------------------*/

class IFP_to_SimpleCSR_VectorConverter : public IVectorConverter 
{
public:
  IFP_to_SimpleCSR_VectorConverter();
  virtual ~IFP_to_SimpleCSR_VectorConverter() { }
public:
  Alien::BackEndId sourceBackend() const { return AlgebraTraits<BackEnd::tag::ifpsolver>::name(); }
  Alien::BackEndId targetBackend() const { return AlgebraTraits<BackEnd::tag::simplecsr>::name(); }
  void convert(const IVectorImpl * sourceImpl, IVectorImpl * targetImpl) const;
};

/*---------------------------------------------------------------------------*/

IFP_to_SimpleCSR_VectorConverter::
IFP_to_SimpleCSR_VectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
IFP_to_SimpleCSR_VectorConverter::
convert(const IVectorImpl * sourceImpl, IVectorImpl * targetImpl) const
{
  const IFPVector & v = cast<IFPVector>(sourceImpl, sourceBackend());
  SimpleCSRVector<double> & v2 = cast<SimpleCSRVector<double> >(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting IFPVector: " << &v << " to SimpleCSRVector " << &v2;
    });

  Arccore::ArrayView<Arccore::Real> values = v2.getValuesView();

  v.getValues(values.size(), v2.getDataPtr());
}

/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(IFP_to_SimpleCSR_VectorConverter);
