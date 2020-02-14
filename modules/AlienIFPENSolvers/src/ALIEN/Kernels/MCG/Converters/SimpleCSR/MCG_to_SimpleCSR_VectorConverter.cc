#include "ALIEN/Core/Backend/IVectorConverter.h"
#include "ALIEN/Core/Backend/VectorConverterRegisterer.h"

#include <iostream>

#include <ALIEN/Kernels/SimpleCSR/DataStructure/SimpleCSRVector.h>
#include <ALIEN/Kernels/SimpleCSR/SimpleCSRBackEnd.h>
#include <ALIEN/Kernels/MCG/DataStructure/MCGVector.h>

#include <ALIEN/Kernels/MCG/MCGBackEnd.h>
using namespace Alien;

/*---------------------------------------------------------------------------*/

class MCG_to_SimpleCSR_VectorConverter : public IVectorConverter
{
public:
  MCG_to_SimpleCSR_VectorConverter();
  virtual ~MCG_to_SimpleCSR_VectorConverter() { }
public:
  BackEndId sourceBackend() const { return AlgebraTraits<BackEnd::tag::mcgsolver>::name(); }
  BackEndId targetBackend() const { return AlgebraTraits<BackEnd::tag::simplecsr>::name(); }
  void convert(const IVectorImpl * sourceImpl, IVectorImpl * targetImpl) const;
};

/*---------------------------------------------------------------------------*/

MCG_to_SimpleCSR_VectorConverter::
MCG_to_SimpleCSR_VectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
MCG_to_SimpleCSR_VectorConverter::
convert(const IVectorImpl * sourceImpl, IVectorImpl * targetImpl) const
{
  const MCGVector & v = cast<MCGVector>(sourceImpl, sourceBackend());
  SimpleCSRVector<double> & v2 = cast<SimpleCSRVector<double> >(targetImpl, targetBackend());
  
  alien_debug([&] {
    cout() << "Converting MCGVector: " << &v << " to SimpleCSRVector " << &v2;
  });

  //v.getValues(space.localSize(), v2.getDataPtr());
  ArrayView<Real> values = v2.getValuesView();
  v.getValues(values.size(), values.unguardedBasePointer());
}

/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(MCG_to_SimpleCSR_VectorConverter);
