#include "alien/core/backend/IVectorConverter.h"
#include "alien/core/backend/VectorConverterRegisterer.h"

#include <iostream>

#include <alien/kernels/simple_csr/data_structure/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>
#include <alien/Kernels/MCG/DataStructure/MCGVector.h>

#include <alien/Kernels/MCG/MCGBackEnd.h>
using namespace Alien;

/*---------------------------------------------------------------------------*/

class SimpleCSR_to_MCG_VectorConverter : public IVectorConverter
//: public ICompositeVectorConverter<2>
{
public:
  SimpleCSR_to_MCG_VectorConverter();
  virtual ~SimpleCSR_to_MCG_VectorConverter() { }
public:
  BackEndId sourceBackend() const { return AlgebraTraits<BackEnd::tag::simplecsr>::name(); }
  BackEndId targetBackend() const { return AlgebraTraits<BackEnd::tag::mcgsolver>::name(); }
  void convert(const IVectorImpl * sourceImpl, IVectorImpl * targetImpl) const;
  //void convert(const IVectorImpl * sourceImpl, IVectorImpl * targetImpl,int i) const;
};

/*---------------------------------------------------------------------------*/

SimpleCSR_to_MCG_VectorConverter::
SimpleCSR_to_MCG_VectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
SimpleCSR_to_MCG_VectorConverter::
convert(const IVectorImpl * sourceImpl, IVectorImpl * targetImpl) const
{
  const SimpleCSRVector<double> & v = cast<SimpleCSRVector<double> >(sourceImpl, sourceBackend());
  MCGVector & v2 = cast<MCGVector>(targetImpl, targetBackend());
  
  alien_debug([&] {
    cout() << "Converting SimpleCSRVector: " << &v << " to MCGVector " << &v2;
  });

  ConstArrayView<Real> values = v.values();
  //v2.setValues(sourceImpl->distribution().localSize(),dataPtr(values));
  v2.setValues(values.size(),dataPtr(values));
}

/*
void
SimpleCSR_to_MCG_VectorConverter::
convert(const IVectorImpl * sourceImpl, IVectorImpl * targetImpl,int i) const
{

  const SimpleCSRVector<double> & v = cast<SimpleCSRVector<double> >(sourceImpl, sourceBackend());
  MCGVector & v2 = cast<MCGVector>(targetImpl, targetBackend());

  const Space & space = v.space();
  ConstArrayView<Real> values = v.values();
  if(i==0)
    v2.setValues(space.localSize(),dataPtr(values));
  else
    v2.setExtraEqValues(v.space().localSize(),dataPtr(values)) ;
}
*/

/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(SimpleCSR_to_MCG_VectorConverter);
