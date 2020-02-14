#include <ALIEN/Core/Backend/IVectorConverter.h>
#include <ALIEN/Core/Backend/VectorConverterRegisterer.h>

#include <iostream>
#include <ALIEN/Kernels/Hypre/DataStructure/HypreVector.h>

#include <ALIEN/Kernels/Hypre/HypreBackEnd.h>
#include <ALIEN/Kernels/SimpleCSR/DataStructure/CSRStructInfo.h>
#include <ALIEN/Kernels/SimpleCSR/DataStructure/SimpleCSRVector.h>
#include <ALIEN/Kernels/SimpleCSR/SimpleCSRBackEnd.h>

using namespace Alien;
using namespace Alien::SimpleCSRInternal;

/*---------------------------------------------------------------------------*/

class Hypre_to_SimpleCSR_VectorConverter : public IVectorConverter
{
 public:
  Hypre_to_SimpleCSR_VectorConverter();
  virtual ~Hypre_to_SimpleCSR_VectorConverter() { }
public:
  Alien::BackEndId sourceBackend() const { return AlgebraTraits<BackEnd::tag::hypre>::name(); }
  Alien::BackEndId targetBackend() const { return AlgebraTraits<BackEnd::tag::simplecsr>::name(); }
  void convert(const IVectorImpl * sourceImpl, IVectorImpl * targetImpl) const;
};

/*---------------------------------------------------------------------------*/

Hypre_to_SimpleCSR_VectorConverter::Hypre_to_SimpleCSR_VectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
Hypre_to_SimpleCSR_VectorConverter::convert(
    const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const auto& v = cast<HypreVector>(sourceImpl, sourceBackend());
  SimpleCSRVector<Arccore::Real> & v2 = cast<SimpleCSRVector<Arccore::Real> >(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting HypreVector: " << &v << " to SimpleCSRVector " << &v2;
  });
  Arccore::ArrayView<Arccore::Real> values = v2.values();
  v.getValues(values.size(), values.unguardedBasePointer());
}

/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(Hypre_to_SimpleCSR_VectorConverter);
