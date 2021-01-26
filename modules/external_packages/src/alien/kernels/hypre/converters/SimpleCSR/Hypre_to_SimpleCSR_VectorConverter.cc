#include <alien/core/backend/IVectorConverter.h>
#include <alien/core/backend/VectorConverterRegisterer.h>

#include <iostream>
#include <alien/kernels/hypre/data_structure/HypreVector.h>

#include <alien/kernels/hypre/HypreBackEnd.h>
#include <alien/kernels/simple_csr/CSRStructInfo.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>

using namespace Alien;
using namespace Alien::SimpleCSRInternal;

/*---------------------------------------------------------------------------*/

class Hypre_to_SimpleCSR_VectorConverter : public IVectorConverter
{
 public:
  Hypre_to_SimpleCSR_VectorConverter();
  virtual ~Hypre_to_SimpleCSR_VectorConverter() {}
 public:
  Alien::BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::hypre>::name();
  }
  Alien::BackEndId targetBackend() const
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }
  void convert(const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const;
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
  SimpleCSRVector<Arccore::Real>& v2 =
      cast<SimpleCSRVector<Arccore::Real>>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting HypreVector: " << &v << " to SimpleCSRVector " << &v2;
  });
  Arccore::ArrayView<Arccore::Real> values = v2.values();
  v.getValues(values.size(), values.unguardedBasePointer());
}

/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(Hypre_to_SimpleCSR_VectorConverter);
