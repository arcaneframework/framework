#include <ALIEN/Core/Backend/IVectorConverter.h>
#include <ALIEN/Core/Backend/VectorConverterRegisterer.h>

#include <iostream>
#include <ALIEN/Kernels/PETSc/DataStructure/PETScVector.h>

#include <ALIEN/Kernels/PETSc/PETScBackEnd.h>
#include <ALIEN/Kernels/SimpleCSR/DataStructure/SimpleCSRVector.h>
#include <ALIEN/Kernels/SimpleCSR/SimpleCSRBackEnd.h>

using namespace Alien;

/*---------------------------------------------------------------------------*/

class PETSc_to_SimpleCSR_VectorConverter : public IVectorConverter
{
 public:
  PETSc_to_SimpleCSR_VectorConverter();
  virtual ~PETSc_to_SimpleCSR_VectorConverter() { }
public:
  Alien::BackEndId sourceBackend() const { return AlgebraTraits<BackEnd::tag::petsc>::name(); }
  Alien::BackEndId targetBackend() const { return AlgebraTraits<BackEnd::tag::simplecsr>::name(); }
  void convert(const IVectorImpl * sourceImpl, IVectorImpl * targetImpl) const;
};

/*---------------------------------------------------------------------------*/

PETSc_to_SimpleCSR_VectorConverter::PETSc_to_SimpleCSR_VectorConverter()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
PETSc_to_SimpleCSR_VectorConverter::convert(
    const IVectorImpl* sourceImpl, IVectorImpl* targetImpl) const
{
  const PETScVector& v = cast<PETScVector>(sourceImpl, sourceBackend());
  SimpleCSRVector<double>& v2 =
      cast<SimpleCSRVector<double>>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting PETScVector: " << &v << " to SimpleCSRVector " << &v2;
  });

  Arccore::ArrayView<Arccore::Real> values = v2.values();
  v.getValues(values.size(), values.unguardedBasePointer());
}

/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(PETSc_to_SimpleCSR_VectorConverter);
