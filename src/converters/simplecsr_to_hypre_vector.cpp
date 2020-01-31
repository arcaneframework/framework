#include "hypre_backend.h"
#include "hypre_vector.h"

#include <ALIEN/Core/Backend/IVectorConverter.h>
#include <ALIEN/Core/Backend/VectorConverterRegisterer.h>
#include <ALIEN/Kernels/SimpleCSR/DataStructure/SimpleCSRVector.h>
#include <ALIEN/Kernels/SimpleCSR/SimpleCSRBackEnd.h>

class SimpleCSR_to_Hypre_VectorConverter : public Alien::IVectorConverter {
public:
  SimpleCSR_to_Hypre_VectorConverter() {}

  virtual ~SimpleCSR_to_Hypre_VectorConverter() {}

public:
  Alien::BackEndId sourceBackend() const { return Alien::AlgebraTraits<Alien::BackEnd::tag::simplecsr>::name(); }

  Alien::BackEndId targetBackend() const { return Alien::AlgebraTraits<Alien::BackEnd::tag::hypre>::name(); }

  void convert(const Alien::IVectorImpl *sourceImpl, Alien::IVectorImpl *targetImpl) const;
};

void
SimpleCSR_to_Hypre_VectorConverter::convert(
        const Alien::IVectorImpl *sourceImpl, Alien::IVectorImpl *targetImpl) const {
  const auto &v = cast<Alien::SimpleCSRVector<Arccore::Real>>(sourceImpl, sourceBackend());
  auto &v2 = cast<Alien::Hypre::Vector>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting Alien::SimpleCSRVector: " << &v << " to Hypre::Vector " << &v2;
  });
  auto values = v.values();
  v2.setValues(values.size(), values.unguardedBasePointer());
}

REGISTER_VECTOR_CONVERTER(SimpleCSR_to_Hypre_VectorConverter);
