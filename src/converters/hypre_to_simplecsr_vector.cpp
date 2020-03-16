#include "hypre_vector.h"

#include <Alien/hypre/backend.h>
#include <Alien/Core/Backend/IVectorConverter.h>
#include <Alien/Core/Backend/VectorConverterRegisterer.h>
#include <Alien/Kernels/SimpleCSR/DataStructure/SimpleCSRVector.h>
#include <Alien/Kernels/SimpleCSR/SimpleCSRBackEnd.h>

class Hypre_to_SimpleCSR_VectorConverter : public Alien::IVectorConverter {
public:
  Hypre_to_SimpleCSR_VectorConverter() {}

  virtual ~Hypre_to_SimpleCSR_VectorConverter() {}

public:
  Alien::BackEndId sourceBackend() const { return Alien::AlgebraTraits<Alien::BackEnd::tag::hypre>::name(); }

  Alien::BackEndId targetBackend() const { return Alien::AlgebraTraits<Alien::BackEnd::tag::simplecsr>::name(); }

  void convert(const Alien::IVectorImpl *sourceImpl, Alien::IVectorImpl *targetImpl) const;
};

void
Hypre_to_SimpleCSR_VectorConverter::convert(
        const Alien::IVectorImpl *sourceImpl, Alien::IVectorImpl *targetImpl) const {
    const auto &v = cast<Alien::Hypre::Vector>(sourceImpl, sourceBackend());
    auto &v2 = cast<Alien::SimpleCSRVector<Arccore::Real> >(targetImpl, targetBackend());

    alien_debug([&] {
        cout() << "Converting Hypre::Vector: " << &v << " to Alien::SimpleCSRVector " << &v2;
    });
    auto values = v2.values();

    v.getValues(values);
}

REGISTER_VECTOR_CONVERTER(Hypre_to_SimpleCSR_VectorConverter);
