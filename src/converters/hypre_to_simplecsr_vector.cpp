#include "hypre_vector.h"

#include <alien/core/backend/IVectorConverter.h>
#include <alien/core/backend/VectorConverterRegisterer.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>

#include <alien/hypre/backend.h>

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
