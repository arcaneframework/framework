#include "hypre_vector.h"

#include <ALIEN/hypre/backend.h>
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

void SimpleCSR_to_Hypre_VectorConverter::convert(const Alien::IVectorImpl *sourceImpl,
                                                 Alien::IVectorImpl *targetImpl) const {
    const auto &v = cast<Alien::SimpleCSRVector<Arccore::Real>>(sourceImpl, sourceBackend());
    auto &v2 = cast<Alien::Hypre::Vector>(targetImpl, targetBackend());

    alien_debug([&] {
        cout() << "Converting Alien::SimpleCSRVector: " << &v << " to Hypre::Vector " << &v2;
    });

    auto block_size = 1;
    const auto *block = v2.block();
    if (v2.block())
        block_size *= block->size();
    else if (v2.vblock())
        throw Arccore::FatalErrorException(A_FUNCINFO, "Not implemented yet");

    const auto localOffset = v2.distribution().offset();
    const auto localSize = v2.distribution().localSize();
    const auto ilower = localOffset * block_size;
    const auto iupper = ilower + localSize * block_size - 1;

    alien_debug([&] {
        cout() << "Vector range : "
               << "[" << ilower << ":" << iupper << "]";
    });

    auto values = v.values();

    v2.setValues(values);

    v2.assemble();
}

REGISTER_VECTOR_CONVERTER(SimpleCSR_to_Hypre_VectorConverter);
