#pragma once

#include <Alien/Core/Impl/IVectorImpl.h>

#include <HYPRE_IJ_mv.h>

namespace Alien::Hypre {

    class VectorInternal;

    class Vector : public IVectorImpl {
    public:

        Vector(const MultiVectorImpl *multi_impl);

        virtual ~Vector();

    public:

        void setProfile(int ilower, int iupper);

        void setValues(Arccore::ConstArrayView<double> values);

        void getValues(Arccore::ArrayView<double> values) const;

        void assemble();

        HYPRE_IJVector internal() { return m_hypre; }

        HYPRE_IJVector internal() const { return m_hypre; }

    private:

        HYPRE_IJVector m_hypre;
        MPI_Comm m_comm;

        Arccore::UniqueArray<Arccore::Integer> m_rows;
    };

}
