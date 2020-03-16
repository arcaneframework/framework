#include "hypre_vector.h"

#include <Alien/hypre/backend.h>

#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>

#include <HYPRE.h>

namespace Alien::Hypre {

    Vector::Vector(const MultiVectorImpl *multi_impl)
            : IVectorImpl(multi_impl, AlgebraTraits<BackEnd::tag::hypre>::name()), m_hypre(nullptr) {
        auto block_size = 1;
        const auto *block = this->block();
        if (block)
            block_size *= block->size();
        else if (this->vblock())
            throw Arccore::FatalErrorException(A_FUNCINFO, "Not implemented yet");

        const auto localOffset = distribution().offset();
        const auto localSize = distribution().localSize();
        const auto ilower = localOffset * block_size;
        const auto iupper = ilower + localSize * block_size - 1;

        setProfile(ilower, iupper);
    }

    Vector::~Vector() {
        if (m_hypre)
            HYPRE_IJVectorDestroy(m_hypre);
    }

    void Vector::setProfile(int ilower, int iupper) {
        if (m_hypre)
            HYPRE_IJVectorDestroy(m_hypre);

        auto* pm = dynamic_cast<Arccore::MessagePassing::Mpi::MpiMessagePassingMng *>(distribution().parallelMng());
        m_comm = pm?(*pm->getMPIComm()):MPI_COMM_WORLD;

        // -- B Vector --
        auto ierr = HYPRE_IJVectorCreate(m_comm, ilower, iupper, &m_hypre);
        ierr |= HYPRE_IJVectorSetObjectType(m_hypre, HYPRE_PARCSR);
        ierr |= HYPRE_IJVectorInitialize(m_hypre);

        if (ierr) {
            throw Arccore::FatalErrorException(A_FUNCINFO, "Hypre Initialisation failed");
        }

        m_rows.resize(iupper - ilower + 1);
        for (int i = 0; i < m_rows.size(); ++i)
            m_rows[i] = ilower + i;
    }

    void Vector::setValues(Arccore::ConstArrayView<double> values) {
        auto ierr = HYPRE_IJVectorSetValues(m_hypre, m_rows.size(), m_rows.data(), values.data());

        if (ierr) {
            throw Arccore::FatalErrorException(A_FUNCINFO, "Hypre set values failed");
        }
    }

    void Vector::getValues(Arccore::ArrayView<double> values) const {
        auto ierr = HYPRE_IJVectorGetValues(m_hypre, m_rows.size(), m_rows.data(), values.data());

        if (ierr) {
            throw Arccore::FatalErrorException(A_FUNCINFO, "Hypre get values failed");
        }
    }

    void Vector::assemble() {
        auto ierr = HYPRE_IJVectorAssemble(m_hypre);

        if (ierr) {
            throw Arccore::FatalErrorException(A_FUNCINFO, "Hypre assembling failed");
        }
    }
}