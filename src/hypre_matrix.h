#pragma once

#include <alien/core/impl/IMatrixImpl.h>

#include <HYPRE_IJ_mv.h>

namespace Alien::Hypre {

    class Matrix : public IMatrixImpl {
    public:

        Matrix(const MultiMatrixImpl *multi_impl);

        virtual ~Matrix();

    public:

        void setProfile(int ilower, int iupper,
                        int jlower, int jupper,
                        Arccore::ConstArrayView<int> row_sizes);

        void setRowValues(int rows,
                          Arccore::ConstArrayView<int> cols,
                          Arccore::ConstArrayView<double> values);

        void assemble();

        HYPRE_IJMatrix internal() const { return m_hypre; }

    private:

        HYPRE_IJMatrix m_hypre;
        MPI_Comm m_comm;
    };

}
