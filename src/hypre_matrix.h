#pragma once

#include <ALIEN/Core/Impl/IMatrixImpl.h>

namespace Alien::Hypre::Internal {
  class MatrixInternal;
}

namespace Alien::Hypre {

  class Matrix
          : public IMatrixImpl {
  public:

    typedef Internal::MatrixInternal MatrixInternal;

  public:

    Matrix(const MultiMatrixImpl *multi_impl);

    virtual ~Matrix();

  public:

    void clear() {}

  public:

    bool initMatrix(const int ilower, const int iupper,
                    const int jlower, const int jupper,
                    const Arccore::ConstArrayView<Arccore::Integer> &lineSizes);

    // FIXME use Arccore::ArrayView
    bool addMatrixValues(const int nrow, const int *rows,
                         const int *ncols, const int *cols,
                         const Arccore::Real *values);

    // FIXME use Arccore::ArrayView
    bool setMatrixValues(const int nrow, const int *rows,
                         const int *ncols, const int *cols,
                         const Arccore::Real *values);

    bool assemble();

  public:

    MatrixInternal *internal() { return m_internal; }

    const MatrixInternal *internal() const { return m_internal; }

    Arccore::MessagePassing::IMessagePassingMng *getParallelMng() const { return m_pm; }

  private:

    Arccore::Integer ijk(Arccore::Integer i, Arccore::Integer j, Arccore::Integer k, Arccore::Integer block_size,
                         Arccore::Integer unknowns_num) const {
      return k * block_size + i * unknowns_num + j;
    }

    MatrixInternal *m_internal;
    Arccore::MessagePassing::IMessagePassingMng *m_pm;
  };

}
