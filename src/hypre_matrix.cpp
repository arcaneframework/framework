#include "hypre_matrix.h"
#include "hypre_vector.h"
#include "internal/hypre_internal.h"
#include "hypre_backend.h"

#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>

#include <ALIEN/Core/Impl/MultiMatrixImpl.h>
#include <ALIEN/Data/ISpace.h>

namespace Alien::Hypre {

  Matrix::Matrix(const MultiMatrixImpl *multi_impl)
          : IMatrixImpl(multi_impl, AlgebraTraits<BackEnd::tag::hypre>::name()), m_internal(nullptr), m_pm(nullptr) {
    const auto &row_space = multi_impl->rowSpace();
    const auto &col_space = multi_impl->colSpace();
    if (row_space.size() != col_space.size())
      throw Arccore::FatalErrorException("Hypre matrix must be square");
    m_pm = multi_impl->distribution().parallelMng();
  }

  Matrix::
  ~Matrix() {
    delete m_internal;
  }

  bool
  Matrix::initMatrix(const int ilower, const int iupper, const int jlower,
                     const int jupper, const Arccore::ConstArrayView<Arccore::Integer> &lineSizes) {
    delete m_internal;
    MPI_Comm comm = MPI_COMM_WORLD;
    auto *mpi_comm_mng = dynamic_cast<Arccore::MessagePassing::Mpi::MpiMessagePassingMng *>(m_pm);
    if (mpi_comm_mng)
      comm = *(mpi_comm_mng->getMPIComm());

    m_internal = new MatrixInternal(comm);
    return m_internal->init(ilower, iupper, jlower, jupper, lineSizes);
  }

  bool
  Matrix::addMatrixValues(const int nrow, const int *rows, const int *ncols,
                          const int *cols, const Arccore::Real *values) {
    return m_internal->addMatrixValues(nrow, rows, ncols, cols, values);
  }

  bool
  Matrix::setMatrixValues(const int nrow, const int *rows, const int *ncols,
                          const int *cols, const Arccore::Real *values) {
    return m_internal->setMatrixValues(nrow, rows, ncols, cols, values);
  }

  bool
  Matrix::assemble() {
    return m_internal->assemble();
  }

}