#include "hypre_vector.h"
#include "internal/hypre_internal.h"
#include "hypre_backend.h"

#include <mpi.h>

#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>

namespace Alien::Hypre {

  Vector::Vector(const MultiVectorImpl *multi_impl)
          : IVectorImpl(multi_impl, AlgebraTraits<BackEnd::tag::hypre>::name())
          , m_internal(nullptr)
          , m_block_size(1)
          , m_offset(0) {}

  Vector::
  ~Vector() {
    delete m_internal;
  }

  void
  Vector::init(const VectorDistribution &dist, const bool need_allocate) {
    const Block *block = this->block();
    if (this->block())
      m_block_size *= block->size();
    else if (this->vblock())
      throw Arccore::FatalErrorException(A_FUNCINFO, "Not implemented yet");
    else
      m_block_size = 1;
    m_offset = dist.offset();
    if (need_allocate)
      allocate();
  }

  void
  Vector::allocate() {
    delete m_internal;
    const VectorDistribution &dist = this->distribution();
    auto *pm = dist.parallelMng();
    MPI_Comm comm = MPI_COMM_WORLD;
    auto *mpi_comm_mng = dynamic_cast<Arccore::MessagePassing::Mpi::MpiMessagePassingMng *>(pm);
    if (mpi_comm_mng)
      comm = *(mpi_comm_mng->getMPIComm());

    m_internal = new VectorInternal(comm);
    int ilower = dist.offset() * m_block_size;
    int iupper = ilower + dist.localSize() * m_block_size - 1;
    m_internal->init(ilower, iupper);
    m_rows.resize(dist.localSize() * m_block_size);
    for (int i = 0; i < dist.localSize() * m_block_size; ++i)
      m_rows[i] = ilower + i;
  }

  bool
  Vector::setValues(const int nrow, const int *rows, const double *values) {
    if (m_internal == NULL)
      return false;
    return m_internal->setValues(nrow, rows, values);
  }

  bool
  Vector::setValues(const int nrow, const double *values) {
    if (m_internal == NULL)
      return false;

    return m_internal->setValues(nrow, m_rows.data(), values);
  }

  bool
  Vector::getValues(const int nrow, const int *rows, double *values) const {
    if (m_internal == NULL)
      return false;
    return m_internal->getValues(nrow, rows, values);
  }

  bool
  Vector::getValues(const int nrow, double *values) const {
    if (m_internal == NULL)
      return false;
    return m_internal->getValues(nrow, m_rows.data(), values);
  }

  bool
  Vector::assemble() {
    if (m_internal == NULL)
      return false;
    return m_internal->assemble();
  }

  void
  Vector::update(const Vector &v) {
    ALIEN_ASSERT((this == &v), ("Unexpected error"));
  }

}