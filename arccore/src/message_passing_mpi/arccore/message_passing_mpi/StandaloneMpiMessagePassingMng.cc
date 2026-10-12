// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StandaloneMpiMessagePassingMng.cc                           (C) 2000-2026 */
/*                                                                           */
/* MPI implementation of the message exchange manager.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/StandaloneMpiMessagePassingMng.h"

#include "arccore/message_passing/Dispatchers.h"
#include "arccore/message_passing/Stat.h"
#include "arccore/trace/ITraceMng.h"
#include "arccore/base/ReferenceCounter.h"
#include "arccore/base/BFloat16.h"
#include "arccore/base/Float16.h"

#include "arccore/message_passing_mpi/MpiDatatype.h"
#include "arccore/message_passing_mpi/internal/MpiMessagePassingMng.h"
#include "arccore/message_passing_mpi/internal/MpiAdapter.h"
#include "arccore/message_passing_mpi/internal/MpiTypeDispatcher.h"
#include "arccore/message_passing_mpi/internal/MpiControlDispatcher.h"
#include "arccore/message_passing_mpi/internal/MpiSerializeDispatcher.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_MESSAGEPASSINGMPI_EXPORT StandaloneMpiMessagePassingMng2
: public MpiMessagePassingMng
{
  class Impl;
  friend class StandaloneMpiMessagePassingMng;

 private:

  explicit StandaloneMpiMessagePassingMng2(Impl* p);

 public:

  ~StandaloneMpiMessagePassingMng2() override;

 private:

  //! Creates a manager associated with the communicator \a comm.
  static MpiMessagePassingMng* create(MPI_Comm comm, bool clean_comm = false);

  /*!
   * \brief Creates a manager associated with the communicator \a comm.
   *
   * If \a clean_comm is true, MPI_Comm_free() is called on \a comm
   * when the instance is destroyed.
   */
  static Ref<IMessagePassingMng> createRef(MPI_Comm comm, bool clean_comm = false);

  static Ref<IMessagePassingMng> createAndInitializeMPI(int* argc, char** argv[], int wanted_thread_level);

 private:

  Impl* m_p = nullptr;

 private:

  static MpiMessagePassingMng* _create(Impl* p);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class StandaloneMpiMessagePassingMng2::Impl
{
 public:

  explicit Impl(MPI_Comm mpi_comm, bool clean_comm = false)
  : m_communicator(mpi_comm)
  , m_clean_comm(clean_comm)
  {
    _init();
  }

  explicit Impl(int* argc, char** argv[], int wanted_thread_level)
  : m_communicator(MPI_COMM_WORLD)
  {
    int provided = 0;
    int r = MPI_Init_thread(argc, argv, wanted_thread_level, &provided);
    if (r != MPI_SUCCESS)
      ARCCORE_FATAL("Error when calling MPI_Init_thread err={0}", r);
    if (provided != wanted_thread_level) {
      int rank = 0;
      ::MPI_Comm_rank(m_communicator, &rank);
      if (rank == 0)
        std::cerr << "WARNING: MPI_Init_thread provided level (" << provided
                  << ") is not the wanted level (" << wanted_thread_level << ")\n";
    }
    m_do_finalize = true;
    _init();
  }

  ~Impl()
  {
    try {
      m_adapter->destroy();
    }
    catch (const Exception& ex) {
      std::cerr << "ERROR: msg=" << ex << "\n";
    }

    delete m_dispatchers;
    delete m_stat;

    if (m_clean_comm)
      MPI_Comm_free(&m_communicator);
    if (m_do_finalize)
      MPI_Finalize();
  }

  void _init()
  {
    m_trace_mng = Arccore::arccoreCreateDefaultTraceMng();
    MPI_Comm mpi_comm = m_communicator;
    ::MPI_Comm_rank(mpi_comm, &m_comm_rank);
    ::MPI_Comm_size(mpi_comm, &m_comm_size);
    m_stat = new Stat();
    MpiLock* mpi_lock = nullptr;
    m_adapter = new MpiAdapter(m_trace_mng.get(), m_stat, mpi_comm, mpi_lock);

    m_dispatchers = new Dispatchers();
    m_dispatchers->setDeleteDispatchers(true);
  }

  MpiMessagePassingMng::BuildInfo
  buildInfo() const
  {
    return MpiMessagePassingMng::BuildInfo(m_comm_rank, m_comm_size, m_dispatchers, m_communicator);
  }

 public:

  ReferenceCounter<ITraceMng> m_trace_mng;
  IStat* m_stat = nullptr;
  Dispatchers* m_dispatchers = nullptr;
  MpiAdapter* m_adapter = nullptr;
  int m_comm_rank = A_NULL_RANK;
  int m_comm_size = A_NULL_RANK;
  MPI_Comm m_communicator = MPI_COMM_NULL;
  bool m_clean_comm = false;
  bool m_do_finalize = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StandaloneMpiMessagePassingMng2::
StandaloneMpiMessagePassingMng2(Impl* p)
: MpiMessagePassingMng(p->buildInfo())
, m_p(p)
{
  setTraceMng(p->m_trace_mng.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StandaloneMpiMessagePassingMng2::
~StandaloneMpiMessagePassingMng2()
{
  delete m_p;
}

namespace
{
  template <typename DataType> void
  _createAndSetCustomDispatcher(Dispatchers* dispatchers, IMessagePassingMng* mpm,
                                MpiAdapter* adapter, MpiDatatype* datatype)
  {
    auto* x = new MpiTypeDispatcher<DataType>(mpm, adapter, datatype);
    x->setDestroyDatatype(true);
    dispatchers->setDispatcher(x);
  }

  template <typename DataType> void
  _createAndSetDispatcher(Dispatchers* dispatchers, IMessagePassingMng* mpm,
                          MpiAdapter* adapter)
  {
    MPI_Datatype mpi_dt = MpiBuiltIn::datatype(DataType());
    auto dt = new MpiDatatype(mpi_dt);
    _createAndSetCustomDispatcher<DataType>(dispatchers, mpm, adapter, dt);
  }

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiMessagePassingMng* StandaloneMpiMessagePassingMng2::
_create(Impl* p)
{
  auto mpm = new StandaloneMpiMessagePassingMng2(p);
  auto adapter = p->m_adapter;
  auto dsp = p->m_dispatchers;

  _createAndSetDispatcher<char>(dsp, mpm, adapter);
  _createAndSetDispatcher<signed char>(dsp, mpm, adapter);
  _createAndSetDispatcher<unsigned char>(dsp, mpm, adapter);
  _createAndSetDispatcher<short>(dsp, mpm, adapter);
  _createAndSetDispatcher<unsigned short>(dsp, mpm, adapter);
  _createAndSetDispatcher<int>(dsp, mpm, adapter);
  _createAndSetDispatcher<unsigned int>(dsp, mpm, adapter);
  _createAndSetDispatcher<long>(dsp, mpm, adapter);
  _createAndSetDispatcher<unsigned long>(dsp, mpm, adapter);
  _createAndSetDispatcher<long long>(dsp, mpm, adapter);
  _createAndSetDispatcher<unsigned long long>(dsp, mpm, adapter);
  _createAndSetDispatcher<float>(dsp, mpm, adapter);
  _createAndSetDispatcher<double>(dsp, mpm, adapter);
  _createAndSetDispatcher<long double>(dsp, mpm, adapter);

  dsp->setDispatcher(new MpiControlDispatcher(adapter, mpm));
  dsp->setDispatcher(new MpiSerializeDispatcher(adapter, mpm));

  MPI_Datatype uint8_datatype = MpiBuiltIn::datatype(uint8_t{});
  {
    // BFloat16
    MPI_Datatype mpi_datatype;
    MPI_Type_contiguous(2, uint8_datatype, &mpi_datatype);
    MPI_Type_commit(&mpi_datatype);
    auto* x = new MpiDatatype(mpi_datatype, false, new StdMpiReduceOperator<BFloat16>(true));
    _createAndSetCustomDispatcher<BFloat16>(dsp, mpm, adapter, x);
  }
  {
    // Float16
    MPI_Datatype mpi_datatype;
    MPI_Type_contiguous(2, uint8_datatype, &mpi_datatype);
    MPI_Type_commit(&mpi_datatype);
    auto* x = new MpiDatatype(mpi_datatype, false, new StdMpiReduceOperator<Float16>(true));
    _createAndSetCustomDispatcher<Float16>(dsp, mpm, adapter, x);
  }
  return mpm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiMessagePassingMng* StandaloneMpiMessagePassingMng2::
create(MPI_Comm mpi_comm, bool clean_comm)
{
  Impl* p = new Impl(mpi_comm, clean_comm);
  return _create(p);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IMessagePassingMng> StandaloneMpiMessagePassingMng2::
createRef(MPI_Comm mpi_comm, bool clean_comm)
{
  MpiMessagePassingMng* v = create(mpi_comm, clean_comm);
  return makeRef<IMessagePassingMng>(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IMessagePassingMng> StandaloneMpiMessagePassingMng2::
createAndInitializeMPI(int* argc, char** argv[], int wanted_thread_level)
{
  Impl* p = new Impl(argc, argv, wanted_thread_level);
  MpiMessagePassingMng* v = _create(p);
  return makeRef<IMessagePassingMng>(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMessagePassingMng* StandaloneMpiMessagePassingMng::
create(MPI_Comm mpi_comm, bool clean_comm)
{
  return StandaloneMpiMessagePassingMng2::create(mpi_comm, clean_comm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IMessagePassingMng> StandaloneMpiMessagePassingMng::
createRef(MPI_Comm mpi_comm, bool clean_comm)
{
  return StandaloneMpiMessagePassingMng2::createRef(mpi_comm, clean_comm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IMessagePassingMng> StandaloneMpiMessagePassingMng::
createWorldRef()
{
  return StandaloneMpiMessagePassingMng2::createRef(MPI_COMM_WORLD);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IMessagePassingMng> StandaloneMpiMessagePassingMng::
createAndInitializeMPI(int* argc, char** argv[], int wanted_thread_level)
{
  return StandaloneMpiMessagePassingMng2::createAndInitializeMPI(argc, argv, wanted_thread_level);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
