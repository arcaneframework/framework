// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StandaloneMpiMessagePassingMng.cc                           (C) 2000-2025 */
/*                                                                           */
/* Implémentation MPI du gestionnaire des échanges de messages.              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/StandaloneMpiMessagePassingMng.h"

#include "arccore/message_passing/Dispatchers.h"
#include "arccore/message_passing/Stat.h"
#include "arccore/trace/ITraceMng.h"
#include "arccore/base/ReferenceCounter.h"
#include "arccore/base/BFloat16.h"
#include "arccore/base/Float16.h"

#include "arccore/message_passing_mpi/MpiAdapter.h"
#include "arccore/message_passing_mpi/MpiDatatype.h"
#include "arccore/message_passing_mpi/MpiTypeDispatcher.h"
#include "arccore/message_passing_mpi/MpiControlDispatcher.h"
#include "arccore/message_passing_mpi/MpiSerializeDispatcher.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class StandaloneMpiMessagePassingMng::Impl
{
 public:
  explicit Impl(MPI_Comm mpi_comm, bool clean_comm=false)
  : m_communicator(mpi_comm), m_clean_comm(clean_comm)
  {
    m_trace_mng = Arccore::arccoreCreateDefaultTraceMng();
    ::MPI_Comm_rank(mpi_comm, &m_comm_rank);
    ::MPI_Comm_size(mpi_comm, &m_comm_size);

    m_stat = new Stat();
    MpiLock* mpi_lock = nullptr;
    m_adapter = new MpiAdapter(m_trace_mng.get(), m_stat, mpi_comm, mpi_lock);

    m_dispatchers = new Dispatchers();
    m_dispatchers->setDeleteDispatchers(true);
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
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StandaloneMpiMessagePassingMng::
StandaloneMpiMessagePassingMng(Impl* p)
: MpiMessagePassingMng(p->buildInfo())
, m_p(p)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

StandaloneMpiMessagePassingMng::
~StandaloneMpiMessagePassingMng()
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

MpiMessagePassingMng* StandaloneMpiMessagePassingMng::
create(MPI_Comm mpi_comm, bool clean_comm)
{
  Impl* p = new Impl(mpi_comm, clean_comm);
  auto mpm = new StandaloneMpiMessagePassingMng(p);
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

  dsp->setDispatcher(new MpiControlDispatcher(adapter));
  dsp->setDispatcher(new MpiSerializeDispatcher(adapter));

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

Ref<IMessagePassingMng> StandaloneMpiMessagePassingMng::
createRef(MPI_Comm mpi_comm, bool clean_comm)
{
  MpiMessagePassingMng* v = create(mpi_comm, clean_comm);
  return makeRef<IMessagePassingMng>(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
