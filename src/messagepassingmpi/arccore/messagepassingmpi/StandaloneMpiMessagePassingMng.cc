/*---------------------------------------------------------------------------*/
/* StandaloneMpiMessagePassingMng.cc                           (C) 2000-2018 */
/*                                                                           */
/* Implémentation MPI du gestionnaire des échanges de messages.              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/messagepassingmpi/StandaloneMpiMessagePassingMng.h"
#include "arccore/messagepassingmpi/MpiAdapter.h"
#include "arccore/messagepassingmpi/MpiDatatype.h"
#include "arccore/messagepassingmpi/MpiTypeDispatcher.h"
#include "arccore/messagepassing/Dispatchers.h"
#include "arccore/messagepassing/Stat.h"
#include "arccore/trace/ITraceMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
namespace MessagePassing
{
namespace Mpi
{

class StandaloneMpiMessagePassingMng::Impl
{
 public:
  Impl(MPI_Comm mpi_comm)
  : m_trace_mng(nullptr), m_stat(nullptr), m_dispatchers(nullptr),
    m_adapter(nullptr), m_comm_rank(-1), m_comm_size(-1)
  {
    m_trace_mng = Arccore::arccoreCreateDefaultTraceMng();
    ::MPI_Comm_rank(mpi_comm, &m_comm_rank);
    ::MPI_Comm_size(mpi_comm, &m_comm_size);

    m_stat = new Stat();
    MpiLock *mpi_lock = nullptr;
    m_adapter = new MpiAdapter(m_trace_mng, m_stat, mpi_comm, mpi_lock);

    m_dispatchers = new Dispatchers();
  }

  ~Impl()
  {
    m_adapter->destroy();
    delete m_stat;
    delete m_trace_mng;
  }

  MpiMessagePassingMng::BuildInfo buildInfo() const
  {
    return MpiMessagePassingMng::BuildInfo(m_comm_rank,m_comm_size,m_dispatchers);
  }
 public:
  ITraceMng* m_trace_mng;
  IStat* m_stat;
  Dispatchers* m_dispatchers;
  MpiAdapter* m_adapter;
  int m_comm_rank;
  int m_comm_size;
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

namespace{
template<typename DataType> void
_createAndSetDispatcher(Dispatchers* dispatchers,IMessagePassingMng* mpm,MpiAdapter* adapter)
{
  // TODO: gérer la destruction de ces objets.
  MPI_Datatype mpi_dt = MpiBuiltIn::datatype(DataType());
  auto dt = new MpiDatatype(mpi_dt);
  dispatchers->setDispatcher(new MpiTypeDispatcher<DataType>(mpm,adapter,dt));
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiMessagePassingMng* StandaloneMpiMessagePassingMng::
create(MPI_Comm mpi_comm)
{
  Impl* p = new Impl(mpi_comm);
  auto mpm = new StandaloneMpiMessagePassingMng(p);
  auto adapter = p->m_adapter;
  auto dsp = p->m_dispatchers;

  _createAndSetDispatcher<char>(dsp,mpm,adapter);
  _createAndSetDispatcher<signed char>(dsp,mpm,adapter);
  _createAndSetDispatcher<unsigned char>(dsp,mpm,adapter);
  _createAndSetDispatcher<short>(dsp,mpm,adapter);
  _createAndSetDispatcher<unsigned short>(dsp,mpm,adapter);
  _createAndSetDispatcher<int>(dsp,mpm,adapter);
  _createAndSetDispatcher<unsigned int>(dsp,mpm,adapter);
  _createAndSetDispatcher<long>(dsp,mpm,adapter);
  _createAndSetDispatcher<unsigned long>(dsp,mpm,adapter);
  _createAndSetDispatcher<long long>(dsp,mpm,adapter);
  _createAndSetDispatcher<unsigned long long>(dsp,mpm,adapter);
  _createAndSetDispatcher<float>(dsp,mpm,adapter);
  _createAndSetDispatcher<double>(dsp,mpm,adapter);
  _createAndSetDispatcher<long double>(dsp,mpm,adapter);

  return mpm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Mpi
} // End namespace MessagePassing
} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
