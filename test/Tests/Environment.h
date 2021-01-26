#ifndef TESTS_ENVIRONMENT_H
#define TESTS_ENVIRONMENT_H

#include <alien/AlienLegacyConfig.h>
#include <alien/data/Universe.h>

#include <arccore/message_passing_mpi/StandaloneMpiMessagePassingMng.h>
#include <arccore/trace/ITraceMng.h>

namespace Environment {

struct Private
{
  Arccore::ITraceMng* tm;
  Arccore::MessagePassing::IMessagePassingMng* pm;
} __private;

extern void
initialize(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  // Gestionnaire de parallélisme
  __private.pm = Arccore::MessagePassing::Mpi::StandaloneMpiMessagePassingMng::create(
      MPI_COMM_WORLD);

  // Gestionnaire de trace
  __private.tm = Arccore::arccoreCreateDefaultTraceMng();
}

extern void
finalize()
{
  MPI_Finalize();
}

extern Arccore::MessagePassing::IMessagePassingMng*
parallelMng()
{
  Arccore::MessagePassing::IMessagePassingMng* pm = __private.pm;
  return pm;
}

extern Arccore::ITraceMng*
traceMng()
{
  Arccore::ITraceMng* tm = __private.tm;
  return tm;
}
}

namespace Environment {

template <typename T>
extern int
execute(int argc, char** argv, T&& t)
{
  initialize(argc, argv);

  int ret = t();

  finalize();

  return ret;
}
}

#endif
