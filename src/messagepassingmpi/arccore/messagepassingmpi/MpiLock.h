/*---------------------------------------------------------------------------*/
/* MpiLock.h                                                   (C) 2000-2018 */
/*                                                                           */
/* Verrou pour les appels MPI.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_MPI_MPILOCK_H
#define ARCANE_PARALLEL_MPI_MPILOCK_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/concurrency/SpinLock.h"
#include "arccore/concurrency/Mutex.h"

#include "arccore/messagepassingmpi/MessagePassingMpiGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
namespace MessagePassing
{
namespace Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Verrou pour les appels MPI.
 *
 * Ce verrou sert en multi-threading pour sérialiser les appels
 * MPI en mode MPI_THREAD_SERIALIZED de MPI_Init_thread.
 */
class MpiLock
{
 public:

  // Le spin lock est plus performant mais ne permet pas d'utiliser
  // valgrind.

  //typedef SpinLock LockType;

  typedef Mutex LockType;

 public:
  class Section
  {
   public:
    Section(MpiLock* lock) : mpi_lock(lock)
    {
      if (mpi_lock){
        manual_lock.lock(mpi_lock->m_lock);
      }
    }
    ~Section()
    {
      if (mpi_lock)
        manual_lock.unlock(mpi_lock->m_lock);
    }
   private:
    MpiLock* mpi_lock;
    LockType::ManualLock manual_lock;
  };
  friend class Section;
 public:
  MpiLock() {}
 public:
 private:
  LockType m_lock;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Mpi
} // End namespace MessagePassing
} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

