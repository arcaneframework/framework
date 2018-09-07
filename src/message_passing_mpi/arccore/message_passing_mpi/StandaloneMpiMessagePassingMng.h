/*---------------------------------------------------------------------------*/
/* StandaloneMpiMessagePassingMng.h                            (C) 2000-2018 */
/*                                                                           */
/* Version autonome de MpiMessagePassingMng.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_STANDALONEMPIMESSAGEPASSINGMNG_H
#define ARCCORE_MESSAGEPASSINGMPI_STANDALONEMPIMESSAGEPASSINGMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MpiMessagePassingMng.h"

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
 * \brief Version autonome de MpiMessagePassingMng.
 *
 * La création se fait via la méthode statique create().
 */
class ARCCORE_MESSAGEPASSINGMPI_EXPORT StandaloneMpiMessagePassingMng
: public MpiMessagePassingMng
{
  class Impl;

 private:

  StandaloneMpiMessagePassingMng(Impl* p);

 public:

  ~StandaloneMpiMessagePassingMng() override;

 public:

  //! Créé un gestionnaire associé au communicateur \a comm.
  static MpiMessagePassingMng* create(MPI_Comm comm);

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Mpi
} // End namespace MessagePassing
} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
