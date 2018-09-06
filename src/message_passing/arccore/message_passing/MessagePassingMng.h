/*---------------------------------------------------------------------------*/
/* MessagePassingMng.h                                         (C) 2000-2018 */
/*                                                                           */
/* Gestionnaire des échanges de messages.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_MESSAGEPASSINGMNG_H
#define ARCCORE_MESSAGEPASSING_MESSAGEPASSINGMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/messagepassing/IMessagePassingMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
namespace MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du gestionnaire des échanges de messages.
 */
class ARCCORE_MESSAGEPASSING_EXPORT MessagePassingMng
: public IMessagePassingMng
{
 public:

  MessagePassingMng(Int32 comm_rank,Int32 comm_size,IDispatchers* d);
  ~MessagePassingMng() override;

 public:

  Int32 commRank() const override { return m_comm_rank; }
  Int32 commSize() const override { return m_comm_size; }
  IDispatchers* dispatchers() override;

 private:

  Int32 m_comm_rank;
  Int32 m_comm_size;
  IDispatchers* m_dispatchers;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace MessagePassing
} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
