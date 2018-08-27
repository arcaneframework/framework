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

  MessagePassingMng(IDispatchers* d);
  ~MessagePassingMng() override;

 public:

  IDispatchers* dispatchers() override;

 private:

  IDispatchers* m_dispatchers;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace MessagePassing
} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
