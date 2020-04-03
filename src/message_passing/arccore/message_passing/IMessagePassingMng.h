// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* IMessagePassingMng.h                                        (C) 2000-2020 */
/*                                                                           */
/* Interface du gestionnaire des échanges de messages.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_IMESSAGEPASSINGMNG_H
#define ARCCORE_MESSAGEPASSING_IMESSAGEPASSINGMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du gestionnaire des échanges de messages.
 *
 * Ce gestionnaire ne fait pas grand chose en lui même et se contente de
 * déléguer les opérations via l'interface IDispatchers.
 * de
 */
class ARCCORE_MESSAGEPASSING_EXPORT IMessagePassingMng
{
 public:

  virtual ~IMessagePassingMng(){}

 public:

  //! Rang de cette instance dans le communicateur
  virtual Int32 commRank() const =0;

  //! Nombre d'instance dans le communicateur
  virtual Int32 commSize() const =0;

  //! Interface pour collecter les temps d'exéctution (peut être nul)
  virtual ITimeMetricCollector* timeMetricCollector() const =0;

 public:

  virtual IDispatchers* dispatchers() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
