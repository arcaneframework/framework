// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* IRequestList.h                                              (C) 2000-2020 */
/*                                                                           */
/* Interface d'une liste de requêtes de messages.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_REQUESTLIST_H
#define ARCCORE_MESSAGEPASSING_REQUESTLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/Request.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Liste de requête de messages.
 */
class ARCCORE_MESSAGEPASSING_EXPORT IRequestList
{
 public:
  virtual ~IRequestList() = default;
 public:

  virtual void addRequest(Request r) =0;
  virtual Integer nbRequest() const =0;
  virtual Integer wait(eWaitType wait_type) =0;
  virtual void removeDoneRequests() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

