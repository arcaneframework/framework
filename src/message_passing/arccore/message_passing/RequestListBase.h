// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* RequestListBase.h                                           (C) 2000-2020 */
/*                                                                           */
/* Classe de base d'une liste de requêtes.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_REQUESTLISTBASE_H
#define ARCCORE_MESSAGEPASSING_REQUESTLISTBASE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/IRequestList.h"
#include "arccore/message_passing/Request.h"
#include "arccore/collections/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing::internal
{
/*!
 * \internal
 * \brief Classe de base d'une liste de requêtes.
 */
class ARCCORE_MESSAGEPASSING_EXPORT RequestListBase
: public IRequestList
{
 public:

  RequestListBase() = default;

 public:

  void addRequest(Request r) override { _addRequest(r); }
  Integer nbRequest() const override { return m_requests.size(); }
  void removeDoneRequests() override;
  bool isRequestDone(Integer index) const override { return m_requests_done[index]; }
  Request request(Integer index) const override { return m_requests[index]; }

 protected:

  virtual void _addRequest(Request r)
  {
    m_requests.add(r);
    m_requests_done.add(false);
  }
  virtual void _removeRequestAtIndex(Integer pos)
  {
    m_requests.remove(pos);
    m_requests_done.remove(pos);
  }

 protected:

  ArrayView<Request> _requests() { return m_requests; }
  ArrayView<bool> _requestsDone() { return m_requests_done; }

 private:

  UniqueArray<Request> m_requests;
  UniqueArray<bool> m_requests_done;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

