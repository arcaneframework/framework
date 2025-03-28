// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RequestListBase.h                                           (C) 2000-2025 */
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

namespace Arcane::MessagePassing::internal
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

  void add(Request r) override { _add(r); }
  void add(Span<Request> r) override { _add(r); }
  Int32 wait(eWaitType wait_type) final;
  Int32 size() const override { return m_requests.size(); }
  void clear() final;
  void removeDoneRequests() override;
  bool isRequestDone(Int32 index) const override { return m_requests_done[index]; }
  Request request(Int32 index) const override { return m_requests[index]; }
  ConstArrayView<Int32> doneRequestIndexes() const final;

 protected:

  virtual void _add(const Request& r)
  {
    m_requests.add(r);
    m_requests_done.add(false);
  }
  virtual void _add(Span<Request> rlist)
  {
    m_requests.addRange(rlist);
    m_requests_done.addRange(false,rlist.size());
  }
  virtual void _removeRequestAtIndex(Integer pos)
  {
    m_requests.remove(pos);
    m_requests_done.remove(pos);
  }
  /*!
   * \brief Effectue l'attente ou le test.
   *
   * L'implémentation doit remplir à \a _requestsDone() avec la
   * valeur \a true pour chaque requête terminée sauf si
   * \a wait_type vaut WaitAll.
   */
  virtual void _wait(eWaitType wait_type) =0;

 protected:

  ArrayView<Request> _requests() { return m_requests; }
  ArrayView<bool> _requestsDone() { return m_requests_done; }

 private:

  UniqueArray<Request> m_requests;
  UniqueArray<bool> m_requests_done;
  UniqueArray<Int32> m_done_request_indexes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

namespace Arccore::MessagePassing::internal
{
using Arcane::MessagePassing::internal::RequestListBase;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

