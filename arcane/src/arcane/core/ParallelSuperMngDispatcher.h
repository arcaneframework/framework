// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelSuperMngDispatcher.h                                (C) 2000-2025 */
/*                                                                           */
/* Interface du gestionnaire du parallélisme sur un domaine.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_PARALLELSUPERMNGDISPATCHER_H
#define ARCANE_CORE_PARALLELSUPERMNGDISPATCHER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IParallelSuperMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
template <class T> class IParallelDispatchT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Redirige la gestion des messages des sous-domaines
 * suivant le type de l'argument.
 */
class ARCANE_CORE_EXPORT ParallelSuperMngDispatcher
: public IParallelSuperMng
{
 public:

  ParallelSuperMngDispatcher();
  ~ParallelSuperMngDispatcher() override;

 protected:

  void _setDispatchers(IParallelDispatchT<Byte>* b, IParallelDispatchT<Int32>* i32,
                       IParallelDispatchT<Int64>* i64, IParallelDispatchT<Real>* r);
  void _finalize();

 public:

  virtual void allGather(ByteConstArrayView send_buf, ByteArrayView recv_buf);
  virtual void allGather(Int32ConstArrayView send_buf, Int32ArrayView recv_buf);
  virtual void allGather(Int64ConstArrayView send_buf, Int64ArrayView recv_buf);
  virtual void allGather(RealConstArrayView send_buf, RealArrayView recv_buf);

  virtual Int32 reduce(eReduceType rt, Int32 v);
  virtual Int64 reduce(eReduceType rt, Int64 v);
  virtual Real reduce(eReduceType rt, Real v);

  virtual void reduce(eReduceType rt, Int32ArrayView v);
  virtual void reduce(eReduceType rt, Int64ArrayView v);
  virtual void reduce(eReduceType rt, RealArrayView v);

  void broadcast(ByteArrayView send_buf, Integer id) override;
  void broadcast(Int32ArrayView send_buf, Integer id) override;
  void broadcast(Int64ArrayView send_buf, Integer id) override;
  void broadcast(RealArrayView send_buf, Integer id) override;

  virtual void send(ByteConstArrayView values, Integer id);
  virtual void send(Int32ConstArrayView values, Integer id);
  virtual void send(Int64ConstArrayView values, Integer id);
  virtual void send(RealConstArrayView values, Integer id);

  virtual void recv(ByteArrayView values, Integer id);
  virtual void recv(Int32ArrayView values, Integer id);
  virtual void recv(Int64ArrayView values, Integer id);
  virtual void recv(RealArrayView values, Integer id);

  virtual Request send(ByteConstArrayView values, Integer id, bool is_blocked);
  virtual Request send(Int32ConstArrayView values, Integer id, bool is_blocked);
  virtual Request send(Int64ConstArrayView values, Integer id, bool is_blocked);
  virtual Request send(RealConstArrayView values, Integer id, bool is_blocked);

  virtual Request recv(ByteArrayView values, Integer id, bool is_blocked);
  virtual Request recv(Int32ArrayView values, Integer id, bool is_blocked);
  virtual Request recv(Int64ArrayView values, Integer id, bool is_blocked);
  virtual Request recv(RealArrayView values, Integer id, bool is_blocked);

  virtual void sendRecv(ByteConstArrayView send_buf, ByteArrayView recv_buf, Integer id);
  virtual void sendRecv(Int32ConstArrayView send_buf, Int32ArrayView recv_buf, Integer id);
  virtual void sendRecv(Int64ConstArrayView send_buf, Int64ArrayView recv_buf, Integer id);
  virtual void sendRecv(RealConstArrayView send_buf, RealArrayView recv_buf, Integer id);

  virtual void allToAll(ByteConstArrayView send_buf, ByteArrayView recv_buf, Integer count);
  virtual void allToAll(Int32ConstArrayView send_buf, Int32ArrayView recv_buf, Integer count);
  virtual void allToAll(Int64ConstArrayView send_buf, Int64ArrayView recv_buf, Integer count);
  virtual void allToAll(RealConstArrayView send_buf, RealArrayView recv_buf, Integer count);

  virtual Int32 scan(eReduceType rt, Int32 v);
  virtual Int64 scan(eReduceType rt, Int64 v);
  virtual Real scan(eReduceType rt, Real v);

  virtual void scan(eReduceType rt, Int32ArrayView v);
  virtual void scan(eReduceType rt, Int64ArrayView v);
  virtual void scan(eReduceType rt, RealArrayView v);

 private:

  IParallelDispatchT<Byte>* m_byte = nullptr;
  IParallelDispatchT<Int32>* m_int32 = nullptr;
  IParallelDispatchT<Int64>* m_int64 = nullptr;
  IParallelDispatchT<Real>* m_real = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
