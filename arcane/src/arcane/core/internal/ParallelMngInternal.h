// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelMngInternal.h                                       (C) 2000-2025 */
/*                                                                           */
/* Implémentation de la partie interne à Arcane de IParallelMng.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_PARALLELMNGINTERNAL_H
#define ARCANE_CORE_INTERNAL_PARALLELMNGINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/internal/IParallelMngInternal.h"

#include "arcane/accelerator/core/Runner.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ParallelMngDispatcher;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Partie interne de IParallelMng.
 */
class ARCANE_CORE_EXPORT ParallelMngInternal
: public IParallelMngInternal
{
 public:

  explicit ParallelMngInternal(ParallelMngDispatcher* pm);

  ~ParallelMngInternal() override = default;

 public:

  Runner runner() const override;
  RunQueue queue() const override;
  bool isAcceleratorAware() const override;
  Ref<IParallelMng> createSubParallelMngRef(Int32 color, Int32 key) override;
  void setDefaultRunner(const Runner& runner) override;
  Ref<MessagePassing::IMachineMemoryWindowBaseInternal> createMachineMemoryWindowBase(Int64 sizeof_segment, Int32 sizeof_type) override;

 private:

  ParallelMngDispatcher* m_parallel_mng = nullptr;
  Runner m_runner;
  RunQueue m_queue;
  bool m_is_accelerator_aware_disabled = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
