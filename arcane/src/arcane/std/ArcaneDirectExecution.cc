// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneDirectExecution.cc                                    (C) 2000-2011 */
/*                                                                           */
/* Direct execution service.                                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/core/ISubDomain.h"
#include "arcane/core/IDirectExecution.h"

#include "arcane/std/ArcaneDirectExecution_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Direct execution service
 */
class ArcaneDirectExecution
: public ArcaneArcaneDirectExecutionObject
{
 public:

  ArcaneDirectExecution(const ServiceBuildInfo& sb)
  : ArcaneArcaneDirectExecutionObject(sb)
  , m_parallel_mng(0)
  {
    sb.subDomain()->setDirectExecution(this);
  }
  virtual ~ArcaneDirectExecution() {}

 public:

  virtual void build() {}
  virtual void initialize() {}
  virtual void execute()
  {
    Integer nb_tool = options()->tool().size();
    for (Integer i = 0; i < nb_tool; ++i)
      options()->tool()[i]->execute();
  }
  virtual bool isActive() const
  {
    Integer nb_tool = options()->tool().size();
    info() << "** NB_TOOL2=" << nb_tool;
    return options()->tool().size() != 0;
  }
  virtual void setParallelMng(IParallelMng* pm)
  {
    m_parallel_mng = pm;
  }

 public:

  IParallelMng* parallelMng() const { return m_parallel_mng; }

 private:

  IParallelMng* m_parallel_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_ARCANEDIRECTEXECUTION(ArcaneDirectExecution, ArcaneDirectExecution);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
