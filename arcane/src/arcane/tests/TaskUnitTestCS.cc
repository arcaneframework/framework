// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TaskUnitTestCS.cc                                           (C) 2000-2019 */
/*                                                                           */
/* Task testing service using C# wrapping.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ITraceMng.h"

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/Concurrency.h"
#include "arcane/core/ServiceFactory.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/VariableCollection.h"
#include "arcane/core/IDataWriter.h"

#include "arcane/tests/ArcaneTestGlobal.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

namespace TaskTest
{

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*!
 * \brief Task testing module
 */
  class TaskUnitTestCS
  : public BasicUnitTest
  {
   public:
   public:

    TaskUnitTestCS(const ServiceBuildInfo& cb);
    ~TaskUnitTestCS();

   public:

    virtual void initializeTest();
    virtual void executeTest();

   private:

    Ref<IDataWriter> m_cs_data_writer;
    UniqueArray<IVariable*> m_variables;
    std::atomic<Int32> m_nb_done;

   private:

    void _oneWrite(Integer begin, Integer size);
  };

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  ARCANE_REGISTER_SERVICE(TaskUnitTestCS,
                          ServiceProperty("TaskUnitTestCS", ST_CaseOption),
                          ARCANE_SERVICE_INTERFACE(IUnitTest));

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  TaskUnitTestCS::
  TaskUnitTestCS(const ServiceBuildInfo& sb)
  : BasicUnitTest(sb)
  {
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  TaskUnitTestCS::
  ~TaskUnitTestCS()
  {
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  void TaskUnitTestCS::
  executeTest()
  {
    Integer nb_var = m_variables.size();
    info() << "EXECUTE TEST nb_var=" << nb_var;
    m_nb_done = 0;
    ParallelLoopOptions loop_opt;
    loop_opt.setGrainSize(1);
    arcaneParallelFor(0, nb_var, loop_opt, [&](Int32 a, Int32 n) { _oneWrite(a, n); });
    //m_cs_data_writer.reset();
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  void TaskUnitTestCS::
  _oneWrite(Integer begin, Integer size)
  {
    m_nb_done += size;
    info() << "EXECUTE ONE WRITE thread_index = " << TaskFactory::currentTaskThreadIndex()
           << " begin=" << begin << " size=" << size << " (nb_done=" << m_nb_done << ")";

    IVariable* var = m_variables[begin];
    m_cs_data_writer->write(var, var->data());
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  void TaskUnitTestCS::
  initializeTest()
  {
    info() << "INIT TEST!!!";
    m_cs_data_writer = ServiceBuilder<IDataWriter>::createReference(subDomain(), "DotNetDataWriter");

    VariableCollection used_variables = subDomain()->variableMng()->usedVariables();
    for (VariableCollection::Enumerator i(used_variables); ++i;) {
      m_variables.add(*i);
    }
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace TaskTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
