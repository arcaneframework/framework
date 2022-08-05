// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TaskUnitTestCS.cc                                           (C) 2000-2019 */
/*                                                                           */
/* Service de test des tâches en utilisant le wrapping C#.                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ITraceMng.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/Concurrency.h"
#include "arcane/ServiceFactory.h"
#include "arcane/ServiceBuilder.h"
#include "arcane/IVariableMng.h"
#include "arcane/VariableCollection.h"

#include "arcane/IDataWriter.h"

#include "arcane/tests/ArcaneTestGlobal.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

namespace TaskTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de test des taches
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

  void _oneWrite(Integer begin,Integer size);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(TaskUnitTestCS,
                        ServiceProperty("TaskUnitTestCS",ST_CaseOption),
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
  arcaneParallelFor(0,nb_var,loop_opt,[&](Int32 a,Int32 n){ _oneWrite(a,n); });
  //m_cs_data_writer.reset();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void  TaskUnitTestCS::
_oneWrite(Integer begin,Integer size)
{
  m_nb_done += size;
  info() << "EXECUTE ONE WRITE thread_index = " << TaskFactory::currentTaskThreadIndex()
         << " begin=" << begin << " size=" << size << " (nb_done=" << m_nb_done << ")";

  IVariable* var = m_variables[begin];
  m_cs_data_writer->write(var,var->data());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TaskUnitTestCS::
initializeTest()
{
  info() << "INIT TEST!!!";
  m_cs_data_writer = ServiceBuilder<IDataWriter>::createReference(subDomain(),"DotNetDataWriter");

  VariableCollection used_variables = subDomain()->variableMng()->usedVariables();
  for( VariableCollection::Enumerator i(used_variables); ++i; ){
    m_variables.add(*i);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
