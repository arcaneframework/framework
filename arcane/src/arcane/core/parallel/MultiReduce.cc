// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MultiReduce.cc                                              (C) 2000-2013 */
/*                                                                           */
/* Multiple reduction management.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/String.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/core/IParallelMng.h"

#include "arcane/parallel/IMultiReduce.h"

#include <map>
#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class PostProcessingMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 */
class MultiReduce
: public TraceAccessor
, public IMultiReduce
{
 public:

  MultiReduce(IParallelMng* pm);
  ~MultiReduce();

 public:

  virtual void execute();
  virtual bool isStrict() const { return m_is_strict; }
  virtual void setStrict(bool is_strict) { m_is_strict = is_strict; }

 public:

  virtual ReduceSumOfRealHelper* getSumOfReal(const String& name)
  {
    ReduceSumOfRealHelperMap::const_iterator i = m_sum_helpers.find(name);
    ReduceSumOfRealHelper* v = 0;
    if (i == m_sum_helpers.end()) {
      v = new ReduceSumOfRealHelper(m_is_strict);
      m_sum_helpers.insert(std::make_pair(name, v));
    }
    else
      v = i->second;
    return v;
  }

 private:

  typedef std::map<String, ReduceSumOfRealHelper*> ReduceSumOfRealHelperMap;

  IParallelMng* m_parallel_mng;
  bool m_is_strict;
  ReduceSumOfRealHelperMap m_sum_helpers;

 private:

  void _execStrict(ReduceSumOfRealHelper* v);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMultiReduce* IMultiReduce::
create(IParallelMng* pm)
{
  return new MultiReduce(pm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MultiReduce::
MultiReduce(IParallelMng* pm)
: TraceAccessor(pm->traceMng())
, m_parallel_mng(pm)
, m_is_strict(false)
{
  if (!platform::getEnvironmentVariable("ARCANE_STRICT_REDUCE").null())
    m_is_strict = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MultiReduce::
~MultiReduce()
{
  ReduceSumOfRealHelperMap::const_iterator i = m_sum_helpers.begin();
  for (; i != m_sum_helpers.end(); ++i)
    delete i->second;
  m_sum_helpers.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultiReduce::
execute()
{
  if (m_is_strict) {
    ReduceSumOfRealHelperMap::const_iterator i = m_sum_helpers.begin();
    for (; i != m_sum_helpers.end(); ++i)
      _execStrict(i->second);
    return;
  }

  // If we do not want a strict reduction, we store in an
  // array all the sums to be reduced and we perform
  // a single reduction on this array.

  Integer nb_val = arcaneCheckArraySize(m_sum_helpers.size());
  RealUniqueArray values(nb_val);

  // Copy the sums to be reduced into the array values
  {
    Integer index = 0;
    ReduceSumOfRealHelperMap::const_iterator i = m_sum_helpers.begin();
    for (; i != m_sum_helpers.end(); ++i) {
      // In this non-strict case, a single value in the array.
      values[index] = i->second->values()[0];
      ++index;
    }
  }

  // Perform the reduction
  m_parallel_mng->reduce(Parallel::ReduceSum, values);

  {
    Integer index = 0;
    ReduceSumOfRealHelperMap::const_iterator i = m_sum_helpers.begin();
    for (; i != m_sum_helpers.end(); ++i) {
      // In this non-strict case, a single value in the array.
      i->second->setReducedValue(values[index]);
      ++index;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultiReduce::
_execStrict(ReduceSumOfRealHelper* v)
{
  // We want a strict reduction that always gives the same results.
  // For this, a proc (the 0) retrieves all accumulated values.
  // They are then sorted and accumulated and then sent back to everyone.
  RealUniqueArray all_values;
  m_parallel_mng->gatherVariable(v->values(), all_values, 0);
  //info() << "NB_VAL=" << all_values.size();
  std::sort(std::begin(all_values), std::end(all_values));
  Real sum = 0.0;
  for (Integer i = 0, n = all_values.size(); i < n; ++i)
    sum += all_values[i];
  // TODO: it is possible to perform a single broadcast once the reductions
  // of all the v's are completed
  m_parallel_mng->broadcast(RealArrayView(1, &sum), 0);
  v->setReducedValue(sum);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
