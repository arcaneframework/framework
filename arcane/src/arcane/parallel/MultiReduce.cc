// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MultiReduce.cc                                              (C) 2000-2013 */
/*                                                                           */
/* Gestion de réductions multiples.                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/String.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/IParallelMng.h"

#include "arcane/parallel/IMultiReduce.h"

#include <map>
#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

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
    if (i==m_sum_helpers.end()){
      v = new ReduceSumOfRealHelper(m_is_strict);
      m_sum_helpers.insert(std::make_pair(name,v));
    }
    else
      v = i->second;
    return v;
  }

 private:

  typedef std::map<String,ReduceSumOfRealHelper*> ReduceSumOfRealHelperMap;

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
  for( ; i!=m_sum_helpers.end(); ++i )
    delete i->second;
  m_sum_helpers.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MultiReduce::
execute()
{
  if (m_is_strict){
    ReduceSumOfRealHelperMap::const_iterator i = m_sum_helpers.begin();
    for( ; i!=m_sum_helpers.end(); ++i )
      _execStrict(i->second);
    return;
  }

  // Si on ne souhaite pas de réduction stricte, on stocke dans un
  // tableau l'ensemble des sommes à réduire et on effectue
  // une seule réduction sur ce tableau.

  Integer nb_val = arcaneCheckArraySize(m_sum_helpers.size());
  RealUniqueArray values(nb_val);
    
  // Copie dans \a values les sommes à réduire
  {
    Integer index = 0;
    ReduceSumOfRealHelperMap::const_iterator i = m_sum_helpers.begin();
    for( ; i!=m_sum_helpers.end(); ++i ){
      // Dans ce cas non strict, une seule valeur le tableau.
      values[index] = i->second->values()[0];
      ++index;
    }
  }

  // Effectue la réduction
  m_parallel_mng->reduce(Parallel::ReduceSum,values);

  {
    Integer index = 0;
    ReduceSumOfRealHelperMap::const_iterator i = m_sum_helpers.begin();
    for( ; i!=m_sum_helpers.end(); ++i ){
      // Dans ce cas non strict, une seule valeur le tableau.
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
  // On souhaite une réduction stricte qui donne toujours les même résultats.
  // Pour cela, un proc (le 0) récupère toutes les valeurs accumulées.
  // Elles sont ensuite triées et accumulées puis renvoyées à tous le monde.
  RealUniqueArray all_values;
  m_parallel_mng->gatherVariable(v->values(),all_values,0);
  //info() << "NB_VAL=" << all_values.size();
  std::sort(std::begin(all_values),std::end(all_values));
  Real sum = 0.0;
  for( Integer i=0, n=all_values.size(); i<n; ++i )
    sum += all_values[i];
  // TODO: il est possible de faire un seul broadcast une fois les réductions
  // de tous les \a v effectuées
  m_parallel_mng->broadcast(RealArrayView(1,&sum),0);
  v->setReducedValue(sum);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
