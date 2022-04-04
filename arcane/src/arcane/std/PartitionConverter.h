// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PartitionConverter.h                                             (C) 2010 */
/*                                                                           */
/* Utilisation de ArrayConverter pour convertir les poids en flottants       */
/* en poids en entier en les "scalant" correctement.                         */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_STD_PARTITIONCONVERTER_H
#define ARCANE_STD_PARTITIONCONVERTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#include "arcane/utils/ArrayConverter.h"
#include "arcane/utils/ITraceMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*!
 * \brief Conversion d'un tableau de flottants vers un tableau d'entiers/longs.
 * \abstract Cette classe gere le scaling de la facon suivante:
 * [0,W_max] --> [1,EW_max] avec Sum(EW) < "max"
 */
template<typename TypeA,typename TypeB>
class PartitionConverter
{
 public:

  // Max is hard-coded to work on integers.
  PartitionConverter(IParallelMng* pm=NULL, Real max=(2<<30), bool check=false)
    : m_pm(pm), m_maxAllowed(max), m_sum(0), m_zoomfactor(0), m_ready(false), m_check(check)
  {
    m_sum.fill(0.0);
    m_zoomfactor.fill(1.0);
  }

  //< This converter knows how to deal with multi-weights.
  PartitionConverter(IParallelMng* pm, Real max, ConstArrayView<TypeA> input,
		     Integer ncon=1, bool check=false)
    : m_pm(pm), m_maxAllowed(max), m_ready(false), m_check(check)
  {
    reset(ncon);
    computeContrib(input);
  }

  //< Allow to use the same converter in different contexts.
  void reset(Integer ncon=1, bool check=false)
  {
    m_check = check;
    m_sum.resize(ncon+1);
    m_max.resize(ncon);
    m_max.fill(0.0);
    m_zoomfactor.resize(ncon);
    m_sum.fill(0.0);
    m_ready=false;
  }

  //< Check if input can be evenly distributed with imb balance tolerance
  template<typename DataReal>
  bool isBalancable(ConstArrayView<TypeA> input, ArrayView<DataReal> imb, int partnum) {
    bool cond = true;
    if (!m_check || imb.size() < m_max.size())
      return true;

    if (!m_ready) {
      computeContrib(input);
    }

    for (int i = 0 ; i < m_max.size() ; ++i) {
      while (m_max[i] > imb[i]*m_sum[i]/partnum) {
        imb[i] += 0.1f; // Increase imbalance by 10%
        cond = false;
      }
    }
    return cond;
  }

  //< Scan array to compute correct scaling.
  void computeContrib(ConstArrayView<TypeA> input, Real multiplier=1.0)
  {
    Integer ncon = m_zoomfactor.size();
    for( Integer i=0, is=input.size(); i<is; ++i ) {
      m_sum[i%ncon] += (Real)input[i]*multiplier;
    }
    m_sum[ncon] += input.size();
    m_pm->reduce(Parallel::ReduceSum, m_sum);

    if (m_check) { // Check if partition can respect imbalance constraint
      for( Integer i=0, is=input.size(); i<is; ++i ) {
        m_max[i%ncon] = math::max((Real)input[i]*multiplier, m_max[i%ncon]);
      }
      m_pm->reduce(Parallel::ReduceMax, m_max);
    }
    m_zoomfactor[0] = (m_maxAllowed-m_sum[ncon])/(m_sum[0]+1);
    /* for (Integer i = 1 ; i < ncon ; ++i) { // Allow null weight for the other constraints */
    /*   m_zoomfactor[i] = (m_maxAllowed)/(m_sum[i]+1); */
    /* } */

    for (Integer i = 0 ; i < ncon ; ++i) { // Allow null weight for the other constraints
      m_zoomfactor[i] = (m_maxAllowed-m_sum[ncon])/(m_sum[i]+1);
    }

    for (Integer i = 0 ; i < ncon ; ++i) { // Do not scale if not necessary !
      m_zoomfactor[i] = math::min(1.0, m_zoomfactor[i]);
    }
    m_ready = true;
  }

  //< Real convertion is here !
  void convertFromAToB(ConstArrayView<TypeA> input,ArrayView<TypeB> output)
  {
    if (!m_ready)
      computeContrib(input);
    Integer ncon = m_zoomfactor.size();
    for( Integer i=0, is=input.size(); i<is; ++i )
      output[i] = (TypeB)((Real)input[i]*m_zoomfactor[i%ncon])+1;
    // First constraint have to be != 0
    /* for( Integer i=0, is=input.size(); i<is; i+=ncon) */
    /*   output[i] += 1; */
  }

  //< Not implemented as not needed by partitioners.
  void convertFromBToA(ConstArrayView<TypeB> input,ArrayView<TypeA> output)
  {
    ARCANE_UNUSED(input);
    ARCANE_UNUSED(output);
  }

 private:
  IParallelMng* m_pm;
  Real m_maxAllowed;
  SharedArray<Real> m_max;         //< Initial max for each component
  SharedArray<Real> m_sum;         //< Initial sum for each component
  SharedArray<Real> m_zoomfactor;
  bool m_ready;
  bool m_check;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#endif //ARCANE_STD_PARTITIONCONVERTER_H
