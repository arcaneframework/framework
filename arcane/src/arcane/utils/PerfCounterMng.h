// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#ifndef ARCANE_PERFCOUNTERMNG_H_
#define ARCANE_PERFCOUNTERMNG_H_

//GG: Toute cette partie est tres linux/x64 spécifique et n'a rien à faire ici:
// mettre tous ce qui est spécifique dans PlatformUtils
#ifdef ARCANE_OS_LINUX

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

//#define ACTIVATE_PERF_COUNTER
#ifdef ACTIVATE_PERF_COUNTER
#define CHECKPERF(instruction) (instruction) ;
#else
#define CHECKPERF(instruction)
#endif

ARCANE_BEGIN_NAMESPACE

#if defined(__x86_64__)
#ifndef WIN32
static inline void rdtsc(volatile unsigned long long int *counter){

  asm volatile ("rdtsc \n\t"
      "movl %%eax,%0 \n\t"
      "movl %%edx,%1 \n\t"
      : "=m" (((unsigned *)counter)[0]), "=m" (((unsigned *)counter)[1])
      :
      : "eax" , "edx");
}

#define RDTSC(X) asm volatile ("rdtsc \n\t"\
                               "movl %%eax,%0 \n\t"                     \
                               "movl %%edx,%1 \n\t"                     \
                               : "=m" (((unsigned *)(X))[0]), "=m" (((unsigned *)(X))[1]) \
                               :                                        \
                               : "eax" , "edx")
#else

#include <intrin.h>
#pragma intrinsic(__rdtsc)

static inline void rdtsc(volatile unsigned long long int *counter){
    *counter = __rdtsc();
}

#define RDTSC(X) *X=__rdtsc()

#endif // WIN32

#else // defined(__x86_64__)

// IMPLEMENTATION NE FAISANT RIEN: A supprimer avec ce fichier.
static inline void rdtsc(volatile unsigned long long int *counter)
{
  *counter = 1;
}

#endif // defined(__x86_64__)

//! Retourne la fréquence du CPU en Mhz
extern "C++" int arcaneGetCpuBaseFrequency();

template<typename PerfCounterT>
class PerfCounterMng
{
public:
  typedef unsigned long long int ValueType ;
  typedef typename PerfCounterT::eType PhaseType;
  typedef std::pair<ValueType,ValueType> CountType;
  typedef std::vector<CountType> CountListType;

  PerfCounterMng()
  : m_last_value(0)
  {
    m_cpu_frec = arcaneGetCpuBaseFrequency() ;
    m_counts.resize(PerfCounterT::NbCounters) ;
    init() ;
  }

  virtual ~PerfCounterMng(){}

  void init()
  {
    for(std::size_t i=0;i<m_counts.size();++i)
    {
      m_counts[i].first = 0 ;
      m_counts[i].second = 0 ;
    }
  }

  void init(PhaseType const& phase)
  {
    CountType& count = m_counts[phase];
    count.first = 0 ;
    count.second = 0 ;
  }
  void start(PhaseType const& phase)
  {
    rdtsc(&m_counts[phase].second);
  }
  void stop(PhaseType const& phase)
  {
    CountType& count = m_counts[phase];
    rdtsc(&m_last_value) ;
    m_last_value = m_last_value - count.second;
    count.first += m_last_value;
  }

  ValueType getLastValue()
  {
    return m_last_value;
  }

  ValueType getValue(PhaseType const& phase)
  {
    return m_counts[phase].first;
  }

  double getValueInSeconds(PhaseType const& phase)
  {
    return m_counts[phase].first/m_cpu_frec*1E-6;
  }

  void printInfo() const
  {
    std::cout<<"PERF INFO : "<<std::endl ;
    std::cout<<std::setw(10)<<"COUNT"<<" : "<<"VALUE"<<std::endl ;
    //for(typename CountListType::const_iterator iter = m_counts.begin();iter!=m_counts.end();++iter)
    for(std::size_t i=0;i<m_counts.size();++i){
      std::cout<<std::setw(10)<<PerfCounterT::m_names[i]<< " : "<<m_counts[i].first/m_cpu_frec*1E-6<<'\n' ;
    }
  }

  void printInfo(std::ostream& stream) const
  {
    stream<<"PERF INFO : "<<std::endl ;
    stream<<std::setw(10)<<"COUNT"<<" : "<<"VALUE"<<std::endl ;
    //for(typename CountListType::const_iterator iter = m_counts.begin();iter!=m_counts.end();++iter)
    for(std::size_t i=0;i<m_counts.size();++i){
      stream<<std::setw(10)<<PerfCounterT::m_names[i]<< " : "<<m_counts[i].first/m_cpu_frec*1E-6<<'\n' ;
    }
  }


private :
  ValueType m_last_value;
  CountListType m_counts;
  double m_cpu_frec;
};

ARCANE_END_NAMESPACE

#else  // ARCANE_OS_LINUX

#define CHECKPERF(instruction)

#endif // ARCANE_OS_LINUX

#endif /* PERFCOUNTERMNG_H_ */

