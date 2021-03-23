// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef ARRAYUTILS_H_
#define ARRAYUTILS_H_

#include "Utils/Utils.h"

BEGIN_ARCGEOSIM_NAMESPACE

inline
void insert(Array<Integer>& list,
    Array<Real>& value,
    Integer entry)
{
  if(entry==0)
  {
      list[0] = 0 ;
      return ;
  }
  Integer size = entry ;
  Integer i = size ;
  Real z = value[entry] ;
  for(Integer k=0;k<size;k++)
  {
    if(z>=value[list[k]])
    { 
      i=k ;
      break ;
    }
  }
  Integer tmp = entry ;
  Integer last = entry ;
  for(Integer j=i;j<size;j++)
  {
      tmp = list[j] ;
      list[j] = last ;
      last = tmp ; 
  }
  list[size] = last ;
}

inline
Real average(ArrayView<Real> x,
             ArrayView<Real> coef,
             Integer n)
{
  Real xx = 0. ;
  for(Integer i=0;i<n;i++)
    xx += coef[i]*x[i] ;
  return xx ;
}

END_ARCGEOSIM_NAMESPACE

#endif /*ARRAYUTILS_H_*/
