// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Math.cc                                                     (C) 2000-2015 */
/*                                                                           */
/* Fonctions mathématiques diverses.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Math.h"
#include "arcane/utils/ArrayView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
union DoubleInt
{
  double dblVal;
  unsigned long long intVal;
  explicit DoubleInt(const double x) : dblVal(x) {}
};

//                       1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
int digit_to_bits[] = {  0,  0,  0,  0,  0,  0,  0,  0, 41, 31, 26, 21, 16, 12, 11 };

inline double _doTruncate(double x,unsigned int numDiscardBits)
{
  DoubleInt num(x);
  if (numDiscardBits>0){
    unsigned long long halfBit = (1ULL << (numDiscardBits -1));
    unsigned long long mantissaDiscardMask = ~(halfBit-1) ^ 0xfff0000000000000ULL;
    unsigned long long test = num.intVal & mantissaDiscardMask;
    num.intVal &= 0xfff0000000000000ULL;
    test += halfBit;
    if (test== 0x0010000000000000ULL){
      num.dblVal *= 2.0;
    }
    else{
      num.intVal |= test & ~halfBit;
    }
  }
  return num.dblVal;
}

inline double _doTruncateDigit(double x,int nb_digit)
{
  if (nb_digit<=0)
    return x;
  if (nb_digit<=8)
    return (float)x;
  if (nb_digit>=15)
    return x;
  int nb_bit = digit_to_bits[nb_digit];
  if (nb_bit==0)
    return x;
  return _doTruncate(x,nb_bit);
}

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern double math::
truncateDouble(double v,Integer nb_digit)
{
  return _doTruncateDigit(v,nb_digit); 
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern void math::
truncateDouble(ArrayView<double> values,Integer nb_digit)
{
  Integer n = values.size();
  if (n==0)
    return;

  if (nb_digit<=0)
    return;
  if (nb_digit<=8){
    for( Integer i=0; i<n; ++i )
      values[i] = (float)values[i];
    return;
  }
  if (nb_digit>=15)
    return;
  int nb_bit = digit_to_bits[nb_digit];
  if (nb_bit==0)
    return;
  for( Integer i=0; i<n; ++i )
    values[i] = _doTruncate(values[i],nb_bit);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
