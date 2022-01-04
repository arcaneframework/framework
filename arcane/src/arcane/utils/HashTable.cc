// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HashTable.cc                                                (C) 2000-2014 */
/*                                                                           */
/* Table de hachage.                                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/HashTable.h"
#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/HashTableSet.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/FatalErrorException.h"

#include <exception>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HashTableException
: public std::exception
{
 public:
 public:
  virtual const char* what() const ARCANE_NOEXCEPT
    {
      return "HashTable::throwNotFound()";
    }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HashTableBase::
_throwNotFound() const
{
  cerr << "** FATAL: HashTable:_throwNotFound()\n";
  arcaneDebugPause("HashTableBase::throwNotFound()");
  throw FatalErrorException(A_FUNCINFO,"key not found");
  //throw HashTableException();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer HashTableBase::
nearestPrimeNumber(Integer n)
{
  // Nouvelle liste avec plus d'éléments pour réduire l'usage
  // mémoire
  static Integer PRIME_NUMBER[] = {
    53l,
    97l,
    193l,
    389l,
    769l,
    1009l,
    1213l,
    1459l,
    1753l,
    2111l,
    2539l,
    3049l,
    3659l,
    4391l,
    5273l,
    6329l,
    7603l,
    9127l,
    10957l,
    13151l,
    15787l,
    18947l,
    22739l,
    27299l,
    32771l,
    39341l,
    47221l,
    56671l,
    68023l,
    81629l,
    97961l,
    117563l,
    141079l,
    169307l,
    203173l,
    243809l,
    292573l,
    351097l,
    421331l,
    505601l,
    606731l,
    728087l,
    873707l,
    1048507l,
    1258211l,
    1509857l,
    1811837l,
    2174243l,
    2609107l,
    3145739l,
    3757147l,
    4508597l,
    5410331l,
    6492403l,
    7790897l,
    9349079l,
    11218903l,
    13462693l,
    16155239l,
    19386313l,
    23263577l,
    27916297l,
    //TODO AJOUTER ICI d'autres valeurs
    50331653l,
    100663319l,
    201326611l,
    402653189l,
    805306457l,
    1610612741l
  };
  int nb = sizeof(PRIME_NUMBER) / sizeof(Integer);

  for( Integer i=0; i<nb; ++i )
    if (PRIME_NUMBER[i]>=n){
      return PRIME_NUMBER[i];
    }
  return n;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
