// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HashFunction.h                                              (C) 2000-2020 */
/*                                                                           */
/* Fonction de hachage.                                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/HashFunction.h"
#include "arcane/utils/String.h"

#include <cstdint>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//-----------------------------------------------------------------------------
// MurmurHash2, 64-bit versions, by Austin Appleby

//typedef unsigned Int64 uint64_t;

namespace
{
uint64_t _MurmurHash64A(const void* key,uint64_t len,unsigned int seed)
{
  const uint64_t m = 0xc6a4a7935bd1e995;
  const int r = 47;

  uint64_t h = seed ^ (len * m);

  const uint64_t * data = (const uint64_t *)key;
  const uint64_t * end = data + (len/8);

  while(data != end){
    uint64_t k = *data++;

    k *= m; 
    k ^= k >> r; 
    k *= m; 
		
    h ^= k;
    h *= m; 
  }

  const unsigned char * data2 = (const unsigned char*)data;

  switch(len & 7){
  case 7: h ^= uint64_t(data2[6]) << 48; [[fallthrough]];
  case 6: h ^= uint64_t(data2[5]) << 40; [[fallthrough]];
  case 5: h ^= uint64_t(data2[4]) << 32; [[fallthrough]];
  case 4: h ^= uint64_t(data2[3]) << 24; [[fallthrough]];
  case 3: h ^= uint64_t(data2[2]) << 16; [[fallthrough]];
  case 2: h ^= uint64_t(data2[1]) << 8; [[fallthrough]];
  case 1: h ^= uint64_t(data2[0]); h *= m;
  };
 
  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  return h;
}

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 IntegerHashFunctionT<StringView>::
hashfunc(StringView str)
{
  Span<const Byte> bytes(str.bytes());
  return _MurmurHash64A(bytes.data(),bytes.size(),0x424);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

