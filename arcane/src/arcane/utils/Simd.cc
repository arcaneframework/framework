// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Simd.cc                                                     (C) 2000-2018 */
/*                                                                           */
/* Types pour la vectorisation.                                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Simd.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \defgroup ArcaneSimd Vectorisation
 *
 * Ensemble des classes gérant la vectorisation.
 * Pour plus d'informations, se reporter à la page \ref arcanedoc_parallel_simd.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
template<typename SimdRealType> void
_printSimd(std::ostream& o,const SimdRealType& s)
{
  for( Integer z=0, n=SimdRealType::BLOCK_SIZE; z<n; ++z ){
    if (z!=0)
      o << ' ';
    o << s[z];
  }
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_HAS_AVX512
std::ostream&
operator<<(std::ostream& o,const AVX512SimdReal& s)
{
  _printSimd(o,s);
  return o;
}
#endif

#ifdef ARCANE_HAS_AVX
std::ostream&
operator<<(std::ostream& o,const AVXSimdReal& s)
{
  _printSimd(o,s);
  return o;
}
#endif

#ifdef ARCANE_HAS_SSE
std::ostream&
operator<<(std::ostream& o,const SSESimdReal& s)
{
  _printSimd(o,s);
  return o;
}
#endif

std::ostream&
operator<<(std::ostream& o,const EMULSimdReal& s)
{
  _printSimd(o,s);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SimdEnumeratorBase::
_checkValidHelper()
{
  arcaneCheckAlignment(m_local_ids,SimdIndexType::Alignment);
  if (!arcaneIsCheck())
    return;
  Integer size = m_count;
  if (size==0)
    return;
  Integer padding_size = arcaneSizeWithPadding(size);
  if (padding_size==size)
    return;

  // Vérifie que le padding est fait avec la dernière valeur valide.
  Int32 last_local_id = m_local_ids[size-1];
  for( Integer k=size; k<padding_size; ++k )
    if (m_local_ids[k]!=last_local_id)
      ARCANE_FATAL("Bad padding value i={0} expected={1} value={2}",
                   k,last_local_id,m_local_ids[k]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
