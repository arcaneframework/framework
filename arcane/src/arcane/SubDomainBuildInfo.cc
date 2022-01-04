// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SubDomainBuildInfo.cc                                       (C) 2000-2020 */
/*                                                                           */
/* Informations pour construire un sous-domaine.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/SubDomainBuildInfo.h"
#include "arcane/IParallelMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SubDomainBuildInfo::
SubDomainBuildInfo(Ref<IParallelMng> pm,Int32 index)
: m_parallel_mng(pm)
, m_index(index)
, m_all_replica_parallel_mng(m_parallel_mng)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SubDomainBuildInfo::
SubDomainBuildInfo(Ref<IParallelMng> pm,Int32 index,Ref<IParallelMng> all_replica_pm)
: m_parallel_mng(pm)
, m_index(index)
, m_all_replica_parallel_mng(all_replica_pm)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ByteConstArrayView SubDomainBuildInfo::
caseBytes() const
{
  ConstArrayView<std::byte> x { m_case_content.view() };
  return ByteConstArrayView(x.size(), reinterpret_cast<const Byte*>(x.data()));
}

void SubDomainBuildInfo::
setCaseBytes(ByteConstArrayView bytes)
{
  auto d = reinterpret_cast<const std::byte*>(bytes.data());
  m_case_content = ByteConstSpan(d,bytes.size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ByteConstSpan SubDomainBuildInfo::
caseContent() const
{
  return m_case_content;
}

void SubDomainBuildInfo::
setCaseContent(ByteConstSpan content)
{
  m_case_content = content;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
