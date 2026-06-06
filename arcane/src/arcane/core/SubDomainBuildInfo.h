// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SubDomainBuildInfo.h                                        (C) 2000-2025 */
/*                                                                           */
/* Information for building a subdomain.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_SUBDOMAINBUILDINFO_H
#define ARCANE_CORE_SUBDOMAINBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Ref.h"
#include "arcane/core/Parallel.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Parameters necessary for building a subdomain.
 
 Info to create a subdomain associated with the parallelism manager
 \a parallelMng(), using the data set name \a caseFileName()
 and content \a caseBytes. \a index() is the index in
 the list of subdomains for this subdomain's session.
 
 The file name is purely informative, and only the content \a caseBytes()
 is used. If \a caseBytes() is not empty, it must contain a
 valid XML document.

 In domain replication, the instance must be constructed by specifying
 the IParallelMng corresponding to the set of subdomains and replicas,
 which is allReplicaParallelMng(). Without replication, this corresponds to the standard
 parallel manager. 
*/
class ARCANE_CORE_EXPORT SubDomainBuildInfo
{
 public:

  SubDomainBuildInfo(Ref<IParallelMng> pm, Int32 index);
  SubDomainBuildInfo(Ref<IParallelMng> pm, Int32 index, Ref<IParallelMng> all_replica_pm);

 public:

  Ref<IParallelMng> parallelMng() const
  {
    return m_parallel_mng;
  }

  String caseFileName() const { return m_case_file_name; }

  void setCaseFileName(const String& filename)
  {
    m_case_file_name = filename;
  }

  ByteConstArrayView caseBytes() const;
  ByteConstSpan caseContent() const;

  void setCaseBytes(ByteConstArrayView bytes);
  void setCaseContent(ByteConstSpan content);

  Integer index() const { return m_index; }

  Ref<IParallelMng> allReplicaParallelMng() const
  {
    return m_all_replica_parallel_mng;
  }

 private:

  Ref<IParallelMng> m_parallel_mng;
  String m_case_file_name; //!< Name of the file containing the data set.
  UniqueArray<std::byte> m_case_content; //!< Data set content
  Int32 m_index; //!< Subdomain number in the session
  //! Parallelism manager containing all replicas of m_parallel_mng
  Ref<IParallelMng> m_all_replica_parallel_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
