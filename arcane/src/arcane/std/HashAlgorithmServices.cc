// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HashAlgorithmServices.h                                     (C) 2000-2023 */
/*                                                                           */
/* Services de calcul de Hashs.                                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IOException.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/SHA3HashAlgorithm.h"
#include "arcane/utils/SHA1HashAlgorithm.h"
#include "arcane/utils/MD5HashAlgorithm.h"

#include "arcane/core/AbstractService.h"
#include "arcane/core/ServiceBuildInfo.h"
#include "arcane/core/ServiceFactory.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename HashAlgoImplementation>
class GenericHashAlgorithmService
: public AbstractService
, public HashAlgoImplementation
{
 public:

  explicit GenericHashAlgorithmService(const ServiceBuildInfo& sbi)
  : AbstractService(sbi)
  , m_name(sbi.serviceInfo()->localName())
  {
  }

 public:

  String name() const override { return m_name; }

 private:

  String m_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using SHA3_256HashAlgorithmService = GenericHashAlgorithmService<SHA3_256HashAlgorithm>;
using SHA3_224HashAlgorithmService = GenericHashAlgorithmService<SHA3_224HashAlgorithm>;
using SHA3_384HashAlgorithmService = GenericHashAlgorithmService<SHA3_384HashAlgorithm>;
using SHA3_512HashAlgorithmService = GenericHashAlgorithmService<SHA3_512HashAlgorithm>;
using MD5HashAlgorithmService = GenericHashAlgorithmService<MD5HashAlgorithm>;
using SHA1HashAlgorithmService = GenericHashAlgorithmService<SHA1HashAlgorithm>;

ARCANE_REGISTER_SERVICE(SHA3_256HashAlgorithmService,
                        ServiceProperty("SHA3_256HashAlgorithm", ST_Application | ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(IHashAlgorithm));

ARCANE_REGISTER_SERVICE(SHA3_224HashAlgorithmService,
                        ServiceProperty("SHA3_224HashAlgorithm", ST_Application | ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(IHashAlgorithm));

ARCANE_REGISTER_SERVICE(SHA3_384HashAlgorithmService,
                        ServiceProperty("SHA3_384HashAlgorithm", ST_Application | ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(IHashAlgorithm));

ARCANE_REGISTER_SERVICE(SHA3_512HashAlgorithmService,
                        ServiceProperty("SHA3_512HashAlgorithm", ST_Application | ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(IHashAlgorithm));

ARCANE_REGISTER_SERVICE(MD5HashAlgorithmService,
                        ServiceProperty("MD5HashAlgorithm", ST_Application | ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(IHashAlgorithm));

ARCANE_REGISTER_SERVICE(SHA1HashAlgorithmService,
                        ServiceProperty("SHA1HashAlgorithm", ST_Application | ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(IHashAlgorithm));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
