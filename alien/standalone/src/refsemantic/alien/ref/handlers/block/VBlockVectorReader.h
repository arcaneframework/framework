// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

#include <alien/handlers/block/BaseBlockVectorReader.h>

#include <alien/ref/data/block/VBlockVector.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VBlockVectorReader
: public Common::BlockVectorReaderT<Arccore::Real, Parameters<GlobalIndexer>>
{
 public:
  VBlockVectorReader(const VBlockVector& vector)
  : Common::BlockVectorReaderT<Arccore::Real, Parameters<GlobalIndexer>>(vector)
  {}

  virtual ~VBlockVectorReader() {}
};

/*---------------------------------------------------------------------------*/

class LocalVBlockVectorReader
: public Common::BlockVectorReaderT<Arccore::Real, Parameters<LocalIndexer>>
{
 public:
  LocalVBlockVectorReader(const BlockVector& vector)
  : Common::BlockVectorReaderT<Arccore::Real, Parameters<LocalIndexer>>(vector)
  {}

  virtual ~LocalVBlockVectorReader() {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
