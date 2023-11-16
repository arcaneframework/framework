// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <alien/handlers/block/BaseBlockVectorWriter.h>

#include <alien/ref/data/block/VBlockVector.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VBlockVectorWriter : public Common::BlockVectorWriterT<Real>
{
 public:
  VBlockVectorWriter(VBlockVector& vector)
  : Common::BlockVectorWriterT<Arccore::Real>(vector)
  {}

  virtual ~VBlockVectorWriter() {}

  using Common::BlockVectorWriterT<Real>::operator=;
};

/*---------------------------------------------------------------------------*/

class LocalVBlockVectorWriter : public Common::LocalBlockVectorWriterT<Real>
{
 public:
  LocalVBlockVectorWriter(VBlockVector& vector)
  : Common::LocalBlockVectorWriterT<Arccore::Real>(vector)
  {}

  virtual ~LocalVBlockVectorWriter() {}

  using Common::LocalBlockVectorWriterT<Real>::operator=;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
