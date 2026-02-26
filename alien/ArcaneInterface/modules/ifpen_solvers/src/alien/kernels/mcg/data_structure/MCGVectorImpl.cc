// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0

#include "MCGVectorImpl.h"

namespace Alien {
template class MCGVector<Real,MCGInternal::eMemoryDomain::Host>;
template class MCGVector<Real,MCGInternal::eMemoryDomain::Device>;
}