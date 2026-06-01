// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshGenerator.h                                            (C) 2000-2016 */
/*                                                                           */
/* Interface of the mesh generation service.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_IMESHGENERATOR_H
#define ARCANE_STD_IMESHGENERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real3.h"
#include "arcane/utils/UtilsTypes.h"
#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ISubDomain;
class IPrimaryMesh;
class XmlNode;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class NodeInfo
{
 public:

  NodeInfo()
  : m_owner(NULL_SUB_DOMAIN_ID)
  {}
  NodeInfo(Integer owner, const Real3& coord)
  : m_owner(owner)
  , m_coord(coord)
  {}

 public:

  Integer m_owner;
  Real3 m_coord;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a mesh generator.
 */
class IMeshGenerator
{
 public:

  virtual ~IMeshGenerator() {}

 public:

  virtual IntegerConstArrayView communicatingSubDomains() const = 0;
  virtual bool readOptions(XmlNode node) = 0;
  virtual bool generateMesh() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
