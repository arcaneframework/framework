// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITiedInterface.h                                            (C) 2000-2025 */
/*                                                                           */
/* Interface of a class managing semi-conforming mesh.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITIEDINTERFACE_H
#define ARCANE_CORE_ITIEDINTERFACE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/MultiArray2View.h"
#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/TiedNode.h"
#include "arcane/core/TiedFace.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef ConstMultiArray2View<TiedNode> TiedInterfaceNodeList;
typedef ConstMultiArray2View<TiedFace> TiedInterfaceFaceList;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Mesh
 * \brief Interface of a class managing semi-conforming mesh.
 */
class ITiedInterface
{
 public:

  virtual ~ITiedInterface() = default; //!< Releases resources

 public:

  /*!
   * \brief Group containing the master faces.
   *
   * It is a group containing only the entities
   * specific to this subdomain.
   */
  virtual FaceGroup masterInterface() const = 0;

  //! Name of the group containing the master meshes
  virtual String masterInterfaceName() const = 0;

  /*!
   * \brief Group containing the slave faces.
   *
   * It is a group containing only the entities
   * specific to this subdomain.
   */
  virtual FaceGroup slaveInterface() const = 0;

  //! Name of the group containing the slave meshes
  virtual String slaveInterfaceName() const = 0;

  //! List of information about the slave nodes of a master face
  virtual TiedInterfaceNodeList tiedNodes() const = 0;

  //! List of information about the slave faces of a master face
  virtual TiedInterfaceFaceList tiedFaces() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
