// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshVariableRef.h                                           (C) 2000-2025 */
/*                                                                           */
/* Class managing a variable on a mesh entity.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MESHVARIABLEREF_H
#define ARCANE_CORE_MESHVARIABLEREF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/VariableRef.h"
#include "arcane/core/ItemGroup.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Variable
 * \brief Base class for a variable on mesh entities.
 */
class ARCANE_CORE_EXPORT MeshVariableRef
: public VariableRef
{
 public:

  //! Constructs a reference linked to the \a module
  explicit MeshVariableRef(const VariableBuildInfo& vb);
  ~MeshVariableRef() override = default;

 protected:

  MeshVariableRef(const MeshVariableRef& rhs);
  MeshVariableRef(IVariable* var);
  MeshVariableRef() {}
  void operator=(const MeshVariableRef& rhs);

 public:

  void synchronize();
  void synchronize(Int32ConstArrayView local_ids);

 protected:

  void _internalInit(IVariable*);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Macro to check that an entity has the same kind as the group
#define ARCANE_CHECK_VALID_ITEM_AND_GROUP_KIND(i) \
  ARCANE_ASSERT((i.kind() == this->itemGroup().itemKind()), ("Item and group kind not same"));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
