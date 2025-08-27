// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshVariableRef.h                                           (C) 2000-2024 */
/*                                                                           */
/* Classe gérant une variable sur une entité du maillage.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESHVARIABLEREF_H
#define ARCANE_MESHVARIABLEREF_H
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
 * \brief Classe de base d'une variable sur des entités du maillage.
 */
class ARCANE_CORE_EXPORT MeshVariableRef
: public VariableRef
{
 public:

  //! Construit une référence liée au module \a module
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
//! Macro pour vérifier qu'une entité à le même genre que le groupe
#define ARCANE_CHECK_VALID_ITEM_AND_GROUP_KIND(i) \
  ARCANE_ASSERT((i.kind() == this->itemGroup().itemKind()),("Item and group kind not same"));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
