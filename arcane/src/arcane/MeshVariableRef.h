// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshVariableRef.h                                           (C) 2000-2009 */
/*                                                                           */
/* Classe gérant une variable sur une entité du maillage.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESHVARIABLEREF_H
#define ARCANE_MESHVARIABLEREF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/VariableRef.h"
#include "arcane/ItemGroup.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

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
  MeshVariableRef(const VariableBuildInfo& vb);
  virtual ~MeshVariableRef() {}

protected:

  MeshVariableRef(const MeshVariableRef& rhs);
  MeshVariableRef(IVariable* var);
  MeshVariableRef() {}
  void operator=(const MeshVariableRef& rhs);

public:

  void synchronize();

protected:

  void _internalInit(IVariable*);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
