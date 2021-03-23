// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariablePrivate.h                               (C) 2000-2019 */
/*                                                                           */
/* Partie privée d'une variable sur un matériau du maillage.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHMATERIALVARIABLEPRIVATE_H
#define ARCANE_MATERIALS_MESHMATERIALVARIABLEPRIVATE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/VariableDependInfo.h"

#include "arcane/materials/MeshMaterialVariableDependInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
class IObserver;
MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Partie privée d'une variable matériau.
 */
class MeshMaterialVariablePrivate
{
 public:
  MeshMaterialVariablePrivate(const MaterialVariableBuildInfo& v,MatVarSpace mvs);
  ~MeshMaterialVariablePrivate();
 public:
  MatVarSpace space() const { return m_var_space; }
  bool hasRecursiveDepend() const { return m_has_recursive_depend; }
  const String& name() const { return m_name; }
  IMeshMaterialMng* materialMng() const { return m_material_mng; }
 public:
  Int32 m_nb_reference;
  MeshMaterialVariableRef* m_first_reference; //! Première référence sur la variable
 private:
  String m_name;
  IMeshMaterialMng* m_material_mng;
 public:
  /*!
   * \brief Stocke les références sur les variables tableaux qui servent pour
   * stocker les valeurs par matériau.
   * Il faut garder une référence pour éviter que la variable ne soit détruite
   * si elle n'est plus utilisée par ailleurs.
   */
  UniqueArray<VariableRef*> m_refs;

  bool m_keep_on_change;
  IObserver* m_global_variable_changed_observer;

  //! Liste des dépendances de cette variable
  UniqueArray<MeshMaterialVariableDependInfo> m_mat_depends;

  //! Liste des dépendances de cette variable
  UniqueArray<VariableDependInfo> m_depends;

  //! Tag de la dernière modification par matériau
  UniqueArray<Int64> m_modified_times;

  //! Fonction de calcul
  ScopedPtrT<IMeshMaterialVariableComputeFunction> m_compute_function;

 private:

  bool m_has_recursive_depend;
  MatVarSpace m_var_space;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

