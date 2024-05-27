// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IvariableInternal.h                                         (C) 2000-2024 */
/*                                                                           */
/* Partie interne à Arcane de IVariable.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_IVARIABLEINTERNAL_H
#define ARCANE_CORE_INTERNAL_IVARIABLEINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Partie interne de Ivariable.
 */
class ARCANE_CORE_EXPORT IVariableInternal
{
 public:

  virtual ~IVariableInternal() = default;

 public:

  /*!
   * \brief Calcule de Hash de comparaison pour la variable.
   *
   * \a sorted_data doit être trié en fonction des uniqueId() et aussi
   * par rang du IParallelMng associé à la variable.
   *
   * Cette méthode est collective mais seul le rang maitre (celui pour lequel
   * IParallelMng::isMasterIO() est vrai) retourne un hash valide. Les autres
   * retournent une chaîne nulle.
   *
   * Retourn aussi une chaîne nulle si la donnée n'est pas numérique
   * (si sorted_data->_commonInternal()->numericData()==nullptr) ou si
   * la variable n'est pas associée à une entité du maillage.
   */
  virtual String computeComparisonHashCollective(IHashAlgorithm* hash_algo,
                                                 IData* sorted_data) = 0;

  /*!
   * \brief Change l'allocateur de la variable.
   *
   * Actuellemt valide uniquement pour les variables 1D. Ne fait rien pour
   * les autres.
   *
   * \warning For experimental use only.
   */
  virtual void changeAllocator(const MemoryAllocationOptions& alloc_info) = 0;

  //! Redimensionne la variable en ajoutant une capacité additionnelle
  virtual void resizeWithReserve(Int32 new_size,Int32 additional_capacity) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
