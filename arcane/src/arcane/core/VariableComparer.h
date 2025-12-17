// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableComparer.h                                          (C) 2000-2025 */
/*                                                                           */
/* Classe pour effectuer des comparaisons entre les variables.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARIABLECOMPARER_H
#define ARCANE_CORE_VARIABLECOMPARER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Méthode de comparaison à utiliser
enum class eVariableComparerCompareMode
{
  //! Compare avec une référence
  Same = 0,
  //! Vérifie que la variable est bien synchronisée
  Sync = 1,
  //! Vérifie que les valeurs de la variable sont les même sur tous les replica
  SameOnAllReplica = 2
};

//! Méthode utilisée pour calculer la différence entre deux valeurs \a v1 et \a v2.
enum class eVariableComparerComputeDifferenceMethod
{
  //! Utilise (v1-v2) / v1
  Relative,
  /*!
   * \brief Utilise (v1-v2) / local_norm_max.
   *
   * \a local_norm_max est le maximum des math::abs() des valeurs sur le sous-domaine.
   */
  LocalNormMax,
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Arguments des méthodes de VariableComparer.
 */
class ARCANE_CORE_EXPORT VariableComparerArgs
{
 public:

  /*!
   * \brief Positionne le nombre d'erreurs à afficher dans le listing.
   *
   * Si 0, aucun élément n'est affiché. Si positif, affiche au plus
   * \a v élément. Si négatif, tous les éléments sont affichés.
   */
  void setMaxPrint(Int32 v) { m_max_print = v; }
  Int32 maxPrint() const { return m_max_print; }

  /*!
   * \brief Indique sur quelles entités on fait la comparaison.
   *
   * Si \a v si vrai, compare les valeurs à la fois sur les entités
   * propres et les entités fantômes. Sinon, ne fait la comparaison que sur les
   * entités propres.
   *
   * Ce paramètre n'est utilisé que si compareMode() vaut eCompareMode::Same.
   */
  void setCompareGhost(bool v) { m_is_compare_ghost = v; }
  bool isCompareGhost() const { return m_is_compare_ghost; }

  void setDataReader(IDataReader* v) { m_data_reader = v; }
  IDataReader* dataReader() const { return m_data_reader; }

  void setCompareMode(eVariableComparerCompareMode v) { m_compare_mode = v; }
  eVariableComparerCompareMode compareMode() const { return m_compare_mode; }

  void setComputeDifferenceMethod(eVariableComparerComputeDifferenceMethod v) { m_compute_difference_method = v; }
  eVariableComparerComputeDifferenceMethod computeDifferenceMethod() const { return m_compute_difference_method; }

 private:

  Int32 m_max_print = 0;
  bool m_is_compare_ghost = false;
  IDataReader* m_data_reader = nullptr;
  eVariableComparerCompareMode m_compare_mode = eVariableComparerCompareMode::Same;
  eVariableComparerComputeDifferenceMethod m_compute_difference_method = eVariableComparerComputeDifferenceMethod::Relative;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Résultats d'une opération de comparaison.
 */
class ARCANE_CORE_EXPORT VariableComparerResults
{
 public:

  VariableComparerResults() = default;
  explicit VariableComparerResults(Int32 nb_diff)
  : m_nb_diff(nb_diff)
  {}

 public:

  void setNbDifference(Int32 v) { m_nb_diff = v; }
  Int32 nbDifference() const { return m_nb_diff; }

 public:

  Int32 m_nb_diff = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour effectuer des comparaisons entre les variables.
 *
 * Pour utiliser cette classe, il faut créer une instance de
 * VariableComparerArgs via une des méthodes suivantes:
 *
 * - buildForCheckIfSame()
 * - buildForCheckIfSync()
 * - buildForCcheckIfSameOnAllReplica()
 *
 * Il faut ensuite appeler la méthode apply() avec l'instance créée pour
 * chaque variable pour laquelle on souhaite faire la comparaison.
 */
class ARCANE_CORE_EXPORT VariableComparer
{
 public:

  VariableComparer() = default;

 public:

  /*!
   * \brief Créé une comparaison pour vérifie qu'une variable est
   * bien synchronisée.
   *
   * Cette opération ne fonctionne que pour les variables de maillage.
   *
   * Un variable est synchronisée lorsque ses valeurs sont les mêmes
   * sur tous les sous-domaines à la fois sur les éléments propres et
   * les éléments fantômes.
   *
   * Il est possible d'appeler sur l'instance retournée les méthodes
   * VariableComparerArgs::setMaxPrint(),
   * VariableComparerArgs::setCompareGhost()
   * ou VariableComparerArgs::setComputeDifferenceMethod() pour modifier
   * le comportement.
   */
  VariableComparerArgs buildForCheckIfSync();

  /*!
   * \brief Créé une comparaison pour vérifie qu'une variable est identique
   * sur tous les réplicas.
   *
   * Compare les valeurs de la variable avec celle du même sous-domaine
   * des autres réplicas. Pour chaque élément différent,
   * un message est affiché.
   *
   * L'utilisation de apply() pour les comparaisons de ce type est une
   * méthode collective sur le replica de la variable passée en argument.
   * Il ne faut donc l'appeler que si la variable existe sur tous les sous-domaines
   * sinon cela provoque un blocage.
   *
   * Cette comparaison ne fonctionne que pour les variables sur les types numériques.
   * Dans ce cas, elle renvoie une exception de type NotSupportedException.
   *
   * Il est possible d'appeler sur l'instance retournée les méthodes
   * VariableComparerArgs::setMaxPrint() ou
   * VariableComparerArgs::setComputeDifferenceMethod() pour modifier le comportement.
   */
  VariableComparerArgs buildForCheckIfSameOnAllReplica();

 public:

  /*!
   * \brief Créé une comparaison pour vérifie qu'une variable est identique
   * à une valeur de référence.
   *
   * Cette opération vérifie que les valeurs de la variable sont identiques
   * à une valeur de référence qui sera lue à partir du lecteur \a data_reader.
   *
   * Il est possible d'appeler sur l'instance retournée les méthodes
   * VariableComparerArgs::setMaxPrint(),
   * VariableComparerArgs::setCompareGhost() ou
   * VariableComparerArgs::setComputeDifferenceMethod() pour modifier le comportement.
   *
   * Il est ensuite possible d'appeler la méthode apply() sur l'instance
   * retournée pour effectuer les comparaisons sur une variable.
   */
  VariableComparerArgs buildForCheckIfSame(IDataReader* data_reader);

 public:

  //! Applique la comparaison \a compare_args à la variable \a var
  VariableComparerResults apply(IVariable* var, const VariableComparerArgs& compare_args);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
