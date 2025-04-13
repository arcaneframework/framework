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
/*!
 * \brief Arguments des méthodes de VariableComparer.
 */
class ARCANE_CORE_EXPORT VariableComparerArgs
{
 public:

  //! Méthode de comparaison
  enum class eCompareMode
  {
    //! Compare avec une référence
    Same = 0,
    //! Vérifie que la variable est bien synchronisée
    Sync = 1,
    //! Vérifie que les valeurs de la variable sont les même sur tous les replica
    SameReplica = 2
  };

  //! Méthode utilisée pour calculer la différence entre deux valeurs \a v1 et \a v2.
  enum class eComputeDifferenceMethod
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

 public:

  void setMaxPrint(Int32 v) { m_max_print = v; }
  Int32 maxPrint() const { return m_max_print; }

  void setCompareGhost(bool v) { m_is_compare_ghost = v; }
  bool isCompareGhost() const { return m_is_compare_ghost; }

  void setDataReader(IDataReader* v) { m_data_reader = v; }
  IDataReader* dataReader() const { return m_data_reader; }

  void setCompareMode(eCompareMode v) { m_compare_mode = v; }
  eCompareMode compareMode() const { return m_compare_mode; }

  void setComputeDifferenceMethod(eComputeDifferenceMethod v) { m_compute_difference_method = v; }
  eComputeDifferenceMethod computeDifferenceMethod() const { return m_compute_difference_method; }

  void setReplicaParallelMng(IParallelMng* pm) { m_replica_parallel_mng = pm; }
  IParallelMng* replicaParallelMng() const { return m_replica_parallel_mng; }

 private:

  Int32 m_max_print = 0;
  bool m_is_compare_ghost = false;
  IDataReader* m_data_reader = nullptr;
  eCompareMode m_compare_mode = eCompareMode::Same;
  eComputeDifferenceMethod m_compute_difference_method = eComputeDifferenceMethod::Relative;
  IParallelMng* m_replica_parallel_mng = nullptr;
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
 */
class ARCANE_CORE_EXPORT VariableComparer
: public TraceAccessor
{
 public:

  explicit VariableComparer(ITraceMng* tm);

 public:

  /*!
   * \brief Vérifie si la variable \a var est bien synchronisée.
   *
   * Cette opération ne fonctionne que pour les variables de maillage.
   *
   * Un variable est synchronisée lorsque ses valeurs sont les mêmes
   * sur tous les sous-domaines à la fois sur les éléments propres et
   * les éléments fantômes.
   *
   * Pour chaque élément non synchronisé, un message est affiché.
   *
   * \param max_print nombre maximum de messages à afficher.
   * Si 0, aucun élément n'est affiché. Si positif, affiche au plus
   * \a max_print élément. Si négatif, tous les éléments sont affichés.
   *
   * \return le nombre de valeurs différentes de la référence
   */
  Int32 checkIfSync(IVariable* var, Int32 max_print);

  /*!
   * \brief Vérifie que la variable \a var est identique à une valeur de référence
   *
   * Cette opération vérifie que les valeurs de la variable sont identique
   * à une valeur de référence qui est lu à partir du lecteur \a reader.
   *
   * Pour chaque valeur différente de la référence, un message est affiché.
   *
   * \param max_print nombre maximum de messages à afficher.
   * Si 0, aucun élément n'est affiché. Si positif, affiche au plus
   * \a max_print élément. Si négatif, tous les éléments sont affichés.
   * \param compare_ghost si vrai, compare les valeurs à la fois sur les entités
   * propres et les entités fantômes. Sinon, ne fait la comparaison que sur les
   * entités propres.
   *
   * \return le nombre de valeurs différentes de la référence
   */
  Int32 checkIfSame(IVariable* var, IDataReader* reader, Int32 max_print, bool compare_ghost);

  /*!
   * \brief Vérifie si la variable \a var a les mêmes valeurs sur tous les réplicas.
   *
   * Compare les valeurs de la variable avec celle du même sous-domaine
   * des autres réplicas. Pour chaque élément différent,
   * un message est affiché.
   *
   * Cette méthode est collective sur le même sous-domaine des autres réplica.
   * Il ne faut donc l'appeler que si la variable existe sur tous les sous-domaines
   * sinon cela provoque un blocage.
   *
   * Cette méthode ne fonctionne que pour les variables sur les types numériques.
   * Dans ce cas, elle renvoie une exception de type NotSupportedException.
   *
   * \param max_print nombre maximum de messages à afficher.
   * Si 0, aucun élément n'est affiché. Si positif, affiche au plus
   * \a max_print élément. Si négatif, tous les éléments sont affichés.
   * Pour chaque élément différent est affiché la valeur minimale et
   * maximale.
   *
   * \return le nombre de valeurs différentes de la référence.
   */
  Int32 checkIfSameOnAllReplica(IVariable* var, Integer max_print);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
