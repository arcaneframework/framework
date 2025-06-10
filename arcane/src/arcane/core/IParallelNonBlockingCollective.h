// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParallelNonBlockingCollective.h                            (C) 2000-2025 */
/*                                                                           */
/* Interface des opérations parallèles collectives non bloquantes.           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IPARALLELNONBLOCKINGCOLLECTIVE_H
#define ARCANE_CORE_IPARALLELNONBLOCKINGCOLLECTIVE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include "arcane/core/Parallel.h"
#include "arcane/core/VariableTypedef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * NOTE:
 * Le but est est IParallelNonBlockingCollective possède les
 * mêmes méthodes collectives que IParallelMng. Cependant, certaines
 * méthodes collectives de IParallleMng font en fait appel dans leur
 * implémentation à plusieurs appels collectifs. Il n'est donc pas
 * possible de transformer cela directement en opérations collectives.
 * Pour implémenter cela avec MPI, il faudrait pouvoir associer un callback
 * à chaque requête (ce callback serait appelé lorsque la requête est terminée)
 * qui permettrait de poursuivre les opérations. Mais cela
 * n'est pas disponible actuellement (peut-être cela est-il possible
 * avec les requêtes généralisées).
 * En attendant, on supprime de l'interface ces appels en les protégeant
 * par un define _NEED_ADVANCED_NBC.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Parallel
 * \brief Interface des opérations parallèles collectives non bloquantes.
 */
class ARCANE_CORE_EXPORT IParallelNonBlockingCollective
{
 public:

  virtual ~IParallelNonBlockingCollective() = default; //!< Libère les ressources.

 public:

  typedef Parallel::Request Request;
  typedef Parallel::eReduceType eReduceType;

 public:

  //! Construit l'instance.
  virtual void build() = 0;

 public:

  //! Gestionnaire de parallélisme associé.
  virtual IParallelMng* parallelMng() const = 0;

 public:

  //! @name allGather
  //@{
  /*!
   * \brief Effectue un regroupement sur tous les processeurs.
   * Il s'agit d'une opération collective. Le tableau \a send_buf
   * doit avoir la même taille, notée \a n, pour tous les processeurs et
   * le tableau \a recv_buf doit avoir une taille égale au nombre
   * de processeurs multiplié par \a n.
   */
  virtual Request allGather(ConstArrayView<char> send_buf, ArrayView<char> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<unsigned char> send_buf, ArrayView<unsigned char> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<signed char> send_buf, ArrayView<signed char> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<int> send_buf, ArrayView<int> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<unsigned int> send_buf, ArrayView<unsigned int> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<short> send_buf, ArrayView<short> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<unsigned short> send_buf, ArrayView<unsigned short> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<long> send_buf, ArrayView<long> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<unsigned long> send_buf, ArrayView<unsigned long> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<long long> send_buf, ArrayView<long long> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<unsigned long long> send_buf, ArrayView<unsigned long long> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<float> send_buf, ArrayView<float> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<double> send_buf, ArrayView<double> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<long double> send_buf, ArrayView<long double> recv_buf) = 0;
#ifdef ARCANE_REAL_NOT_BUILTIN
  virtual Request allGather(ConstArrayView<Real> send_buf, ArrayView<Real> recv_buf) = 0;
#endif
  virtual Request allGather(ConstArrayView<Real2> send_buf, ArrayView<Real2> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<Real3> send_buf, ArrayView<Real3> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<Real2x2> send_buf, ArrayView<Real2x2> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<Real3x3> send_buf, ArrayView<Real3x3> recv_buf) = 0;
  virtual Request allGather(ConstArrayView<HPReal> send_buf, ArrayView<HPReal> recv_buf) = 0;
  //virtual Request allGather(ISerializer* send_serializer,ISerializer* recv_serializer) =0;
  //@}

  //! @name gather
  //@{
  /*!
   * \brief Effectue un regroupement sur un processeurs.
   * Il s'agit d'une opération collective. Le tableau \a send_buf
   * doit avoir la même taille, notée \a n, pour tous les processeurs et
   * le tableau \a recv_buf pour le processeur \a rank doit avoir une taille égale au nombre
   * de processeurs multiplié par \a n. Ce tableau \a recv_buf est inutilisé pour
   * les autres rangs que \a rank.
   */
  virtual Request gather(ConstArrayView<char> send_buf, ArrayView<char> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<unsigned char> send_buf, ArrayView<unsigned char> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<signed char> send_buf, ArrayView<signed char> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<int> send_buf, ArrayView<int> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<unsigned int> send_buf, ArrayView<unsigned int> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<short> send_buf, ArrayView<short> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<unsigned short> send_buf, ArrayView<unsigned short> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<long> send_buf, ArrayView<long> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<unsigned long> send_buf, ArrayView<unsigned long> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<long long> send_buf, ArrayView<long long> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<unsigned long long> send_buf, ArrayView<unsigned long long> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<float> send_buf, ArrayView<float> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<double> send_buf, ArrayView<double> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<long double> send_buf, ArrayView<long double> recv_buf, Integer rank) = 0;
#ifdef ARCANE_REAL_NOT_BUILTIN
  virtual Request gather(ConstArrayView<Real> send_buf, ArrayView<Real> recv_buf, Integer rank) = 0;
#endif
  virtual Request gather(ConstArrayView<Real2> send_buf, ArrayView<Real2> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<Real3> send_buf, ArrayView<Real3> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<Real2x2> send_buf, ArrayView<Real2x2> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<Real3x3> send_buf, ArrayView<Real3x3> recv_buf, Integer rank) = 0;
  virtual Request gather(ConstArrayView<HPReal> send_buf, ArrayView<HPReal> recv_buf, Integer rank) = 0;
  //virtual void gather(ISerializer* send_serializer,ISerializer* recv_serializer,Integer rank) =0;
  //@}

  //! @name allGather variable
  //@{

#if _NEED_ADVANCED_NBC
  /*!
   * \brief Effectue un regroupement sur tous les processeurs.
   *
   * Il s'agit d'une opération collective. Le nombre d'éléments du tableau
   * \a send_buf peut être différent pour chaque processeur. Le tableau
   * \a recv_buf contient en sortie la concaténation des tableaux \a send_buf
   * de chaque processeur. Ce tableau \a recv_buf est éventuellement redimensionné
   * pour le processeurs de rang \a rank.
   */
  virtual Request gatherVariable(ConstArrayView<char> send_buf,
                                 Array<char>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<signed char> send_buf,
                                 Array<signed char>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<unsigned char> send_buf,
                                 Array<unsigned char>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<int> send_buf,
                                 Array<int>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<unsigned int> send_buf,
                                 Array<unsigned int>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<short> send_buf,
                                 Array<short>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<unsigned short> send_buf,
                                 Array<unsigned short>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<long> send_buf,
                                 Array<long>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<unsigned long> send_buf,
                                 Array<unsigned long>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<long long> send_buf,
                                 Array<long long>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<unsigned long long> send_buf,
                                 Array<unsigned long long>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<float> send_buf,
                                 Array<float>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<double> send_buf,
                                 Array<double>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<long double> send_buf,
                                 Array<long double>& recv_buf, Integer rank) = 0;
#ifdef ARCANE_REAL_NOT_BUILTIN
  virtual Request gatherVariable(ConstArrayView<Real> send_buf,
                                 Array<Real>& recv_buf, Integer rank) = 0;
#endif
  virtual Request gatherVariable(ConstArrayView<Real2> send_buf,
                                 Array<Real2>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<Real3> send_buf,
                                 Array<Real3>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<Real2x2> send_buf,
                                 Array<Real2x2>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<Real3x3> send_buf,
                                 Array<Real3x3>& recv_buf, Integer rank) = 0;
  virtual Request gatherVariable(ConstArrayView<HPReal> send_buf,
                                 Array<HPReal>& recv_buf, Integer rank) = 0;
  //@}
#endif

  //! @name allGather variable
  //@{

#if _NEED_ADVANCED_NBC
  /*!
   * \brief Effectue un regroupement sur tous les processeurs.
   *
   * Il s'agit d'une opération collective. Le nombre d'éléments du tableau
   * \a send_buf peut être différent pour chaque processeur. Le tableau
   * \a recv_buf contient en sortie la concaténation des tableaux \a send_buf
   * de chaque processeur. Ce tableau \a recv_buf est éventuellement redimensionné.
   */
  virtual Request allGatherVariable(ConstArrayView<char> send_buf,
                                    Array<char>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<signed char> send_buf,
                                    Array<signed char>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<unsigned char> send_buf,
                                    Array<unsigned char>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<int> send_buf,
                                    Array<int>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<unsigned int> send_buf,
                                    Array<unsigned int>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<short> send_buf,
                                    Array<short>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<unsigned short> send_buf,
                                    Array<unsigned short>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<long> send_buf,
                                    Array<long>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<unsigned long> send_buf,
                                    Array<unsigned long>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<long long> send_buf,
                                    Array<long long>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<unsigned long long> send_buf,
                                    Array<unsigned long long>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<float> send_buf,
                                    Array<float>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<double> send_buf,
                                    Array<double>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<long double> send_buf,
                                    Array<long double>& recv_buf) = 0;
#ifdef ARCANE_REAL_NOT_BUILTIN
  virtual Request allGatherVariable(ConstArrayView<Real> send_buf,
                                    Array<Real>& recv_buf) = 0;
#endif
  virtual Request allGatherVariable(ConstArrayView<Real2> send_buf,
                                    Array<Real2>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<Real3> send_buf,
                                    Array<Real3>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<Real2x2> send_buf,
                                    Array<Real2x2>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<Real3x3> send_buf,
                                    Array<Real3x3>& recv_buf) = 0;
  virtual Request allGatherVariable(ConstArrayView<HPReal> send_buf,
                                    Array<HPReal>& recv_buf) = 0;
  //@}
#endif

#if _NEED_ADVANCED_NBC
  //! @name opérations de réduction sur un scalaire
  //@{
  /*!
   * \brief Scinde un tableau sur plusieurs processeurs.
   */
  virtual Request scatterVariable(ConstArrayView<char> send_buf,
                                  ArrayView<char> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<signed char> send_buf,
                                  ArrayView<signed char> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<unsigned char> send_buf,
                                  ArrayView<unsigned char> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<int> send_buf,
                                  ArrayView<int> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<unsigned int> send_buf,
                                  ArrayView<unsigned int> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<long> send_buf,
                                  ArrayView<long> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<unsigned long> send_buf,
                                  ArrayView<unsigned long> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<long long> send_buf,
                                  ArrayView<long long> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<unsigned long long> send_buf,
                                  ArrayView<unsigned long long> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<float> send_buf,
                                  ArrayView<float> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<double> send_buf,
                                  ArrayView<double> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<long double> send_buf,
                                  ArrayView<long double> recv_buf, Integer root) = 0;
#ifdef ARCANE_REAL_NOT_BUILTIN
  virtual Request scatterVariable(ConstArrayView<Real> send_buf,
                                  ArrayView<Real> recv_buf, Integer root) = 0;
#endif
  virtual Request scatterVariable(ConstArrayView<Real2> send_buf,
                                  ArrayView<Real2> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<Real3> send_buf,
                                  ArrayView<Real3> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<Real2x2> send_buf,
                                  ArrayView<Real2x2> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<Real3x3> send_buf,
                                  ArrayView<Real3x3> recv_buf, Integer root) = 0;
  virtual Request scatterVariable(ConstArrayView<HPReal> send_buf,
                                  ArrayView<HPReal> recv_buf, Integer root) = 0;
  //@}
#endif

  //! @name opérations de réduction sur un tableau
  //@{
  /*!
   * \brief Effectue la réduction de type \a rt sur le tableau \a send_buf et
   * stoque le résultat dans \a recv_buf.
   */
  virtual Request allReduce(eReduceType rt, ConstArrayView<char> send_buf, ArrayView<char> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<signed char> send_buf, ArrayView<signed char> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<unsigned char> send_buf, ArrayView<unsigned char> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<short> send_buf, ArrayView<short> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<unsigned short> send_buf, ArrayView<unsigned short> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<int> send_buf, ArrayView<int> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<unsigned int> send_buf, ArrayView<unsigned int> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<long> send_buf, ArrayView<long> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<unsigned long> send_buf, ArrayView<unsigned long> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<long long> send_buf, ArrayView<long long> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<unsigned long long> send_buf, ArrayView<unsigned long long> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<float> send_buf, ArrayView<float> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<double> send_buf, ArrayView<double> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<long double> send_buf, ArrayView<long double> recv_buf) = 0;
#ifdef ARCANE_REAL_NOT_BUILTIN
  virtual Request allReduce(eReduceType rt, ConstArrayView<Real> send_buf, ArrayView<Real> recv_buf) = 0;
#endif
  virtual Request allReduce(eReduceType rt, ConstArrayView<Real2> send_buf, ArrayView<Real2> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<Real3> send_buf, ArrayView<Real3> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<Real2x2> send_buf, ArrayView<Real2x2> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<Real3x3> send_buf, ArrayView<Real3x3> recv_buf) = 0;
  virtual Request allReduce(eReduceType rt, ConstArrayView<HPReal> send_buf, ArrayView<HPReal> recv_buf) = 0;
  //@}

  /*!
   * @name opérations de broadcast
   *
   * \brief Envoie un tableau de valeurs sur tous les sous-domaines.
   *
   * Cette opération envoie le tableau de valeur \a send_buf sur tous
   * les sous-domaines. Le tableau utilisé est celui dont le rang (commRank) est \a rank.
   * Tous les sous-domaines participants doivent appelés cette méthode avec
   * le même paramètre \a rank et avoir un tableau \a send_buf
   * contenant le même nombre d'éléments.
   */
  //@{
  virtual Request broadcast(ArrayView<char> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<signed char> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<unsigned char> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<short> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<unsigned short> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<int> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<unsigned int> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<long> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<unsigned long> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<long long> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<unsigned long long> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<float> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<double> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<long double> send_buf, Integer rank) = 0;
#ifdef ARCANE_REAL_NOT_BUILTIN
  virtual Request broadcast(ArrayView<Real> send_buf, Integer rank) = 0;
#endif
  virtual Request broadcast(ArrayView<Real2> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<Real3> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<Real2x2> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<Real3x3> send_buf, Integer rank) = 0;
  virtual Request broadcast(ArrayView<HPReal> send_buf, Integer rank) = 0;
  //virtual Request broadcastString(String& str,Integer rank) =0;

  //virtual Request broadcastSerializer(ISerializer* values,Integer rank) =0;
  /*! \brief Effectue un broadcast d'une zone mémoire.
   *
   * Le processeur qui effectue le broacast est donnée par \id. Le tableau
   * envoyé est alors donnée par \a bytes. Les processeurs réceptionnent
   * le tableau dans \a bytes. Ce tableau est alloué automatiquement, les processeurs
   * réceptionnant n'ont pas besoin de connaitre le nombre d'octets devant être envoyés.
   *
   */
  //virtual Request broadcastMemoryBuffer(ByteArray& bytes,Integer rank) =0;
  //@}

  virtual Request allToAll(ConstArrayView<char> send_buf, ArrayView<char> recv_buf,
                           Integer count) = 0;
  virtual Request allToAll(ConstArrayView<signed char> send_buf, ArrayView<signed char> recv_buf,
                           Integer count) = 0;
  virtual Request allToAll(ConstArrayView<unsigned char> send_buf, ArrayView<unsigned char> recv_buf,
                           Integer count) = 0;
  virtual Request allToAll(ConstArrayView<int> send_buf, ArrayView<int> recv_buf, Integer count) = 0;
  virtual Request allToAll(ConstArrayView<unsigned int> send_buf, ArrayView<unsigned int> recv_buf,
                           Integer count) = 0;
  virtual Request allToAll(ConstArrayView<short> send_buf, ArrayView<short> recv_buf, Integer count) = 0;
  virtual Request allToAll(ConstArrayView<unsigned short> send_buf, ArrayView<unsigned short> recv_buf,
                           Integer count) = 0;
  virtual Request allToAll(ConstArrayView<long> send_buf, ArrayView<long> recv_buf, Integer count) = 0;
  virtual Request allToAll(ConstArrayView<unsigned long> send_buf, ArrayView<unsigned long> recv_buf,
                           Integer count) = 0;
  virtual Request allToAll(ConstArrayView<long long> send_buf, ArrayView<long long> recv_buf,
                           Integer count) = 0;
  virtual Request allToAll(ConstArrayView<unsigned long long> send_buf,
                           ArrayView<unsigned long long> recv_buf, Integer count) = 0;
  virtual Request allToAll(ConstArrayView<float> send_buf, ArrayView<float> recv_buf,
                           Integer count) = 0;
  virtual Request allToAll(ConstArrayView<double> send_buf, ArrayView<double> recv_buf,
                           Integer count) = 0;
  virtual Request allToAll(ConstArrayView<long double> send_buf, ArrayView<long double> recv_buf,
                           Integer count) = 0;
#ifdef ARCANE_REAL_NOT_BUILTIN
  virtual Request allToAll(ConstArrayView<Real> send_buf, ArrayView<Real> recv_buf,
                           Integer count) = 0;
#endif
  virtual Request allToAll(ConstArrayView<Real2> send_buf, ArrayView<Real2> recv_buf,
                           Integer count) = 0;
  virtual Request allToAll(ConstArrayView<Real3> send_buf, ArrayView<Real3> recv_buf,
                           Integer count) = 0;
  virtual Request allToAll(ConstArrayView<Real2x2> send_buf, ArrayView<Real2x2> recv_buf,
                           Integer count) = 0;
  virtual Request allToAll(ConstArrayView<Real3x3> send_buf, ArrayView<Real3x3> recv_buf,
                           Integer count) = 0;
  virtual Request allToAll(ConstArrayView<HPReal> send_buf, ArrayView<HPReal> recv_buf,
                           Integer count) = 0;

  /*! @name allToAll variable
   *
   * \brief Effectue un allToAll variable
   *
   */
  //@{
  virtual Request allToAllVariable(ConstArrayView<char> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<char> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<signed char> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<signed char> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<unsigned char> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<unsigned char> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<int> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<int> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<unsigned int> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<unsigned int> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<short> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<short> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<unsigned short> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<unsigned short> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<long> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<long> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<unsigned long> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<unsigned long> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<long long> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<long long> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<unsigned long long> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<unsigned long long> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<float> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<float> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<double> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<double> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<long double> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<long double> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
#ifdef ARCANE_REAL_NOT_BUILTIN
  virtual Request allToAllVariable(ConstArrayView<Real> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<Real> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
#endif
  virtual Request allToAllVariable(ConstArrayView<Real2> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<Real2> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<Real3> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<Real3> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<Real2x2> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<Real2x2> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<Real3x3> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<Real3x3> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Request allToAllVariable(ConstArrayView<HPReal> send_buf, Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index, ArrayView<HPReal> recv_buf,
                                   Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  //@}

  //! @name opérations de synchronisation et opérations asynchrones
  //@{
  //! Effectue une barière
  virtual Request barrier() = 0;
  //@}

  /*!
   * \brief Indique si l'implémentation autorise les réductions sur les types
   * dérivés.
   *
   * Les version de OpenMPI jusqu'à la version 1.8.4 incluses semblent avoir
   * un bug (qui se traduit par un plantage) avec les réductions non bloquantes
   * lorsque l'opérateur de réduction est redéfini. C'est le cas avec les
   * types dérivés tels que Real3, Real2, ...
   */
  virtual bool hasValidReduceForDerivedType() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
